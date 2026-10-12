import fnmatch
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import requests


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_users(path: str = None) -> dict:
    if path is None:
        path = str(Path(__file__).parent / "users.json")
    with open(path) as f:
        return json.load(f)["users"]


def resolve_access(patterns: list[str], all_sources: list[str]) -> list[str]:
    if "*" in patterns:
        return all_sources
    allowed = set()
    for pat in patterns:
        for src in all_sources:
            if fnmatch.fnmatch(src, pat):
                allowed.add(src)
    return sorted(allowed)

HEADING_RE = re.compile(
    r"^(?:"
    r"(?:SECTION|PART|CHAPTER|ARTICLE|DIVISION)\s+[\d.]+.*"
    r"|[\d]+\.[\d.]*\s+[A-Z].*"
    r"|[A-Z][A-Z \-/&]{4,}$"
    r"|#{1,4}\s+.+"
    r")",
    re.MULTILINE,
)


def load_pdf_pages(path: str) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def extract_heading(text: str) -> str:
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 4:
            continue
        if HEADING_RE.match(line):
            return line[:120]
    first_lines = [l.strip() for l in text.split("\n") if l.strip()]
    if first_lines and len(first_lines[0]) < 100 and len(first_lines[0]) > 3:
        return first_lines[0][:120]
    return ""


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    chunks = []
    for page_num, page_text in pages:
        start = 0
        while start < len(page_text):
            end = start + chunk_size
            chunk = page_text[start:end].strip()
            if chunk:
                heading = extract_heading(chunk)
                chunks.append({
                    "text": chunk,
                    "page": page_num,
                    "section": heading,
                })
            start += chunk_size - overlap
    return chunks


class RagEngine:
    def __init__(self, persist_dir: str = "./chroma_db", export_dir: str = "./db_export", api_key: str = ""):
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set — add it to .streamlit/secrets.toml or .env")
        self._api_key = api_key
        self._embed_model = "gemini-embedding-001"
        self._gen_model = "gemini-3.8-flash"

        chroma_exists = Path(persist_dir).exists() and (Path(persist_dir) / "chroma.sqlite3").exists()
        export_exists = Path(export_dir).exists() and (Path(export_dir) / "data.json").exists()

        if chroma_exists:
            import chromadb
            self._mode = "chroma"
            self.chroma = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.chroma.get_or_create_collection(
                name="rpf_documents",
                metadata={"hnsw:space": "cosine"},
            )
        elif export_exists:
            self._mode = "numpy"
            data = json.load(open(Path(export_dir) / "data.json"))
            self._np_ids = data["ids"]
            self._np_docs = data["documents"]
            self._np_metas = data["metadatas"]
            emb_data = np.load(Path(export_dir) / "embeddings.npz")
            self._np_embeddings = emb_data["embeddings"].astype(np.float32)
            norms = np.linalg.norm(self._np_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self._np_embeddings_normed = self._np_embeddings / norms
        else:
            import chromadb
            self._mode = "chroma"
            self.chroma = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.chroma.get_or_create_collection(
                name="rpf_documents",
                metadata={"hnsw:space": "cosine"},
            )

    def entry_count(self) -> int:
        if self._mode == "chroma":
            return self.collection.count()
        return len(self._np_ids)

    def _gemini_request(self, endpoint: str, payload: dict) -> dict:
        url = f"{GEMINI_API_BASE}/{endpoint}?key={self._api_key}"
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            data = self._gemini_request(
                f"models/{self._embed_model}:embedContent",
                {
                    "content": {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                },
            )
            results.append(data["embedding"]["values"])
        return results

    def _embed_query(self, text: str) -> list[float]:
        data = self._gemini_request(
            f"models/{self._embed_model}:embedContent",
            {
                "content": {"parts": [{"text": text}]},
                "taskType": "RETRIEVAL_QUERY",
            },
        )
        return data["embedding"]["values"]

    def _generate(self, prompt: str, system_instruction: str = "") -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        data = self._gemini_request(
            f"models/{self._gen_model}:generateContent",
            payload,
        )
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def ingest(
        self,
        pdf_path: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> int:
        if self._mode != "chroma":
            raise RuntimeError("Ingestion only available in local mode")
        filename = Path(pdf_path).name
        pages = load_pdf_pages(pdf_path)
        if not pages:
            return 0
        chunks = chunk_pages(pages, chunk_size, overlap)

        ids = [f"{filename}::p{c['page']}_c{i}" for i, c in enumerate(chunks)]
        metadatas = [
            {
                "source": filename,
                "type": "text",
                "page": c["page"],
                "section": c["section"],
            }
            for c in chunks
        ]

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_texts = [c["text"] for c in chunks[i : i + batch_size]]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            embeddings = self._embed(batch_texts)

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_meta,
                embeddings=embeddings,
            )

        return len(chunks)

    @staticmethod
    def _is_useful_diagram(pix) -> bool:
        from PIL import Image, ImageStat
        import io

        try:
            png_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(png_bytes)).convert("L")
            stat = ImageStat.Stat(img)
            mean = stat.mean[0]
            stddev = stat.stddev[0]
            if mean < 25 or mean > 240:
                return False
            if stddev < 25:
                return False
            hist = img.histogram()
            total_px = img.width * img.height
            white_px = sum(hist[220:])
            black_px = sum(hist[:35])
            if white_px / total_px > 0.88 or black_px / total_px > 0.75:
                return False
            peak_bin = max(range(256), key=lambda i: hist[i])
            nearby = sum(hist[max(0, peak_bin - 15):min(256, peak_bin + 16)])
            if nearby / total_px > 0.7:
                return False
            aspect = max(pix.width, pix.height) / max(min(pix.width, pix.height), 1)
            if aspect > 12:
                return False
            return True
        except Exception:
            return False

    def ingest_images(
        self,
        pdf_path: str,
        images_dir: str = "./images",
        min_size: int = 150,
    ) -> int:
        if self._mode != "chroma":
            raise RuntimeError("Ingestion only available in local mode")
        import fitz
        filename = Path(pdf_path).name
        images_out = Path(images_dir)
        images_out.mkdir(exist_ok=True)

        doc = fitz.open(pdf_path)
        extracted = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                except Exception:
                    continue

                if pix.width < min_size or pix.height < min_size:
                    pix = None
                    continue

                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                if not self._is_useful_diagram(pix):
                    pix = None
                    continue

                img_filename = f"{Path(filename).stem}_p{page_num+1}_img{img_idx}.png"
                img_path = images_out / img_filename
                pix.save(str(img_path))

                image_bytes = img_path.read_bytes()
                import base64
                b64 = base64.b64encode(image_bytes).decode()
                payload = {
                    "contents": [{"parts": [
                        {"inlineData": {"mimeType": "image/png", "data": b64}},
                        {"text": (
                            "Look at this image extracted from a water/wastewater treatment PDF. "
                            "Classify it as ONE of:\n"
                            "- DIAGRAM: a technical diagram, schematic, process flow, P&ID, cross-section, "
                            "equipment drawing, graph, chart, or data table with visual elements\n"
                            "- JUNK: a solid color block, decorative element, logo, small icon, blank page, "
                            "scanned text-only page, header/footer graphic, or border element\n\n"
                            "Respond with ONLY one word: DIAGRAM or JUNK"
                        )},
                    ]}],
                }
                try:
                    data = self._gemini_request(f"models/{self._gen_model}:generateContent", payload)
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    if "DIAGRAM" not in text.strip().upper():
                        pix = None
                        continue
                except Exception:
                    pass
                time.sleep(0.3)

                try:
                    desc_prompt = (
                        "Describe this technical diagram/figure from a water or wastewater treatment document. "
                        "Include: what type of diagram it is (process flow, P&ID, schematic, cross-section, graph, table, etc.), "
                        "the equipment or process shown, key components labeled, flow directions, "
                        "and any numerical values or specifications visible. "
                        "Be thorough and technical — this description will be used for document search."
                        f"\n\nThis image is from: {filename}"
                    )
                    desc_payload = {
                        "contents": [{"parts": [
                            {"inlineData": {"mimeType": "image/png", "data": b64}},
                            {"text": desc_prompt},
                        ]}],
                    }
                    data = self._gemini_request(f"models/{self._gen_model}:generateContent", desc_payload)
                    description = data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    print(f"    Vision failed for {img_filename}: {e}")
                    pix = None
                    continue

                page_text = page.get_text() or ""
                section = extract_heading(page_text)

                doc_text = f"[DIAGRAM from {filename}, page {page_num+1}]\n{description}"
                chunk_id = f"{filename}::img_p{page_num+1}_{img_idx}"

                embeddings = self._embed([doc_text])
                self.collection.upsert(
                    ids=[chunk_id],
                    documents=[doc_text],
                    metadatas=[{
                        "source": filename,
                        "type": "image",
                        "page": page_num + 1,
                        "section": section,
                        "image_file": img_filename,
                    }],
                    embeddings=embeddings,
                )
                extracted += 1
                pix = None
                time.sleep(0.3)

        doc.close()
        return extracted

    def list_sources(self) -> list[str]:
        if self._mode == "chroma":
            results = self.collection.get(include=["metadatas"])
            metas = results["metadatas"] or []
        else:
            metas = self._np_metas
        sources = set()
        for meta in metas:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def _query_numpy(self, query_embedding, n_results, allowed_sources):
        qvec = np.array(query_embedding, dtype=np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm

        similarities = self._np_embeddings_normed @ qvec
        distances = 1 - similarities

        if allowed_sources is not None:
            allowed_set = set(allowed_sources)
            mask = np.array([
                m is not None and m.get("source") in allowed_set
                for m in self._np_metas
            ])
            distances[~mask] = 999.0

        top_k = min(n_results, len(distances))
        top_indices = np.argpartition(distances, top_k)[:top_k]
        top_indices = top_indices[np.argsort(distances[top_indices])]

        chunks = [self._np_docs[i] for i in top_indices if self._np_docs[i] is not None]
        metadatas = [self._np_metas[i] for i in top_indices if self._np_metas[i] is not None]
        dists = [float(distances[i]) for i in top_indices]

        return chunks[:n_results], metadatas[:n_results], dists[:n_results]

    def query(
        self,
        question: str,
        n_results: int = 5,
        allowed_sources: list[str] | None = None,
    ) -> dict:
        query_embedding = self._embed_query(question)

        if self._mode == "numpy":
            chunks, metadatas, distances = self._query_numpy(
                query_embedding, n_results, allowed_sources
            )
        else:
            fetch_n = n_results * 10 if allowed_sources else n_results
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(fetch_n, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            chunks = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            chunks_metas = [
                (doc, meta, dist)
                for doc, meta, dist in zip(chunks, metadatas, distances)
                if meta is not None
            ]
            chunks = [c[0] for c in chunks_metas]
            metadatas = [c[1] for c in chunks_metas]
            distances = [c[2] for c in chunks_metas]

            if allowed_sources is not None:
                allowed_set = set(allowed_sources)
                filtered = [
                    (doc, meta, dist)
                    for doc, meta, dist in zip(chunks, metadatas, distances)
                    if meta.get("source") in allowed_set
                ]
                filtered = filtered[:n_results]
                chunks = [f[0] for f in filtered]
                metadatas = [f[1] for f in filtered]
                distances = [f[2] for f in filtered]

        def _cite(m):
            section = m.get("section", "")
            page = m.get("page", "?")
            if section:
                return f"{section} (p.{page})"
            return f"Page {page}"

        context = "\n\n---\n\n".join(
            f"[Source: {m.get('source', 'unknown')}, {_cite(m)}]\n{doc}"
            for doc, m in zip(chunks, metadatas)
        )

        source_names = sorted({m.get("source", "") for m in metadatas if m.get("source")})
        source_list = ", ".join(source_names) if source_names else "various documents"

        system_prompt = (
            "You are an expert assistant for Ovivo's water and wastewater treatment documents, "
            "including RPF (Request for Proposal) specifications, EPA technical guidance, "
            "UFGS equipment specs, O&M manuals, and state design standards.\n\n"
            "If the user asks a general question like 'how can you help me', 'what can you do', "
            "or 'hello', respond warmly and explain that you can:\n"
            "- Answer technical questions about water and wastewater treatment\n"
            "- Look up EPA guidelines, design standards, and specifications\n"
            "- Find information from RFPs, O&M manuals, and equipment specs\n"
            "- Show relevant diagrams and process flow charts\n"
            "- Cite specific pages and sections from source documents\n"
            f"Mention that relevant documents were found from: {source_list}.\n\n"
            "For technical questions, answer based ONLY on the provided context. "
            "If the context does not contain enough information, say so clearly. "
            "Cite the source document with its section heading or page number. "
            "Some context chunks are descriptions of diagrams/figures — "
            "reference these when relevant and mention the source diagram. "
            "Use technical water/wastewater treatment terminology appropriately."
        )

        answer = self._generate(
            f"Context from documents:\n\n{context}\n\n---\n\nQuestion: {question}",
            system_instruction=system_prompt,
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "text": doc,
                    "source": m.get("source", "unknown"),
                    "page": m.get("page", "?"),
                    "section": m.get("section", ""),
                    "distance": round(d, 4),
                    "type": m.get("type", "text"),
                    "image_file": m.get("image_file", ""),
                }
                for doc, m, d in zip(chunks, metadatas, distances)
            ],
        }
