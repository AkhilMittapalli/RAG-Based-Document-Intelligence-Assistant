import os
import tempfile
from pathlib import Path

from PIL import Image, ImageStat
import streamlit as st
from dotenv import load_dotenv

from rag_engine import RagEngine, load_users, resolve_access

load_dotenv()

# On Streamlit Cloud, load API key from secrets
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"].strip()
except FileNotFoundError:
    pass

CLOUD_MODE = not Path("./chroma_db/chroma.sqlite3").exists()

st.set_page_config(
    page_title="Ovivo Document Intelligence",
    page_icon="https://www.ovivosolutions.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="auto",
)

IMAGES_DIR = Path("./images")
VISION_FILE = Path("./eval_vision.json")
VISION_RESULTS = {}
if VISION_FILE.exists():
    import json as _json
    VISION_RESULTS = _json.load(open(VISION_FILE))
USERS = load_users()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --brand-blue: #0ea5e9;
    --brand-dark: #0c4a6e;
    --surface: #0f1923;
    --surface-raised: #162231;
    --surface-overlay: #1c2d3f;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-green: #22c55e;
    --accent-amber: #f59e0b;
    --border: #1e3a50;
    --radius: 12px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---- Global ---- */
.stApp {
    background: linear-gradient(160deg, #0a1628 0%, #0f1f33 50%, #0a1628 100%);
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--border) !important;
}
.stButton > button:hover {
    border-color: var(--brand-blue) !important;
    box-shadow: 0 0 20px rgba(14, 165, 233, 0.15) !important;
}

/* Form submit button - brand blue */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
}
.stFormSubmitButton > button:hover {
    box-shadow: 0 6px 25px rgba(14, 165, 233, 0.45) !important;
    transform: translateY(-2px) !important;
}
.stFormSubmitButton > button:active {
    transform: translateY(0px) !important;
}

/* ---- Chat messages ---- */
.stChatMessage {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    padding: 1.2rem !important;
    margin-bottom: 1rem !important;
}
.stChatMessage[data-testid="stChatMessage-user"] {
    background: var(--surface-overlay) !important;
}
.stChatMessage[data-testid="stChatMessage-assistant"] {
    background: var(--surface-raised) !important;
    border-left: 3px solid var(--brand-blue) !important;
}

/* ---- Chat input ---- */
.stChatInput > div {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface-raised) !important;
}
.stChatInput > div:focus-within {
    border-color: var(--brand-blue) !important;
    box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background: var(--surface-raised) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ---- Images in diagrams ---- */
.stImage {
    border-radius: 8px !important;
    overflow: hidden !important;
}
.stImage img {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    transition: transform 0.2s ease !important;
}
.stImage img:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3) !important;
}

/* ---- Code blocks for sources ---- */
.stCodeBlock {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}

/* ---- Text input fields ---- */
.stTextInput > div > div {
    border-radius: 8px !important;
}

/* ---- File uploader ---- */
.stFileUploader > div {
    border-radius: var(--radius) !important;
    border: 1px dashed var(--border) !important;
}

/* ---- Slider ---- */
.stSlider > div > div > div > div {
    background: var(--brand-blue) !important;
}

/* ---- Dividers ---- */
hr {
    border-color: var(--border) !important;
    opacity: 0.5;
}

/* ---- Login card ---- */
.login-card {
    background: linear-gradient(180deg, var(--surface-raised) 0%, rgba(22, 34, 49, 0.8) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 2.5rem 2rem;
    margin: 2rem auto;
    max-width: 420px;
    box-shadow: 0 25px 80px rgba(0, 0, 0, 0.4), 0 0 40px rgba(14, 165, 233, 0.05);
    position: relative;
    overflow: hidden;
}
.login-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #0ea5e9, #38bdf8, transparent);
}
.login-header {
    text-align: center;
    margin-bottom: 2rem;
}
.login-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #0ea5e9, #38bdf8, #7dd3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}
.login-header p {
    color: var(--text-secondary);
    font-size: 0.9rem;
    letter-spacing: 0.02em;
}

/* ---- Hero / branding ---- */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(56, 189, 248, 0.08));
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.8rem;
    color: var(--brand-blue);
    font-weight: 500;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-bottom: 0;
}

/* ---- Stat cards ---- */
.stat-row {
    display: flex;
    gap: 12px;
    margin: 1.5rem 0;
}
.stat-card {
    flex: 1;
    background: var(--surface-overlay);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--brand-blue);
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- Capability pills ---- */
.cap-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 1rem 0;
}
.cap-pill {
    background: var(--surface-overlay);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
}
.cap-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.cap-icon.blue   { background: rgba(14, 165, 233, 0.15); }
.cap-icon.green  { background: rgba(34, 197, 94, 0.15); }
.cap-icon.amber  { background: rgba(245, 158, 11, 0.15); }
.cap-icon.purple { background: rgba(168, 85, 247, 0.15); }

/* ---- Diagram gallery ---- */
.diagram-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 1rem 0 0.75rem;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.diagram-header-text {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- Source cards ---- */
.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(14, 165, 233, 0.1);
    color: var(--brand-blue);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* ---- Demo accounts table ---- */
.demo-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-top: 1.5rem;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.demo-table th {
    text-align: left;
    padding: 10px 14px;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: var(--surface-overlay);
    border-bottom: 1px solid var(--border);
}
.demo-table td {
    padding: 10px 14px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    border-bottom: 1px solid rgba(30, 58, 80, 0.3);
    background: var(--surface-raised);
}
.demo-table tr:last-child td {
    border-bottom: none;
}
.demo-table code {
    background: var(--surface-overlay);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    color: var(--brand-blue);
}

/* ---- Powered-by footer ---- */
.powered-by {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    letter-spacing: 0.02em;
}

/* ---- Sidebar user avatar ---- */
.user-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--surface-raised);
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 1rem;
}
.user-avatar {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
}
.user-name {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.2;
}
.user-role {
    color: var(--text-muted);
    font-size: 0.8rem;
}

/* ---- Sidebar section headers ---- */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.5rem;
}

/* ---- Document list in sidebar ---- */
.doc-count {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: var(--surface-raised);
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-bottom: 0.5rem;
}
.doc-count-num {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent-green);
}
.doc-count-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
}
</style>
""", unsafe_allow_html=True)


def format_source_name(filename: str) -> str:
    name = filename.replace(".pdf", "").replace("_", " ")
    return name.title()


def format_citation(src: dict) -> str:
    name = format_source_name(src["source"])
    section = src.get("section", "")
    page = src.get("page", "?")
    if section:
        return f"{name} -- {section} (p.{page})"
    return f"{name} -- Page {page}"


@st.cache_resource
def get_engine():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    return RagEngine(persist_dir="./chroma_db", export_dir="./db_export", api_key=api_key)


engine = get_engine()

# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.markdown("")
    st.markdown("")

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="login-card">
            <div class="login-header">
                <h1>Ovivo</h1>
                <p>Document Intelligence Platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.user = username
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown("")
        st.markdown("""
        <table class="demo-table">
            <tr><th>Username</th><th>Password</th><th>Role</th></tr>
            <tr><td><code>admin</code></td><td><code>admin123</code></td><td>Administrator</td></tr>
            <tr><td><code>bid_analyst</code></td><td><code>bid123</code></td><td>Bid Analyst</td></tr>
            <tr><td><code>field_tech</code></td><td><code>field123</code></td><td>Field Technician</td></tr>
            <tr><td><code>pfas_engineer</code></td><td><code>pfas123</code></td><td>PFAS Sales Engineer</td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="powered-by">
            Powered by RAG &middot; Gemini AI
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Logged in
# ---------------------------------------------------------------------------
user_info = USERS[st.session_state.user]
all_sources = engine.list_sources()
user_sources = resolve_access(user_info["access"], all_sources)
initials = "".join(w[0] for w in user_info["name"].split()[:2]).upper()

# ---- Sidebar ----
with st.sidebar:
    st.markdown(f"""
    <div class="user-card">
        <div class="user-avatar">{initials}</div>
        <div>
            <div class="user-name">{user_info['name']}</div>
            <div class="user-role">{user_info['role']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="sidebar-section">Documents</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="doc-count">
        <div class="doc-count-num">{len(user_sources)}</div>
        <div class="doc-count-label">of {len(all_sources)} documents<br>accessible</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Browse your sources"):
        for src in user_sources:
            st.markdown(f"<span style='color:var(--text-secondary);font-size:0.85rem'>{format_source_name(src)}</span>", unsafe_allow_html=True)

    if "*" not in user_info["access"]:
        restricted = set(all_sources) - set(user_sources)
        if restricted:
            with st.expander(f"{len(restricted)} restricted"):
                for src in sorted(restricted):
                    st.markdown(f"<span style='color:var(--text-muted);font-size:0.85rem'>&#128274; {format_source_name(src)}</span>", unsafe_allow_html=True)

    if not CLOUD_MODE:
        st.markdown('<div class="sidebar-section">Upload</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Add PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("Ingest Documents", use_container_width=True):
            for uploaded in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                with st.spinner(f"Processing {uploaded.name}..."):
                    n_chunks = engine.ingest(tmp_path)
                    n_images = engine.ingest_images(tmp_path, str(IMAGES_DIR))
                    st.success(f"{uploaded.name}: {n_chunks} chunks, {n_images} diagrams")
                os.unlink(tmp_path)

        if st.session_state.user == "admin" and st.button("Ingest All from data/", use_container_width=True):
            data_dir = Path("./data")
            if data_dir.exists():
                pdf_files = list(data_dir.glob("*.pdf"))
                if pdf_files:
                    progress = st.progress(0)
                    for i, pdf in enumerate(pdf_files):
                        with st.spinner(f"Processing {pdf.name}..."):
                            n_chunks = engine.ingest(str(pdf))
                            n_images = engine.ingest_images(str(pdf), str(IMAGES_DIR))
                        progress.progress((i + 1) / len(pdf_files))
                    st.success(f"Ingested {len(pdf_files)} documents")

    st.markdown('<div class="sidebar-section">Settings</div>', unsafe_allow_html=True)
    n_results = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

# ---- Main header ----
st.markdown(f"""
<div class="hero-badge">RAG-Powered Document Intelligence</div>
<div class="hero-title">Ovivo Document Assistant</div>
<div class="hero-subtitle">
    Signed in as <strong>{user_info['name']}</strong> &middot;
    {len(user_sources)} of {len(all_sources)} documents accessible
</div>
""", unsafe_allow_html=True)
st.markdown("")

# ---- Q&A ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(f"Welcome, **{user_info['name']}**! I'm your Ovivo document assistant.")
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-value">{len(user_sources)}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(all_sources)}</div>
                <div class="stat-label">Total Library</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{engine.entry_count():,}</div>
                <div class="stat-label">Indexed Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="cap-grid">
            <div class="cap-pill"><div class="cap-icon blue">&#128269;</div> Search EPA guidelines &amp; standards</div>
            <div class="cap-pill"><div class="cap-icon green">&#128196;</div> Find info in RFPs &amp; O&amp;M manuals</div>
            <div class="cap-pill"><div class="cap-icon amber">&#128202;</div> Show technical diagrams &amp; P&amp;IDs</div>
            <div class="cap-pill"><div class="cap-icon purple">&#128209;</div> Cite pages &amp; sections precisely</div>
        </div>
        """, unsafe_allow_html=True)
        role_prompts = {
            "admin": (
                '"What are the design criteria for clarifiers?"',
                '"Show me UV disinfection process diagrams."',
            ),
            "Bid Analyst": (
                '"What are the design criteria for clarifiers?"',
                '"Summarize the RFP requirements for tertiary treatment."',
            ),
            "Field Technician": (
                '"What is the startup procedure for the UV system?"',
                '"Show me the process flow diagram for the treatment plant."',
            ),
            "PFAS Sales Engineer": (
                '"What are the EPA-recommended PFAS treatment technologies?"',
                '"Compare GAC vs ion exchange for PFAS removal."',
            ),
        }
        q1, q2 = role_prompts.get(user_info["role"], role_prompts["admin"])
        st.markdown(f'Try: *{q1}* or *{q2}*')


def _is_displayable(img_path: Path) -> bool:
    if VISION_RESULTS:
        entry = VISION_RESULTS.get(img_path.name)
        if entry is not None:
            return entry.get("is_diagram", False)
    try:
        img = Image.open(img_path).convert("L")
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
        return True
    except Exception:
        return False


def render_response(result):
    st.markdown(result["answer"])

    diagram_sources = [
        s for s in result["sources"]
        if s.get("type") == "image"
        and (IMAGES_DIR / s.get("image_file", "")).exists()
        and _is_displayable(IMAGES_DIR / s.get("image_file", ""))
    ]
    if diagram_sources:
        st.markdown("""
        <div class="diagram-header">
            <span style="font-size:1.1rem">&#128200;</span>
            <span class="diagram-header-text">Referenced Diagrams</span>
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(min(len(diagram_sources), 3))
        for idx, src in enumerate(diagram_sources):
            img_file = IMAGES_DIR / src.get("image_file", "")
            with cols[idx % len(cols)]:
                st.image(str(img_file), use_column_width=True)
                st.caption(format_citation(src))

    with st.expander("View sources & citations"):
        for src in result["sources"]:
            citation = format_citation(src)
            similarity = (1 - src["distance"]) * 100
            badge_color = "#22c55e" if similarity > 80 else "#f59e0b" if similarity > 60 else "#94a3b8"

            st.markdown(
                f'**{citation}** '
                f'<span class="source-badge" style="background:rgba({badge_color}, 0.12);color:{badge_color}">'
                f'{similarity:.0f}% match</span>',
                unsafe_allow_html=True,
            )

            if src.get("type") == "image":
                img_file = IMAGES_DIR / src.get("image_file", "")
                if img_file.exists():
                    st.image(str(img_file), width=300)
            st.code(src["text"][:400], language=None)
            st.markdown("---")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("result"):
            render_response(msg["result"])
        else:
            st.markdown(msg["content"])

question = st.chat_input("Ask about water treatment, EPA guidelines, RFPs, diagrams...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not user_sources:
        with st.chat_message("assistant"):
            st.warning("You don't have access to any documents. Contact your administrator.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "No document access."}
        )
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and generating answer..."):
                    result = engine.query(
                        question,
                        n_results=n_results,
                        allowed_sources=user_sources,
                    )
            except Exception as e:
                st.error(f"Error querying: {type(e).__name__}: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )
                st.stop()

            if not result["sources"]:
                st.warning(
                    "No relevant information found in your accessible documents. "
                    "The answer may be in documents you don't have access to."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": "No relevant information found."}
                )
            else:
                render_response(result)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "result": result,
                    }
                )
