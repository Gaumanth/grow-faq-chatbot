"""
Groww Mutual Fund FAQ Chatbot — Streamlit UI
RAG-based facts-only assistant for HDFC Mutual Fund schemes.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import get_or_build_index, build_index, query_rag

load_dotenv()

st.set_page_config(
    page_title="Groww MF FAQ Assistant",
    page_icon="https://groww.in/favicon32.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Groww-Themed CSS ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container { padding-top: 2rem; }

    /* Hide default Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1520 0%, #111d2e 100%);
        border-right: 1px solid #1e2d3d;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #a0aec0;
        font-size: 0.88em;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e6edf3;
    }

    /* ── Header Banner ── */
    .groww-header {
        background: linear-gradient(135deg, #00d09c 0%, #00b386 50%, #009e73 100%);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .groww-header::before {
        content: '';
        position: absolute;
        top: -30px; right: -30px;
        width: 120px; height: 120px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .groww-header::after {
        content: '';
        position: absolute;
        bottom: -20px; left: 40%;
        width: 80px; height: 80px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .groww-header h1 {
        color: #fff;
        font-size: 1.6em;
        font-weight: 700;
        margin: 0 0 6px 0;
        position: relative;
        z-index: 1;
    }
    .groww-header p {
        color: rgba(255,255,255,0.85);
        font-size: 0.92em;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    /* ── Disclaimer ── */
    .disclaimer-card {
        background: #1a2332;
        border: 1px solid #2a3a4a;
        border-left: 4px solid #f0b429;
        padding: 14px 18px;
        border-radius: 10px;
        font-size: 0.84em;
        color: #c9d1d9;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .disclaimer-card b { color: #f0b429; }

    /* ── Example Question Cards ── */
    .example-label {
        color: #8b949e;
        font-size: 0.82em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    div.stButton > button {
        background: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.84em;
        font-weight: 500;
        text-align: left;
        transition: all 0.2s ease;
        line-height: 1.4;
    }
    div.stButton > button:hover {
        background: #1a2332;
        border-color: #00d09c;
        color: #00d09c;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 208, 156, 0.15);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        max-width: 100%;
        border-radius: 12px;
    }
    div[data-testid="stChatMessageContent"] {
        font-size: 0.94em;
        line-height: 1.6;
    }

    /* ── Hide garbled avatar text, show clean circles ── */
    .stChatMessage .stMarkdown:first-child ~ div,
    div[data-testid="chatAvatarIcon-user"],
    div[data-testid="chatAvatarIcon-assistant"],
    img[data-testid="chatAvatarIcon-user"],
    img[data-testid="chatAvatarIcon-assistant"] {
        visibility: hidden;
        position: relative;
    }
    .stChatMessage > div:first-child {
        width: 32px !important;
        min-width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        overflow: hidden !important;
        font-size: 0px !important;
        color: transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
    }
    /* User avatar: grey with U */
    [data-testid="stChatMessage-user"] > div:first-child {
        background: #30363d !important;
    }
    [data-testid="stChatMessage-user"] > div:first-child::after {
        content: "U";
        visibility: visible;
        font-size: 13px !important;
        color: #e6edf3 !important;
        font-weight: 600;
    }
    /* Assistant avatar: green with G */
    [data-testid="stChatMessage-assistant"] > div:first-child {
        background: #00d09c !important;
    }
    [data-testid="stChatMessage-assistant"] > div:first-child::after {
        content: "G";
        visibility: visible;
        font-size: 13px !important;
        color: #0d1117 !important;
        font-weight: 700;
    }


    /* ── Source Chips ── */
    .source-chip {
        display: inline-block;
        background: #1a2332;
        border: 1px solid #30363d;
        color: #00d09c;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78em;
        margin: 3px 4px 3px 0;
        text-decoration: none;
        transition: all 0.2s;
    }
    .source-chip:hover {
        background: #00d09c;
        color: #0d1117;
        text-decoration: none;
    }
    .sources-label {
        color: #8b949e;
        font-size: 0.78em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .source-divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 12px 0 8px 0;
    }

    /* ── Sidebar Scope Card ── */
    .scope-card {
        background: #131d2a;
        border: 1px solid #1e2d3d;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 8px 0;
    }
    .scope-card .scope-title {
        color: #00d09c;
        font-size: 0.78em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .scope-card .scope-amc {
        color: #e6edf3;
        font-weight: 600;
        font-size: 0.92em;
        margin-bottom: 8px;
    }
    .scope-card ul {
        margin: 0; padding-left: 18px;
    }
    .scope-card li {
        color: #8b949e;
        font-size: 0.82em;
        margin-bottom: 3px;
    }
    .scope-card .scheme-tag {
        display: inline-block;
        background: #0d1117;
        color: #00d09c;
        font-size: 0.72em;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 4px;
        font-weight: 600;
    }

    /* ── Sidebar Disclaimer ── */
    .sidebar-disclaimer {
        background: #131d2a;
        border: 1px solid #1e2d3d;
        border-left: 3px solid #f0b429;
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 0.8em;
        color: #8b949e;
        line-height: 1.5;
    }
    .sidebar-disclaimer b { color: #f0b429; }

    /* ── Chat Input ── */
    .stChatInput > div {
        border-color: #30363d;
        border-radius: 12px;
    }

    /* ── Sidebar Buttons ── */
    section[data-testid="stSidebar"] div.stButton > button {
        background: transparent;
        border: 1px solid #30363d;
        color: #8b949e;
        border-radius: 8px;
        font-size: 0.85em;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        border-color: #00d09c;
        color: #00d09c;
        background: rgba(0,208,156,0.05);
        transform: none;
        box-shadow: none;
    }

    /* ── Hide sidebar completely ── */
    section[data-testid="stSidebar"],
    button[data-testid="stBaseButton-headerNoPadding"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

</style>
""", unsafe_allow_html=True)

# ── API Key (hardcoded, no sidebar) ──────────────────────────────

api_key = os.getenv("GOOGLE_API_KEY", "")

# ── Header Banner ────────────────────────────────────────────────

st.markdown("""
<div class="groww-header">
    <h1>Mutual Fund FAQ Assistant</h1>
    <p>Ask factual questions about HDFC Mutual Fund schemes on Groww — expense ratios,
    exit loads, SIP minimums, lock-in periods, riskometer, and more.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-card">
    <b>Facts-only. No investment advice.</b>
    This chatbot answers factual questions only, using data from official public pages.
    Every answer includes a source link. It will not give buy/sell/portfolio recommendations.
</div>
""", unsafe_allow_html=True)

# ── Load / Build Index ───────────────────────────────────────────

if "index" not in st.session_state:
    with st.spinner("Building knowledge index (first run takes ~1 min)..."):
        try:
            index, chunks = get_or_build_index(api_key)
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
        except FileNotFoundError:
            st.error(
                "**No scraped data found.** Run the scraper first:\n\n"
                "```\npython scraper.py\n```\n\nThen reload this page."
            )
            st.stop()
        except Exception as e:
            st.error(f"Error loading index: {e}")
            st.stop()

# ── Example Questions ────────────────────────────────────────────

EXAMPLES = [
    "What is the expense ratio of HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver?",
    "How do I download my mutual fund statement on Groww?",
]

st.markdown('<div class="example-label">Try asking</div>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.setdefault("messages", [])
        st.session_state["messages"].append({"role": "user", "content": example})

# ── Chat History ─────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ───────────────────────────────────────────────────

if prompt := st.chat_input("Ask about HDFC Mutual Funds on Groww..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# ── Process Latest Message ───────────────────────────────────────

messages = st.session_state["messages"]
if messages and messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching..."):
                result = query_rag(
                    messages[-1]["content"],
                    st.session_state["index"],
                    st.session_state["chunks"],
                    api_key,
                )
            st.markdown(result["answer"])

            if result["sources"]:
                source_chips = ""
                for url in result["sources"]:
                    domain = url.split("//")[-1].split("/")[0].replace("www.", "")
                    source_chips += f'<a href="{url}" target="_blank" class="source-chip">{domain}</a> '

                st.markdown(
                    f'<hr class="source-divider">'
                    f'<div class="sources-label">Sources</div>'
                    f'{source_chips}',
                    unsafe_allow_html=True,
                )

            st.session_state["messages"].append(
                {"role": "assistant", "content": result["answer"]}
            )
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                st.warning(
                    "API rate limit reached. The free Gemini tier has daily limits. "
                    "Please wait a minute and try again, or check your quota at "
                    "https://ai.dev/rate-limit"
                )
            else:
                st.error(f"Something went wrong: {err_msg}")
