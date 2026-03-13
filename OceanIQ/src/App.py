import streamlit as st
import pandas as pd
import datetime

from embedder import get_model
from search import local_search
from rag import build_ai_summary, build_semantic_only_summary
from endee_client import EndeeClient
from research_sources import fetch_all_sources

st.set_page_config(
    page_title="OceanIQ · Research Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_QUESTIONS = [
    "What causes ocean acidification?",
    "How do ARGO floats work?",
    "Coral reef bleaching & climate",
    "What is sea level rise?",
    "Deep sea hydrothermal vents",
]


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

    :root {
        --bg:        #07090f;
        --surface:   #0d1220;
        --surface2:  #111827;
        --border:    #1c2a3e;
        --border2:   #243347;
        --accent:    #00d4ff;
        --accent2:   #0099cc;
        --accent3:   #005f80;
        --text:      #dde5f0;
        --muted:     #5a6e8a;
        --muted2:    #8499b8;
        --green:     #00c896;
        --blue:      #3b9eff;
        --yellow:    #f5a623;
        --red:       #e05252;
        --radius:    12px;
        --radius-sm: 7px;
    }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif;
    }

    #MainMenu, footer, header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }

    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        min-width: 270px !important;
        max-width: 270px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: var(--surface) !important;
        padding: 1.5rem 1.25rem 2rem !important;
    }
    [data-testid="stSidebarCollapsedControl"] button,
    button[data-testid="baseButton-headerNoPadding"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--accent) !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        max-width: 0 !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--muted2) !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] .stRadio > label { color: var(--text) !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 6px 8px;
        gap: 4px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: var(--muted2) !important;
        font-size: 13px !important;
        padding: 6px 10px;
        border-radius: 6px;
        cursor: pointer;
        transition: background 0.15s, color 0.15s;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: rgba(0,212,255,0.12);
        color: var(--accent) !important;
    }
    [data-testid="stSidebar"] .stCheckbox label {
        color: var(--muted2) !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] .stCheckbox svg { color: var(--accent) !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: var(--surface2) !important;
        color: var(--muted2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 12px !important;
        padding: 6px 10px !important;
        text-align: left !important;
        transition: all 0.18s ease !important;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--accent3) !important;
        color: var(--accent) !important;
        background: rgba(0,212,255,0.06) !important;
    }
    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(0,153,204,0.08)) !important;
        color: var(--accent) !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 12px !important;
        width: 100%;
    }

    [data-testid="stMain"] > div { padding-top: 1.5rem !important; }

    [data-testid="stChatInput"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(0,212,255,0.12) !important;
    }
    [data-testid="stChatInput"] button { color: var(--accent) !important; }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stChatMessageContent"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        line-height: 1.7;
    }

    [data-testid="stExpander"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        margin-bottom: 8px;
    }
    [data-testid="stExpander"] summary {
        color: var(--muted2) !important;
        font-size: 13px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }
    [data-testid="stExpander"] summary:hover { color: var(--text) !important; }

    [data-testid="stAlert"] {
        background: rgba(245,166,35,0.08) !important;
        border: 1px solid rgba(245,166,35,0.25) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--yellow) !important;
        font-size: 13px !important;
    }

    [data-testid="stMain"] .stButton > button {
        background: var(--surface2) !important;
        color: var(--muted2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        transition: all 0.18s ease !important;
    }
    [data-testid="stMain"] .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: rgba(0,212,255,0.06) !important;
        transform: translateY(-1px);
    }

    .sb-logo {
        font-family: 'Syne', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: var(--accent);
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .sb-tag {
        font-size: 11px;
        color: var(--muted);
        margin-bottom: 22px;
        letter-spacing: 0.3px;
    }
    .sb-section {
        font-family: 'Syne', sans-serif;
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: var(--muted);
        margin: 20px 0 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }
    .sb-status {
        font-size: 12px;
        padding: 6px 10px;
        border-radius: var(--radius-sm);
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .sb-status.online  { background: rgba(0,200,150,0.08); color: var(--green); border: 1px solid rgba(0,200,150,0.2); }
    .sb-status.offline { background: rgba(224,82,82,0.08); color: var(--red);   border: 1px solid rgba(224,82,82,0.2); }

    .r-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 14px 16px;
        margin-bottom: 8px;
        transition: border-color 0.2s, transform 0.15s;
    }
    .r-card:hover { border-color: var(--accent3); transform: translateY(-1px); }
    .r-title {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 14px;
        color: var(--text);
        margin-bottom: 5px;
        line-height: 1.3;
    }
    .r-text {
        font-size: 13px;
        color: var(--muted2);
        line-height: 1.65;
        margin-bottom: 9px;
    }
    .r-meta { font-size: 11.5px; color: var(--muted); margin-bottom: 7px; }

    .badge {
        display: inline-block;
        font-size: 10.5px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
        margin-right: 5px;
        letter-spacing: 0.2px;
    }
    .b-green  { background: rgba(0,200,150,0.12); color: var(--green); }
    .b-blue   { background: rgba(59,158,255,0.12); color: var(--blue); }
    .b-yellow { background: rgba(245,166,35,0.12); color: var(--yellow); }
    .b-red    { background: rgba(224,82,82,0.12);  color: var(--red); }
    .b-cyan   { background: rgba(0,212,255,0.10);  color: var(--accent); }

    .ai-box {
        background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(0,99,128,0.04));
        border: 1px solid rgba(0,212,255,0.18);
        border-radius: var(--radius);
        padding: 18px 20px;
        margin-top: 6px;
        font-size: 14.5px;
        line-height: 1.75;
        color: var(--text);
    }
    .ai-label {
        font-family: 'Syne', sans-serif;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 10px;
        opacity: 0.85;
    }

    .src-link {
        display: inline-block;
        font-size: 11.5px;
        color: var(--accent);
        border: 1px solid rgba(0,212,255,0.22);
        border-radius: 5px;
        padding: 3px 9px;
        text-decoration: none;
        transition: background 0.15s;
    }
    .src-link:hover { background: rgba(0,212,255,0.1); }

    .hero {
        text-align: center;
        padding: 64px 20px 44px;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 44px;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #00a3cc 50%, #00c896 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 14px;
        line-height: 1.08;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 16px;
        color: var(--muted2);
        max-width: 460px;
        margin: 0 auto 36px;
        line-height: 1.65;
        font-weight: 300;
    }
    .hero-pills {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }
    .hero-pill {
        font-size: 11px;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid var(--border2);
        color: var(--muted2);
        background: var(--surface2);
        letter-spacing: 0.3px;
    }
    </style>
    """, unsafe_allow_html=True)


def init_state():
    defaults = {
        "messages":        [],
        "search_history":  [],
        "last_results":    [],
        "mode":            "🔍 Semantic Search",
        "sources_wiki":    True,
        "sources_scholar": True,
        "sources_pubmed":  True,
        "sources_patents": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sb-logo">🌊 OceanIQ</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-tag">Ocean Research Discovery Platform</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Search Mode</div>', unsafe_allow_html=True)
        mode_idx = 0 if st.session_state.mode == "🔍 Semantic Search" else 1
        st.session_state.mode = st.radio(
            "mode_radio",
            ["🔍 Semantic Search", "🤖 AI Summary"],
            index=mode_idx,
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-section">Web Sources</div>', unsafe_allow_html=True)
        st.session_state.sources_wiki    = st.checkbox("🌐 Wikipedia",        value=st.session_state.sources_wiki,    key="cb_wiki")
        st.session_state.sources_scholar = st.checkbox("📄 Semantic Scholar", value=st.session_state.sources_scholar, key="cb_scholar")
        st.session_state.sources_pubmed  = st.checkbox("🧬 PubMed Journals",  value=st.session_state.sources_pubmed,  key="cb_pubmed")
        st.session_state.sources_patents = st.checkbox("💡 Google Patents",    value=st.session_state.sources_patents, key="cb_patents")

        st.markdown('<div class="sb-section">Vector Index</div>', unsafe_allow_html=True)
        try:
            EndeeClient()
            st.markdown('<div class="sb-status online">🟢 Endee connected</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="sb-status offline">🔴 Endee offline</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Recent Searches</div>', unsafe_allow_html=True)
        history = st.session_state.search_history[-8:][::-1]
        if not history:
            st.markdown('<span style="font-size:12px;color:var(--muted);">No searches yet</span>', unsafe_allow_html=True)
        else:
            for q in history:
                label = f"↩ {q[:36]}…" if len(q) > 36 else f"↩ {q}"
                if st.button(label, key=f"hist_{q}", use_container_width=True):
                    _do_search(q)
                    st.rerun()
            if st.button("🗑 Clear history", key="clear_hist", use_container_width=True):
                st.session_state.search_history = []
                st.rerun()

        if st.session_state.last_results:
            st.markdown('<div class="sb-section">Export</div>', unsafe_allow_html=True)
            rows = [
                {
                    "title":      r.get("meta", {}).get("title", ""),
                    "text":       r.get("meta", {}).get("text", "")[:200],
                    "source":     r.get("meta", {}).get("source", ""),
                    "similarity": r.get("similarity", 0),
                }
                for r in st.session_state.last_results
            ]
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "⬇ Export CSV",
                pd.DataFrame(rows).to_csv(index=False),
                file_name=f"oceaniq_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_csv",
            )


def _badge(score: float) -> str:
    if score >= 0.75:
        return f'<span class="badge b-green">Strong {score:.2f}</span>'
    elif score >= 0.60:
        return f'<span class="badge b-blue">Good {score:.2f}</span>'
    elif score >= 0.45:
        return f'<span class="badge b-yellow">Weak {score:.2f}</span>'
    else:
        return f'<span class="badge b-red">Low {score:.2f}</span>'


def render_local_card(r, idx):
    meta  = r.get("meta", {})
    sim   = r.get("similarity", 0)
    topic = meta.get("topic", "")
    st.markdown(f"""
    <div class="r-card">
        <div class="r-title">{idx}. {meta.get('title', 'Unknown')}</div>
        <div class="r-text">{meta.get('text', '')[:260]}…</div>
        {_badge(sim)}
        {'<span class="badge b-cyan">🏷 ' + topic + '</span>' if topic else ''}
        <span class="badge b-blue">📄 {meta.get('source', 'Local Index')}</span>
    </div>""", unsafe_allow_html=True)


def render_web_card(item, source_type, idx):
    title   = item.get("title", "Untitled")
    url     = item.get("url") or item.get("link") or ""
    snippet = (item.get("summary") or item.get("abstract") or "")[:230]
    authors = item.get("authors", "")
    year    = item.get("year", "")

    meta_line = (
        f'<div class="r-meta">{authors}'
        f'{"&nbsp;·&nbsp;" + str(year) if year else ""}</div>'
    ) if authors else ""

    link_html = (
        f'<a class="src-link" href="{url}" target="_blank">↗ Open on {source_type}</a>'
    ) if url else ""

    st.markdown(f"""
    <div class="r-card">
        <div class="r-title">{idx}. {title}</div>
        {meta_line}
        <div class="r-text">{snippet}…</div>
        {link_html}
    </div>""", unsafe_allow_html=True)


def render_assistant(data: dict):
    mode          = data.get("mode", "🔍 Semantic Search")
    answer        = data.get("answer", "")
    low_conf      = data.get("low_confidence", False)
    local_results = data.get("local_results", [])
    web_results   = data.get("web_results", {})
    query         = data.get("query", "")

    with st.chat_message("assistant", avatar="🌊"):
        if low_conf:
            st.warning("⚠️ Low confidence in results — try rephrasing your query.")
        else:
            label = "🤖 AI Summary" if "🤖" in mode else "🔍 Semantic Answer"
            st.markdown(f"""
            <div class="ai-box">
                <div class="ai-label">{label}</div>
                {answer}
            </div>""", unsafe_allow_html=True)

        if local_results:
            with st.expander(f"📂 Local Index — {len(local_results)} result(s)"):
                for i, r in enumerate(local_results, 1):
                    render_local_card(r, i)

        wiki = web_results.get("wikipedia", [])
        if wiki:
            with st.expander(f"🌐 Wikipedia — {len(wiki)} article(s)", expanded=True):
                for i, item in enumerate(wiki, 1):
                    render_web_card(item, "Wikipedia", i)

        scholar = web_results.get("semantic_scholar", [])
        if scholar:
            with st.expander(f"📄 Research Papers — {len(scholar)} result(s)", expanded=True):
                for i, item in enumerate(scholar, 1):
                    render_web_card(item, "Semantic Scholar", i)

        pubmed = web_results.get("pubmed", [])
        if pubmed:
            with st.expander(f"🧬 PubMed Journals — {len(pubmed)} result(s)"):
                for i, item in enumerate(pubmed, 1):
                    render_web_card(item, "PubMed", i)

        patent = web_results.get("patents", {})
        if patent and isinstance(patent, dict) and patent.get("url"):
            st.markdown(
                f'<a class="src-link" href="{patent["url"]}" target="_blank">'
                f'💡 Search Google Patents for "{query}"</a>',
                unsafe_allow_html=True,
            )


def _do_search(query: str):
    query = query.strip()
    if not query:
        return

    if (st.session_state.messages
            and st.session_state.messages[-1].get("role") == "user"
            and st.session_state.messages[-1].get("content") == query):
        return

    get_model()

    local_results = local_search(query, k=5)
    st.session_state.last_results = local_results

    source_flags = {
        "wikipedia":        st.session_state.sources_wiki,
        "semantic_scholar": st.session_state.sources_scholar,
        "pubmed":           st.session_state.sources_pubmed,
        "patents":          st.session_state.sources_patents,
    }

    try:
        web_results = fetch_all_sources(query, source_flags)
    except Exception:
        web_results = {}

    mode = st.session_state.mode
    if "🤖" in mode:
        answer = build_ai_summary(query, local_results, web_results)
    else:
        answer = build_semantic_only_summary(query, local_results)

    top_sim = local_results[0].get("similarity", 0) if local_results else 0
    low_confidence = top_sim < 0.35 and not any(
        v for v in web_results.values() if isinstance(v, list) and v
    )

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({
        "role": "assistant",
        "data": {
            "query":          query,
            "mode":           mode,
            "answer":         answer,
            "low_confidence": low_confidence,
            "local_results":  local_results,
            "web_results":    web_results,
        },
    })

    if query not in st.session_state.search_history:
        st.session_state.search_history.append(query)


def main():
    inject_css()
    init_state()
    render_sidebar()

    if not st.session_state.messages:
        st.markdown("""
        <div class="hero">
            <div class="hero-title">OceanIQ Research<br>Assistant</div>
            <div class="hero-sub">
                Discover papers, Wikipedia articles, journals &amp; patents
                about ocean science — powered by Endee vector search.
            </div>
            <div class="hero-pills">
                <span class="hero-pill">🔬 Semantic Search</span>
                <span class="hero-pill">📚 RAG Retrieval</span>
                <span class="hero-pill">🌐 Live Web Sources</span>
                <span class="hero-pill">🧬 PubMed · Scholar</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Try a sample question")
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for i, q in enumerate(SAMPLE_QUESTIONS):
            with cols[i]:
                if st.button(q, key=f"chip_{i}", use_container_width=True):
                    _do_search(q)
                    st.rerun()
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                render_assistant(msg.get("data", {}))

        with st.expander("💡 Sample questions"):
            cols = st.columns(len(SAMPLE_QUESTIONS))
            for i, q in enumerate(SAMPLE_QUESTIONS):
                with cols[i]:
                    if st.button(q, key=f"chip2_{i}", use_container_width=True):
                        _do_search(q)
                        st.rerun()

    user_input = st.chat_input("Ask about oceans, marine science, climate…")
    if user_input:
        _do_search(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
