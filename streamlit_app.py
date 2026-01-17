import base64
import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src import ask_api

FLAG_PATH = Path(__file__).resolve().parent / "Flag_of_Tunisia.svg"


def load_svg_base64(path: Path) -> str:
    try:
        svg_bytes = path.read_bytes()
    except Exception:
        return ""
    return base64.b64encode(svg_bytes).decode("ascii")


FLAG_SVG_B64 = load_svg_base64(FLAG_PATH)


st.set_page_config(
    page_title="Narratives around the Tunisian Revolution",
    page_icon="N",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
  --paper: #f8f1e6;
  --ink: #1c1916;
  --accent: #c05621;
  --accent-2: #1f4f59;
  --accent-3: #a23e2c;
  --card: rgba(255, 255, 255, 0.92);
  --line: rgba(28, 25, 22, 0.12);
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 600px at 8% -10%, rgba(192, 86, 33, 0.12), transparent 60%),
    radial-gradient(900px 520px at 90% 0%, rgba(31, 79, 89, 0.16), transparent 55%),
    linear-gradient(120deg, #f8f1e6 0%, #f3e2cb 40%, #eef4f6 100%);
}
.block-container {
  padding-top: 1.5rem;
  color: var(--ink);
  font-family: "Baskerville", "Garamond", "Times New Roman", serif;
}
.app-shell {
  position: relative;
  padding-bottom: 1rem;
}
.app-header {
  padding: 1.2rem 1.7rem;
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 14px;
  box-shadow: 0 18px 50px rgba(29, 26, 22, 0.12);
  margin-bottom: 1.25rem;
  position: relative;
  overflow: hidden;
  animation: rise 0.8s ease-out;
}
.app-title {
  font-family: "Baskerville", "Garamond", "Times New Roman", serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--ink);
  margin: 0;
}
.title-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.flag {
  width: 56px;
  height: 38px;
  border-radius: 6px;
  box-shadow: 0 10px 18px rgba(29, 26, 22, 0.12);
  border: 1px solid rgba(28, 25, 22, 0.18);
}
.app-subtitle {
  color: #3f3831;
  margin: 0.25rem 0 0 0;
  font-size: 1rem;
}
.badge {
  display: inline-block;
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  background: rgba(192, 86, 33, 0.2);
  color: #4a250f;
  font-size: 0.8rem;
  margin-right: 0.4rem;
}
.accent-bar {
  position: absolute;
  inset: auto 0 0 0;
  height: 6px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
}
.stat-card {
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 12px;
  padding: 0.9rem 1rem;
  box-shadow: 0 8px 24px rgba(29, 26, 22, 0.08);
  animation: rise 0.9s ease-out;
}
.stat-row {
  margin-bottom: 1.25rem;
}
.stat-label {
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.7rem;
  color: #5b4d41;
  margin-bottom: 0.3rem;
}
.stat-value {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--ink);
}
[data-testid="stChatMessage"] {
  border-radius: 16px;
  padding: 0.5rem 0.4rem;
  margin-bottom: 0.6rem;
  animation: floatin 0.5s ease-out;
}
[data-testid="stChatMessage"][data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
  border-radius: 14px;
  padding: 0.85rem 1rem;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(28, 25, 22, 0.14);
  color: var(--ink);
}
[data-testid="stChatMessage"] .stMarkdown p {
  margin-bottom: 0.6rem;
  color: var(--ink);
}
[data-testid="stChatMessage"]:has(svg) [data-testid="stChatMessageContent"] {
  background: rgba(31, 79, 89, 0.12);
  border-color: rgba(31, 79, 89, 0.28);
  color: var(--ink);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(245, 234, 218, 0.98) 100%);
  border-right: 1px solid var(--line);
  color: var(--ink);
  font-family: "Baskerville", "Garamond", "Times New Roman", serif;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio label span,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label {
  color: #000000 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label span,
section[data-testid="stSidebar"] [role="radiogroup"] label,
section[data-testid="stSidebar"] [role="radiogroup"] label span {
  color: #000000 !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] *,
section[data-testid="stSidebar"] [data-baseweb="radio"] label,
section[data-testid="stSidebar"] [data-baseweb="radio"] span {
  color: #000000 !important;
}
section[data-testid="stSidebar"] .stButton button {
  background: var(--accent);
  color: white;
  border: none;
}
section[data-testid="stSidebar"] .stButton button:hover {
  background: #a94a1b;
}
section[data-testid="stSidebar"] .stRadio > div {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  padding: 0.5rem;
  border: 1px solid rgba(28, 25, 22, 0.16);
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="textarea"] {
  background: rgba(255, 255, 255, 0.98) !important;
  color: var(--ink) !important;
  border-color: rgba(28, 25, 22, 0.2) !important;
}
.stExpander {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.88);
}
.stExpanderHeader {
  font-weight: 600;
  color: var(--ink);
}
@keyframes rise {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes floatin {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""",
    unsafe_allow_html=True,
)

flag_html = ""
if FLAG_SVG_B64:
    flag_html = (
        f'<img class="flag" src="data:image/svg+xml;base64,{FLAG_SVG_B64}" '
        'alt="Tunisia flag" />'
    )

st.markdown(
    f"""
<div class="app-shell">
  <div class="app-header">
    <div class="badge">Narratives Project</div>
    <div class="badge">RAG Archive</div>
    <div class="badge">CI3</div>
    <div class="title-row">
      <h1 class="app-title">Narratives around the Tunisian Revolution</h1>
      {flag_html}
    </div>
    <p class="app-subtitle">Safwen Gharbi | CI3 | Faculte des Sciences de Bizerte</p>
    <p class="app-subtitle">Ask questions and inspect sources, scores, and evaluation hooks.</p>
    <div class="accent-bar"></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="stat-row">', unsafe_allow_html=True)
cols = st.columns(3)
with cols[0]:
    st.markdown(
        """
<div class="stat-card">
  <div class="stat-label">Author</div>
  <div class="stat-value">Safwen Gharbi</div>
</div>
""",
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        """
<div class="stat-card">
  <div class="stat-label">Class</div>
  <div class="stat-value">CI3</div>
</div>
""",
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        """
<div class="stat-card">
  <div class="stat-label">Faculty</div>
  <div class="stat-value">Faculte des Sciences de Bizerte</div>
</div>
""",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


def apply_llm_settings(
    provider_choice: str,
    groq_api_key: str,
    ollama_url: str,
    timeout_s: int,
) -> Dict[str, Any]:
    if provider_choice.startswith("Ollama"):
        os.environ["RAG_LLM_PROVIDER"] = "ollama"
        os.environ["RAG_LLM_MODEL"] = "gemma3"
        if ollama_url:
            os.environ["RAG_OLLAMA_URL"] = ollama_url
        os.environ["RAG_ALLOW_FOREIGN_SERVICES"] = "0"
    else:
        os.environ["RAG_LLM_PROVIDER"] = "groq"
        os.environ["RAG_LLM_MODEL"] = "qwen/qwen3-32b"
        os.environ["RAG_ALLOW_FOREIGN_SERVICES"] = "1"
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

    os.environ["RAG_LLM_TIMEOUT_S"] = str(timeout_s)
    return ask_api.get_llm_config()


def format_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for c in citations:
        rows.append(
            {
                "rank": c.get("rank"),
                "hybrid_score": c.get("hybrid_score"),
                "source_file": c.get("source_file"),
                "chunk_id": c.get("chunk_id"),
                "snippet": c.get("snippet"),
            }
        )
    return rows


def render_details(payload: Dict[str, Any]) -> None:
    citations = payload.get("top_citations", [])

    with st.expander("Sources used", expanded=False):
        if citations:
            st.dataframe(format_citations(citations), use_container_width=True)
        else:
            st.write("No sources returned.")

    with st.expander("Evaluation hooks", expanded=False):
        st.json(payload.get("evaluation_hooks", {}))

    with st.expander("Run details", expanded=False):
        st.json(
            {
                "decision": payload.get("decision"),
                "decision_reason": payload.get("decision_reason"),
                "tool_trace": payload.get("tool_trace"),
                "response_lang": payload.get("response_lang"),
                "low_confidence": payload.get("low_confidence"),
            }
        )


with st.sidebar:
    st.header("Model selection")

    provider_choice = st.radio(
        "LLM provider",
        ["Ollama (gemma3, local)", "Groq (qwen/qwen3-32b)"],
        index=0,
    )

    timeout_default = int(os.getenv("RAG_LLM_TIMEOUT_S", "60"))
    timeout_s = st.number_input(
        "LLM timeout (seconds)",
        min_value=10,
        max_value=600,
        value=timeout_default,
        step=5,
    )

    if provider_choice.startswith("Ollama"):
        ollama_url = st.text_input(
            "Ollama base URL",
            value=os.getenv("RAG_OLLAMA_URL", "http://localhost:11434"),
        )
        groq_api_key = ""
    else:
        ollama_url = ""
        groq_api_key = st.text_input(
            "Groq API key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
        )
        st.caption("Groq calls are treated as foreign services for the safety gate.")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []

    st.caption("Sources and evaluation hooks appear under each assistant reply.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("data"):
            render_details(message["data"])


prompt = st.chat_input("Ask a question about the Tunisian Revolution archive")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and answering..."):
            try:
                if provider_choice.startswith("Groq"):
                    if not (groq_api_key or os.getenv("GROQ_API_KEY")):
                        raise RuntimeError("GROQ_API_KEY is required for Groq.")

                cfg = apply_llm_settings(
                    provider_choice=provider_choice,
                    groq_api_key=groq_api_key,
                    ollama_url=ollama_url,
                    timeout_s=timeout_s,
                )
                payload = ask_api.answer_question(prompt, cfg)
                answer = payload.get("final_answer", "").strip()
                if not answer:
                    answer = "No answer returned."
                st.markdown(answer)
                render_details(payload)
            except Exception as exc:
                answer = f"Error: {exc}"
                payload = {
                    "final_answer": answer,
                    "top_citations": [],
                    "evaluation_hooks": {},
                    "decision": "error",
                    "decision_reason": str(exc),
                    "tool_trace": [],
                    "response_lang": None,
                    "low_confidence": None,
                }
                st.error(answer)
                render_details(payload)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "data": payload,
        }
    )
