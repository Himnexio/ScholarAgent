import streamlit as st
from scholaragent import config as cfg
from scholaragent.agent import run_agent

st.set_page_config(page_title="ScholarAgent",layout="wide")
st.title("ScholarAgent")
st.caption("Agentic RAG over arXiv papers: plans queries, checks its own retrieval, and cites its sources.")

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"**LLM:** {cfg.LLM_PROVIDER} / {cfg.LLM_MODEL or 'default'}")
    st.write(f"**Embedder:** `{cfg.EMBED_MODEL}`")
    st.write(f"**Reranker:** {'on' if cfg.USE_RERANKER else 'off'}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_extras(msg: dict) -> None:
    if msg.get("citations"):
        with st.expander(f"Sources ({len(msg['citations'])})"):
            for c in msg["citations"]:
                st.markdown(f"**[{c['n']}] [{c['title']}]({c['url']})** - {c['authors']} ({c['year']}), _{c['section']}_")
                st.caption(c["snippet"] + " ...")
    if msg.get("trace"):
        with st.expander("Agent trace"):
            for step in msg["trace"]:
                st.write("- " + step)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_extras(m)

if question := st.chat_input("Ask about the papers in the index..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching, checking and writing..."):
                out = run_agent(question, history)
            st.markdown(out["answer"])
            msg = {"role": "assistant", "content": out["answer"], "citations": out["citations"], "trace": out["trace"]}
            render_extras(msg)
            st.session_state.messages.append(msg)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")