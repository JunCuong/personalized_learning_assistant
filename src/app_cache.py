"""Cached resources for Streamlit app (embedder, FAISS, retriever, LLM)."""

from __future__ import annotations

from src.embedder import Embedder
from src.llm_client import LLMClient
from src.retriever import Retriever


def get_cache_version() -> int:
    """Bump this after knowledge-base rebuild to invalidate caches."""
    import streamlit as st

    return int(st.session_state.get("kb_cache_version", 0))


def bump_cache_version() -> None:
    import streamlit as st

    st.session_state["kb_cache_version"] = get_cache_version() + 1


def clear_all_caches() -> None:
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()
    bump_cache_version()


def get_cached_embedder(_version: int) -> Embedder:
    return Embedder()


def get_cached_retriever(_version: int) -> Retriever:
    embedder = get_cached_embedder(_version)
    retriever = Retriever(embedder=embedder)
    retriever.load()
    return retriever


def get_cached_llm(allow_local_fallback: bool = False) -> LLMClient:
    return LLMClient(allow_local_fallback=allow_local_fallback)
