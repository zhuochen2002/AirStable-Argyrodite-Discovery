#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG: Retrieval-Augmented Generation System for Solid-State Battery
- Generation via Intern S1 API
- Vector retrieval, context assembly, Intern S1 answer
- Dependencies: sentence-transformers + chromadb

Environment Variables:
  HF_ENDPOINT=https://hf-mirror.com
  SENTENCE_TRANSFORMERS_HOME: embedding model directory
"""

import os
import json
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# SENTENCE_TRANSFORMERS_HOME will be set based on relative path after _BASE_DIR is defined

# Support both package-relative imports and script execution
try:
    from .rag_config import RAG_CONFIG
    from .vector_store import VectorStore
except ImportError:  # script run directly
    from rag_config import RAG_CONFIG
    from vector_store import VectorStore

# Intern S1 API client (package import or script run)
try:
    from .API.llm_clients import InternS1Client
except ImportError:  # script run directly
    from API.llm_clients import InternS1Client

# ==================== Paths (aligned with vector_store.py, relative to this file) ====================
_BASE_DIR = Path(__file__).parent.resolve()
MODEL_BASE_PATH = str(_BASE_DIR / "models")
VECTOR_DB_PATH = str(_BASE_DIR / "data" / "chroma_db")
PROCESSED_DOCS_PATH = str(_BASE_DIR / "data" / "processed_docs")
PAPERS_DIR = str(_BASE_DIR / "docs" / "papers")

os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_BASE_DIR / "models"))
# ===================================================

PATHS = {
    "models_root": Path(MODEL_BASE_PATH),
    "chroma_db": Path(VECTOR_DB_PATH),
    "runs_dir": Path(_BASE_DIR / "data" / "rag_runs"),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("RAG")


class InternS1LLM:
    """
    Intern S1 API wrapper: build prompt, call Intern S1, return answer.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://chat.intern-ai.org.cn/api/v1/",
                 model: str = "intern-s1", thinking_mode: bool = False):
        self.client = InternS1Client(
            api_key=api_key,
            base_url=base_url,
            model=model,
            thinking_mode=thinking_mode
        )
        log.info(f"[InternS1LLM] API client ready (model={model}, thinking={thinking_mode})")

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build prompt for Intern S1 API
        Returns: prompt string
        """
        require_citations = bool(RAG_CONFIG.get("require_citations", True))
        citation_rule = ""
        if require_citations:
            citation_rule = (
                "\n\nIMPORTANT CITATION RULES:\n"
                "- You MUST cite sources using the exact citation tokens present in the context, e.g. [[paper.pdf#12]].\n"
                "- Every major claim should be supported by at least one citation token.\n"
                "- If the context does not contain enough evidence, explicitly say so.\n"
            )

        prompt = (
            "You are a distinguished expert in solid-state battery. "
            "Your expertise encompasses electrochemical engineering, materials science, and energy storage systems. "
            "Provide comprehensive, evidence-based responses utilizing the provided scientific literature. "
            "Maintain technical accuracy and cite specific findings when possible. "
            "IMPORTANT: Always provide complete, well-structured answers. Do not cut off mid-sentence. "
            "If you need to continue, use clear transitions and ensure logical flow.\n\n"
            f"Scientific Literature Context:\n{context}\n\n"
            f"Research Question:\n{question}\n\n"
            "Please provide a comprehensive, complete answer based on the literature."
            f"{citation_rule}"
        )
        return prompt

    def generate(self, question: str, context: str, max_new_tokens: int = 2048, temperature: float = 0.7) -> str:
        """
        Generate answer using Intern S1 API
        
        Args:
            question: Research question
            context: Scientific literature context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated answer text
        """
        prompt = self._build_prompt(question, context)
        
        def _parse_citations(t: str) -> List[str]:
            import re as _re
            return _re.findall(r"\[\[[^\]]+\]\]", t or "")

        def _has_citations(t: str, allowed: Optional[set] = None) -> bool:
            tokens = _parse_citations(t)
            min_cites = int(RAG_CONFIG.get("min_citations", 0))
            if len(tokens) < max(0, min_cites):
                return False
            if allowed is not None:
                return all(tok in allowed for tok in tokens)
            import re as _re
            return bool(_re.findall(r"\[\[[^\]]+\]\]", t or ""))

        try:
            allowed_tokens = set(_parse_citations(context))
            text = self.client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=0.9
            )
            
            text = self._ensure_complete_answer(text, question)

            # Retry once if citations required but missing
            if bool(RAG_CONFIG.get("require_citations", True)) and not _has_citations(text, allowed_tokens):
                retry_max = int(RAG_CONFIG.get("citation_retry_max", 0))
                if retry_max > 0:
                    retry_prompt = prompt + "\n\nYou did not include enough citation tokens. Rewrite the answer and include citations like [[paper.pdf#12]] throughout."
                    text2 = self.client.generate(
                        prompt=retry_prompt,
                        temperature=min(0.4, temperature),
                        max_tokens=max_new_tokens,
                        top_p=0.9,
                    )
                    text2 = self._ensure_complete_answer(text2, question)
                    if _has_citations(text2, allowed_tokens):
                        text = text2
            
            return text.strip()
        except Exception as e:
            log.error(f"[InternS1LLM] API call failed: {e}")
            return f"Sorry, generation failed: {str(e)}"

    def summarize_context(self, context: str, max_new_tokens: int = 4096) -> str:
        """
        Summarize long retrieved context with the LLM while preserving citation tokens and key evidence.
        """
        prompt = (
            "You are summarizing scientific literature excerpts for a RAG system.\n"
            "The following context contains retrieved chunks with citation tokens like [[PaperTitle#chunk_id]].\n"
            "Produce a condensed summary that:\n"
            "1) Preserves ALL citation tokens [[...]] exactly as they appear - do NOT remove or modify them.\n"
            "2) Keeps the key experimental findings, numbers, and conclusions relevant to solid-state battery / argyrodite research.\n"
            "3) Removes redundancy and minor details.\n"
            "4) Maintains the structure (paper separators, chunk labels) where helpful for attribution.\n"
            "Output the summary directly, no preamble.\n\n"
            f"Context to summarize:\n{context}"
        )
        try:
            text = self.client.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=max_new_tokens,
                top_p=0.9,
            )
            return (text or context).strip()
        except Exception as e:
            log.error(f"[InternS1LLM] summarize_context failed: {e}")
            return context

    def rewrite_for_retrieval(self, question: str, max_new_tokens: int = 128) -> str:
        """
        Rewrite the question for dense retrieval: short English query and key phrases.
        On failure, returns the original question.
        """
        prompt = (
            "You are optimizing a research question for dense vector retrieval over scientific papers.\n"
            "Rewrite the following question into a concise English query focusing on key entities, "
            "materials (e.g., Li6PS5Cl, argyrodite), dopants, properties (ionic conductivity, activation energy, SEI, CCD), "
            "and experimental conditions.\n"
            "Output ONLY the rewritten query, no explanation, no bullet points.\n\n"
            f"Original question:\n{question}\n\n"
            "Rewritten query:"
        )
        try:
            text = self.client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=max_new_tokens,
                top_p=0.9,
            )
            text = (text or "").strip()
            return text or question
        except Exception as e:
            log.error(f"[InternS1LLM] rewrite_for_retrieval failed: {e}")
            return question

    def decompose_question(self, question: str, max_subqueries: int = 4, max_new_tokens: int = 256) -> List[str]:
        """
        Split a complex question into retrievable sub-queries for multi-query retrieval.
        Returns a list of sub-queries; on failure returns an empty list.
        """
        max_subqueries = max(1, min(int(max_subqueries), 8))
        prompt = (
            "You are decomposing a complex materials-science research question into smaller retrieval queries.\n"
            "Generate up to {n} concise English sub-queries that can each be answered by searching scientific papers.\n"
            "Rules:\n"
            "- Each sub-query MUST be short and specific (one sentence).\n"
            "- Focus on key entities/materials/dopants/metrics/conditions.\n"
            "- Avoid redundant sub-queries.\n"
            "- Output ONLY a JSON array of strings.\n\n"
            "Original question:\n{q}\n\n"
            "JSON array:"
        ).format(n=max_subqueries, q=question)
        try:
            raw = self.client.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=max_new_tokens,
                top_p=0.9,
            )
            raw = (raw or "").strip()
            l = raw.find("[")
            r = raw.rfind("]")
            if l >= 0 and r > l:
                raw = raw[l:r + 1]
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
            out: List[str] = []
            for item in data:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append(s)
            seen = set()
            uniq = []
            for s in out:
                k = s.lower()
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(s)
            return uniq[:max_subqueries]
        except Exception as e:
            log.error(f"[InternS1LLM] decompose_question failed: {e}")
            return []

    def _ensure_complete_answer(self, text: str, question: str) -> str:
        """
        Heuristically mark possibly truncated answers.
        """
        if not text:
            return text
        
        # Incomplete-sentence openers
        incomplete_endings = [
            "However,", "Furthermore,", "In addition,", "Moreover,", "Additionally,",
            "Therefore,", "Thus,", "Consequently,", "As a result,", "In conclusion,",
            "To summarize,", "In summary,", "Overall,", "In general,", "Generally,"
        ]
        
        incomplete_clauses = [
            "which", "that", "where", "when", "if", "although", "because", "since",
            "while", "unless", "until", "before", "after", "during", "through"
        ]
        
        text_clean = text.strip()
        
        for ending in incomplete_endings:
            if text_clean.endswith(ending):
                text_clean += " [Answer continues...]"
                break
        
        for clause in incomplete_clauses:
            if text_clean.endswith(clause):
                text_clean += " [Answer continues...]"
                break
        
        if text_clean and not text_clean[-1] in ".!?":
            if not any(text_clean.endswith(ending) for ending in incomplete_endings):
                text_clean += " [Answer may be truncated due to length limits]"
        
        return text_clean


class RAG:
    """
    RAG System:
    - Retrieval (Chroma) → Context Assembly → Local LLM Answer Generation
    - Execution logs written to PATHS["runs_dir"]/run-YYYYmmdd-HHMMSS.jsonl
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        runs_dir: Optional[Path] = None,
        vector_store: Optional[VectorStore] = None,
        embed_model_id: str = "all-MiniLM-L6-v2",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: str = "https://chat.intern-ai.org.cn/api/v1/",
        model: str = "intern-s1",
        thinking_mode: bool = False,
    ):
        _runs_dir = Path(runs_dir) if runs_dir else PATHS["runs_dir"]
        _runs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = _runs_dir / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"

        if vector_store is not None:
            self.vs = vector_store
        elif db_path is not None:
            self.vs = VectorStore(
                persist_dir=str(db_path),
                collection_name="papers",
                model_name=embed_model_id,
                reset_collection=False,
                chunking_strategy="semantic",
            )
        else:
            self.vs = VectorStore(
                persist_dir=str(PATHS["chroma_db"]),
                collection_name="papers",
                model_name=embed_model_id,
                reset_collection=False,
                chunking_strategy="semantic",
            )
        
        self.llm = InternS1LLM(
            api_key=api_key,
            base_url=base_url,
            model=model,
            thinking_mode=thinking_mode
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        log.info(f"ChromaDB: {getattr(self.vs, 'persist_dir', 'N/A')}")
        log.info(f"LLM: Intern S1 API (model: {model}, thinking_mode: {thinking_mode})")
        log.info(f"Execution log: {self.log_file}")

    @staticmethod
    def _pack_context(docs: List[Dict[str, Any]], max_chars: int = 2000) -> str:
        parts: List[str] = []
        used = 0
        for i, d in enumerate(docs, 1):
            txt = d.get("text", "")
            meta = d.get("metadata", {})
            head = f"[Document {i}] Source: {meta.get('filename','unknown')} | Chunk ID: {meta.get('chunk_id','-')}"
            snippet = txt[:500]
            block = f"{head}\n{snippet}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts) if parts else "No relevant scientific documents found in the database."

    def _get_enhanced_context(self, docs: List[Dict[str, Any]], context_window: int = 2) -> List[Dict[str, Any]]:
        """
        Build expanded context: retrieved docs plus neighboring chunks.
        Args:
            docs: Retrieved document list
            context_window: Chunks before/after each hit to include
        """
        enhanced_docs = []
        
        for doc in docs:
            meta = doc.get("metadata", {})
            filename = meta.get("filename", "")
            chunk_id = meta.get("chunk_id", 0)
            
            enhanced_docs.append(doc)
            
            try:
                for i in range(1, context_window + 1):
                    prev_chunk_id = chunk_id - i
                    if prev_chunk_id >= 0:
                        prev_doc = self._get_chunk_by_id(filename, prev_chunk_id)
                        if prev_doc:
                            prev_doc["metadata"]["type"] = "context"
                            enhanced_docs.append(prev_doc)
                        else:
                            placeholder = {
                                "text": f"[Context: Previous chunk {prev_chunk_id} from {filename} - Content not available]",
                                "metadata": {"filename": filename, "chunk_id": prev_chunk_id, "type": "context"},
                                "score": 0.0,
                                "excerpt": f"Context placeholder for chunk {prev_chunk_id}"
                            }
                            enhanced_docs.append(placeholder)
                
                for i in range(1, context_window + 1):
                    next_chunk_id = chunk_id + i
                    next_doc = self._get_chunk_by_id(filename, next_chunk_id)
                    if next_doc:
                        next_doc["metadata"]["type"] = "context"
                        enhanced_docs.append(next_doc)
                    else:
                        placeholder = {
                            "text": f"[Context: Next chunk {next_chunk_id} from {filename} - Content not available]",
                            "metadata": {"filename": filename, "chunk_id": next_chunk_id, "type": "context"},
                            "score": 0.0,
                            "excerpt": f"Context placeholder for chunk {next_chunk_id}"
                        }
                        enhanced_docs.append(placeholder)
                        
            except Exception as e:
                log.warning(f"Error getting context for chunk {chunk_id}: {e}")
        
        return enhanced_docs

    def _get_chunk_by_id(self, filename: str, chunk_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a chunk by metadata (Chroma where filter); faster than semantic search for exact IDs.
        """
        try:
            return self.vs.get_chunk_by_metadata(filename=filename, chunk_id=chunk_id)
        except Exception as e:
            log.warning(f"Error retrieving chunk {chunk_id} from {filename}: {e}")
            return None

    def _pack_enhanced_context(self, docs: List[Dict[str, Any]], max_chars: int = 3000) -> str:
        """
        Pack expanded context including neighbor chunks for coherent reading.
        """
        if not docs:
            return "No relevant scientific documents found in the database."
        
        parts = []
        used = 0

        if max_chars is None:
            max_chars = int(RAG_CONFIG.get("enhanced_context_max_chars", 3000))
        max_chunks_per_file = int(RAG_CONFIG.get("max_chunks_per_file", 4))

        sorted_docs = sorted(
            docs,
            key=lambda x: (
                x.get("metadata", {}).get("filename", ""),
                x.get("metadata", {}).get("chunk_id", 0),
            ),
        )

        current_file = ""
        per_file_count = {}

        retrieved_docs = [
            d for d in sorted_docs
            if (d.get("metadata", {}) or {}).get("type", "retrieved") != "context"
        ]
        def _score_key(d: Dict[str, Any]) -> float:
            meta_score = d.get("ranking_score")
            if meta_score is not None:
                return float(meta_score)
            return float(d.get("score", 0.0))

        top_head = top_tail = None
        if retrieved_docs:
            retrieved_sorted = sorted(retrieved_docs, key=_score_key, reverse=True)
            top_head = retrieved_sorted[0]
            if len(retrieved_sorted) > 1:
                top_tail = retrieved_sorted[1]

        def _citation_token(meta: Dict[str, Any]) -> str:
            display_name = (meta.get("original_title") or meta.get("filename") or "unknown").strip()
            if len(display_name) > 50:
                display_name = display_name[:47] + "..."
            cid = meta.get("chunk_id")
            if cid is None or cid == "-":
                pid = meta.get("parent_id", "")
                cid = pid[-6:] if pid else "p"
            return f"[[{display_name}#{cid}]]"

        for i, doc in enumerate(sorted_docs):
            if doc is top_head or doc is top_tail:
                continue

            txt = doc.get("text", "")
            meta = doc.get("metadata", {}) or {}
            filename = meta.get("filename", "unknown")
            chunk_id = meta.get("chunk_id", "-")
            doc_type = meta.get("type", "retrieved")
            section_type = meta.get("section_type", "")

            cnt = per_file_count.get(filename, 0)
            if cnt >= max_chunks_per_file:
                continue
            per_file_count[filename] = cnt + 1

            if filename != current_file:
                if current_file:
                    parts.append("=" * 60)
                current_file = filename
                paper_label = (meta.get("original_title") or filename).strip()
                if len(paper_label) > 60:
                    paper_label = paper_label[:57] + "..."
                parts.append(f"📄 Paper: {paper_label}")
                parts.append("=" * 60)

            sec_str = f" | SectionType: {section_type}" if section_type else ""
            cite = _citation_token(meta)
            if doc_type == "context":
                head = f"   📍 Context Chunk {chunk_id}{sec_str} {cite}"
            else:
                head = f"   🔍 Retrieved Chunk {chunk_id} (Score: {doc.get('score', 0):.3f}){sec_str} {cite}"

            no_trunc = bool(RAG_CONFIG.get("no_snippet_truncation", True))
            if no_trunc:
                snippet = txt
            else:
                rl = int(RAG_CONFIG.get("retrieved_snippet_chars", 800))
                cl = int(RAG_CONFIG.get("context_snippet_chars", 400))
                snippet = txt[:rl] if doc_type == "retrieved" else txt[:cl]
            block = f"{head}\n{snippet}"

            parts.append(block)

            if i < len(sorted_docs) - 1:
                parts.append("-" * 40)

        def _format_block(doc: Dict[str, Any]) -> str:
            txt = doc.get("text", "")
            meta = doc.get("metadata", {}) or {}
            chunk_id = meta.get("chunk_id", "-")
            doc_type = meta.get("type", "retrieved")
            section_type = meta.get("section_type", "")
            cite = _citation_token(meta)
            sec_str = f" | SectionType: {section_type}" if section_type else ""
            if doc_type == "context":
                head = f"   📍 Context Chunk {chunk_id}{sec_str} {cite}"
            else:
                head = f"   🔍 Retrieved Chunk {chunk_id} (Score: {doc.get('score', 0):.3f}){sec_str} {cite}"
            no_trunc = bool(RAG_CONFIG.get("no_snippet_truncation", True))
            snippet = txt if no_trunc else txt[: int(RAG_CONFIG.get("retrieved_snippet_chars", 800))]
            return f"{head}\n{snippet}"

        if top_head is not None:
            head_block = _format_block(top_head)
            parts.insert(0, head_block)

        if top_tail is not None:
            tail_block = _format_block(top_tail)
            parts.append("-" * 40)
            parts.append(tail_block)

        full_ctx = "\n\n".join(parts)
        enable_summary = bool(RAG_CONFIG.get("enable_context_summarization", False))
        if not enable_summary and len(full_ctx) > max_chars:
            full_ctx = full_ctx[: max_chars - 80] + "\n\n   ... [Context truncated due to length limit]"
        return full_ctx

    def ask(self, question: str, top_k: int = 5, include_context: bool = True, context_window: int = 1) -> Dict[str, Any]:
        log.info(f"Searching for question: {question}")
        
        db_stats = self.vs.get_collection_stats()
        log.info(f"Database stats: {db_stats}")
        
        enable_rewrite = bool(RAG_CONFIG.get("enable_query_rewrite", True))
        min_len = int(RAG_CONFIG.get("min_rewrite_length", 0))
        if enable_rewrite and len(question) >= min_len:
            rewritten_question = self.llm.rewrite_for_retrieval(
                question,
                max_new_tokens=int(RAG_CONFIG.get("rewrite_max_new_tokens", 128)),
            )
        else:
            rewritten_question = question

        if rewritten_question != question:
            log.info(f"Rewritten query for retrieval: {rewritten_question}")

        def _expand_query(q: str) -> str:
            """Expand retrieval query; bias toward interface vs conductivity to reduce overlap."""
            if not RAG_CONFIG.get("enable_query_expansion", True):
                return q
            q_expanded = q
            q_lower = q.lower()
            if "li6ps5cl" in q_lower or "argyrodite" in q_lower:
                q_expanded += " Li6PS5X argyrodite Li6PS5Cl solid electrolyte"
            if any(k in q_lower for k in ["sei", "interphase", "li metal", "ccd", "r_int", "interfacial"]):
                q_expanded += " SEI solid electrolyte interphase Li metal interface stability CCD impedance R_int"
            elif any(k in q_lower for k in ["ionic conductivity", "ion conductivity", "activation energy", "σ"]):
                q_expanded += " ionic conductivity ion transport activation energy Ea"
            return q_expanded

        rewritten_for_retrieval = _expand_query(rewritten_question)

        def _rule_based_subqueries(q: str) -> tuple[List[str], List[str]]:
            ql = (q or "").lower()
            rules = RAG_CONFIG.get("must_have_query_rules") or []
            out: List[str] = []
            hit_names: List[str] = []
            for rule in rules:
                try:
                    patterns = rule.get("patterns") or []
                    name = str(rule.get("name") or "rule")
                    matched = any(re.search(pat, ql) for pat in patterns if isinstance(pat, str) and pat)
                    if not matched:
                        continue
                    hit_names.append(name)
                    for sq in (rule.get("queries") or []):
                        if isinstance(sq, str) and sq.strip():
                            out.append(sq.strip())
                except Exception:
                    continue
            seen = set()
            uniq: List[str] = []
            for s in out:
                k = s.lower()
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(s)
            if hit_names:
                log.info(f"Rule-based must-have queries triggered: {', '.join(hit_names)}")
            return uniq, hit_names

        enable_decomp = bool(RAG_CONFIG.get("enable_decomposition", True))
        min_decomp_len = int(RAG_CONFIG.get("min_decompose_length", 0))
        subqueries: List[str] = []
        if enable_decomp and len(question) >= min_decomp_len:
            subqueries = self.llm.decompose_question(
                question,
                max_subqueries=int(RAG_CONFIG.get("max_subqueries", 4)),
            )
            if subqueries:
                log.info(f"Decomposed into {len(subqueries)} subqueries")

        rule_subqueries, rule_hits = _rule_based_subqueries(question)
        if rule_subqueries:
            merged = rule_subqueries + (subqueries or [])
            seen = set()
            subqueries2: List[str] = []
            for s in merged:
                k = s.lower()
                if k in seen:
                    continue
                seen.add(k)
                subqueries2.append(s)
            subqueries = subqueries2

        pool_mul = int(RAG_CONFIG.get("candidate_pool_multiplier", 1))
        base_pool = max(top_k, top_k * max(1, pool_mul))

        all_docs = self.vs.search_similar(
            rewritten_for_retrieval,
            k=base_pool,
            score_threshold=0.4,
            use_mmr=False,
            query_prefix="solid-state battery",
            kw_query=question,
        )

        fr_cfg = RAG_CONFIG.get("forced_recall") or {}
        if bool(fr_cfg.get("enabled", False)) and rule_hits:
            try:
                qmap = fr_cfg.get("queries") or {}
                thr = float(fr_cfg.get("score_threshold", 0.22))
                pm = int(fr_cfg.get("pool_multiplier", 12))
                min_k = int(fr_cfg.get("min_k", 30))
                require_tag_match = bool(fr_cfg.get("require_tag_match", True))
                text_fb = fr_cfg.get("text_fallback_contains") or {}

                if any("fluoride" in h.lower() or "inf3" in h.lower() for h in rule_hits):
                    topic = "Fluoride"
                    forced_query = str(qmap.get(topic) or "").strip()
                    if forced_query:
                        k_forced = max(min_k, top_k * max(1, pm))
                        forced_hits = self.vs.search_similar(
                            forced_query,
                            k=k_forced,
                            score_threshold=thr,
                            use_mmr=False,
                            query_prefix="solid-state battery",
                            kw_query=question,
                        )

                        if require_tag_match:
                            kept = []
                            fb_terms = [str(x).lower() for x in (text_fb.get(topic) or []) if x]
                            for h in forced_hits:
                                meta = h.get("metadata", {}) or {}
                                tags = str(meta.get("tags") or "")
                                if topic in tags:
                                    kept.append(h)
                                    continue
                                txt = (h.get("text") or "").lower()
                                if any(t in txt for t in fb_terms):
                                    kept.append(h)
                            forced_hits = kept

                        if forced_hits:
                            log.info(f"Forced recall added {len(forced_hits)} hits for topic={topic}")
                            all_docs.extend(forced_hits)
            except Exception as e:
                log.warning(f"Forced recall fallback failed: {e}")

        tgr = RAG_CONFIG.get("title_guided_recall") or {}
        if bool(tgr.get("enabled", False)):
            try:
                ql = (question or "").lower()
                triggers = [str(x).lower() for x in (tgr.get("trigger_terms") or []) if x]
                if any(t in ql for t in triggers):
                    base_dir = Path(__file__).parent.resolve()
                    processed_path = base_dir / "data" / "processed_papers.json"
                    if processed_path.exists():
                        with open(processed_path, "r", encoding="utf-8") as f:
                            pdata = json.load(f)
                    else:
                        pdata = []

                    kw_map = tgr.get("title_keywords") or {}
                    core_kw = [str(x).lower() for x in (kw_map.get("Core") or []) if x]
                    themes = ["Fluoride", "Chloride", "Bromide", "Iodide"]
                    theme_hits = []
                    for th in themes:
                        kws = [str(x).lower() for x in (kw_map.get(th) or []) if x]
                        if any(k in ql for k in kws):
                            theme_hits.append(th)
                    if not theme_hits:
                        if any(x in ql for x in ["inf3", "fluoride", "f-", "f\u207b"]):
                            theme_hits = ["Fluoride"]
                        elif any(x in ql for x in ["chloride", "cl-"]):
                            theme_hits = ["Chloride"]
                        elif any(x in ql for x in ["bromide", "br-"]):
                            theme_hits = ["Bromide"]
                        elif any(x in ql for x in ["iodide", "i-"]):
                            theme_hits = ["Iodide"]

                    scored = []
                    for item in (pdata or []):
                        meta = (item or {}).get("metadata", {}) or {}
                        fn = meta.get("filename")
                        title = (meta.get("original_title") or meta.get("title") or "").strip()
                        if not fn or not title:
                            continue
                        tl = title.lower()
                        score = 0
                        score += sum(1 for k in core_kw if k and k in tl)
                        for th in theme_hits:
                            kws = [str(x).lower() for x in (kw_map.get(th) or []) if x]
                            score += sum(1 for k in kws if k and k in tl) * 2
                        if score > 0:
                            scored.append((score, fn, title))
                    scored.sort(key=lambda x: x[0], reverse=True)

                    max_papers = int(tgr.get("max_papers", 3))
                    per_paper = tgr.get("per_paper") or {}
                    per_thr = float(per_paper.get("score_threshold", 0.0))
                    per_k = int(per_paper.get("k", 10))

                    injected = 0
                    for _, fn, ttl in scored[:max_papers]:
                        hits = self.vs.search_similar(
                            rewritten_for_retrieval,
                            k=per_k,
                            score_threshold=per_thr,
                            use_mmr=False,
                            query_prefix="solid-state battery",
                            kw_query=question,
                            where={"filename": {"$eq": fn}},
                        )
                        if not hits:
                            continue
                        best = hits[0]
                        best_meta = best.setdefault("metadata", {}) or {}
                        best_meta["title_guided"] = True
                        best_meta["title_guided_filename"] = fn
                        best_meta["title_guided_title"] = ttl
                        best["ranking_score"] = float(best.get("ranking_score", best.get("score", 0.0))) + 0.05
                        all_docs.append(best)
                        injected += 1
                    if injected:
                        log.info(f"Title-guided recall injected {injected} chunks from matched-paper titles.")
            except Exception as e:
                log.warning(f"Title-guided recall failed: {e}")

        if subqueries:
            per_k = int(RAG_CONFIG.get("per_subquery_top_k", 6))
            per_pool = max(per_k, per_k * max(1, pool_mul))
            for sq in subqueries:
                sq2 = _expand_query(sq)
                hits = self.vs.search_similar(
                    sq2,
                    k=per_pool,
                    score_threshold=0.35,
                    use_mmr=False,
                    query_prefix="solid-state battery",
                    kw_query=question,
                )
                for h in hits:
                    meta = h.setdefault("metadata", {})
                    meta["subquery"] = sq
                all_docs.extend(hits)

        docs = sorted(all_docs, key=lambda x: float(x.get("ranking_score", x.get("score", 0.0))), reverse=True)

        if docs:
            docs = [d for d in docs if not self.vs._is_reference_like(d.get("text") or "")]

        def _extract_required_anion_terms(q: str) -> list[str]:
            ql = (q or "").lower()
            terms: list[str] = []
            if any(k in ql for k in [" f-", "f⁻", "fluoride", " f2", "f3", "f4"]):
                terms.extend(["f-", "f⁻", "fluoride", "f2", "f3", "f4"])
            if any(k in ql for k in [" cl-", "chloride", "cl2", "cl3", "cl4"]):
                terms.extend(["cl-", "chloride", "licl", "cl2", "cl3", "cl4"])
            if any(k in ql for k in [" br-", "bromide", "br2", "br3", "br4"]):
                terms.extend(["br-", "bromide", "libr", "br2", "br3", "br4"])
            if any(k in ql for k in [" i-", "iodide", "i2", "i3", "i4"]):
                terms.extend(["i-", "iodide", "lii", "i2", "i3", "i4"])
            seen = set()
            uniq = []
            for t in terms:
                tl = t.lower()
                if tl in seen:
                    continue
                seen.add(tl)
                uniq.append(t)
            return uniq

        if docs:
            req_terms = [t.lower() for t in _extract_required_anion_terms(question)]
            if req_terms:
                filtered = []
                for d in docs:
                    txt = (d.get("text") or "").lower()
                    if any(t in txt for t in req_terms):
                        filtered.append(d)
                if filtered:
                    docs = filtered

        parent_context_used = False
        if docs and bool(RAG_CONFIG.get("enable_parent_child", True)) and bool(RAG_CONFIG.get("use_parent_for_context", True)):
            max_children_per_parent = int(RAG_CONFIG.get("max_children_per_parent", 2))
            parent_hits = {}
            new_docs = []
            for d in docs:
                meta = d.get("metadata", {}) or {}
                pid = meta.get("parent_id")
                if not pid:
                    new_docs.append(d)
                    continue
                cnt = parent_hits.get(pid, 0)
                if cnt >= max_children_per_parent:
                    continue
                parent_hits[pid] = cnt + 1
                parent_doc = self.vs.get_parent_by_id(pid)
                if parent_doc:
                    parent_doc["score"] = d.get("score", 0.0)
                    if "ranking_score" in d:
                        parent_doc["ranking_score"] = d["ranking_score"]
                    parent_doc["metadata"]["type"] = "retrieved"
                    new_docs.append(parent_doc)
                else:
                    new_docs.append(d)
            docs = new_docs
            parent_context_used = True

        def _is_halogen_only_question(q: str) -> bool:
            """True if question scopes F/Cl/Br/I halogen doping and should drop other anion systems."""
            ql = (q or "").lower()
            has_argyrodite = "li6ps5cl" in ql or "argyrodite" in ql or "li6ps5x" in ql
            has_halogen_scope = (
                any(k in ql for k in ["f⁻", "f-", "fluoride", "cl⁻", "cl-", "chloride",
                                      "br⁻", "br-", "bromide", "i⁻", "i-", "iodide",
                                      "halogen", "f/cl/br/i", "f, cl, br, i"])
                or "dopant-related" in ql or "anion chem" in ql
            )
            return bool(has_argyrodite and has_halogen_scope)

        def _is_excluded_anion_doc(d: Dict[str, Any], excluded_cfg: Dict[str, Any]) -> bool:
            """Whether doc matches an excluded anion system (Se/Te/O) per anion_scope_filter."""
            txt = (d.get("text") or "").lower()
            for anion_name, patterns in (excluded_cfg or {}).items():
                text_pats = patterns.get("text_patterns") or []
                regex_pats = patterns.get("regex_patterns") or []
                if any(p in txt for p in text_pats):
                    return True
                for pat in regex_pats:
                    if isinstance(pat, str) and pat and re.search(pat, txt):
                        return True
            return False

        anion_cfg = RAG_CONFIG.get("anion_scope_filter") or {}
        if docs and _is_halogen_only_question(question) and anion_cfg:
            docs = [d for d in docs if not _is_excluded_anion_doc(d, anion_cfg)]

        def _tokenize(t: str) -> set:
            import re as _re
            t = (t or "").lower()
            toks = [x for x in _re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", t) if len(x) >= 3]
            return set(toks)

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

        def _infer_aspects(q: str) -> List[str]:
            ql = (q or "").lower()
            aspects = []
            if any(k in ql for k in ["ionic conductivity", "ion conductivity", "activation energy", "ea", "σ", "transport"]):
                aspects.append("conductivity")
            if any(k in ql for k in ["sei", "interphase", "interface", "li metal", "compatibility", "ccd", "impedance"]):
                aspects.append("interface")
            if any(k in ql for k in ["dopant", "doping", "substitution", "f-", "br-", "i-", "o2", "oxygen"]):
                aspects.append("doping")
            if any(k in ql for k in ["structure", "disorder", "site", "occupancy", "phase", "xrd", "rietveld"]):
                aspects.append("structure")
            return aspects or ["general"]

        if docs and bool(RAG_CONFIG.get("dedup_by_parent", True)):
            best_by_parent = {}
            for d in docs:
                meta = d.get("metadata", {}) or {}
                pid = meta.get("parent_id") or f"{meta.get('filename','')}#{meta.get('chunk_id','')}"
                score = float(d.get("ranking_score", d.get("score", 0.0)))
                prev = best_by_parent.get(pid)
                if prev is None or score > float(prev.get("ranking_score", prev.get("score", 0.0))):
                    best_by_parent[pid] = d
            docs = sorted(best_by_parent.values(), key=lambda x: float(x.get("ranking_score", x.get("score", 0.0))), reverse=True)

        if docs:
            import hashlib
            kept = []
            seen_hashes = set()

            def _norm_text(t: str) -> str:
                return "".join(t.lower().split())

            for d in docs:
                txt = d.get("text") or ""
                h = hashlib.md5(_norm_text(txt).encode("utf-8", errors="ignore")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                kept.append(d)
            docs = kept

        if docs and bool(RAG_CONFIG.get("enable_text_dedup", True)):
            thr = float(RAG_CONFIG.get("text_dedup_jaccard_threshold", 0.85))
            kept2 = []
            kept_tokens = []
            for d in docs:
                toks = _tokenize(d.get("text", ""))
                dup = False
                for kt in kept_tokens:
                    if _jaccard(toks, kt) >= thr:
                        dup = True
                        break
                if not dup:
                    kept2.append(d)
                    kept_tokens.append(toks)
            docs = kept2

        if docs and bool(RAG_CONFIG.get("enable_coverage_selection", True)):
            aspects = _infer_aspects(question)
            max_per_aspect = int(RAG_CONFIG.get("max_per_aspect", 2))

            def _doc_aspects(d: Dict[str, Any]) -> List[str]:
                txt = (d.get("text") or "").lower()
                meta = d.get("metadata", {}) or {}
                kw_raw = meta.get("keywords") or []
                if isinstance(kw_raw, str):
                    kw_list = [x.strip() for x in kw_raw.split(";") if x.strip()]
                else:
                    kw_list = list(kw_raw) if kw_raw else []
                kws = " ".join(kw_list).lower()
                bag = txt + " " + kws
                tags = []
                if any(k in bag for k in ["ionic conductivity", "ion conductivity", "activation energy", "ea", "σ"]):
                    tags.append("conductivity")
                if any(k in bag for k in ["sei", "interphase", "interface", "li metal", "ccd", "impedance"]):
                    tags.append("interface")
                if any(k in bag for k in ["dopant", "doping", "substitution", "f-", "br-", "i-", "oxygen"]):
                    tags.append("doping")
                if any(k in bag for k in ["structure", "disorder", "site", "occupancy", "phase", "xrd", "rietveld"]):
                    tags.append("structure")
                return tags or ["general"]

            picked = []
            counts = {a: 0 for a in aspects}
            counts["general"] = 0

            for d in docs:
                tags = _doc_aspects(d)
                ok = False
                assigned_aspect = None
                for a in aspects:
                    if a in tags and counts.get(a, 0) < max_per_aspect:
                        ok = True
                        assigned_aspect = a
                        break
                if not ok and counts.get("general", 0) < max_per_aspect:
                    ok = True
                    assigned_aspect = "general"

                if ok:
                    picked.append(d)
                    if assigned_aspect is not None:
                        counts[assigned_aspect] = counts.get(assigned_aspect, 0) + 1
                if len(picked) >= top_k:
                    break

            if len(picked) < top_k:
                seen = set(id(x) for x in picked)
                for d in docs:
                    if id(d) in seen:
                        continue
                    picked.append(d)
                    if len(picked) >= top_k:
                        break
            docs = picked[:top_k]
        else:
            docs = docs[:top_k]

        try:
            fr_cfg = RAG_CONFIG.get("forced_recall") or {}
            if bool(fr_cfg.get("enabled", False)) and rule_hits and any("fluoride" in h.lower() or "inf3" in h.lower() for h in rule_hits):
                topic = "Fluoride"
                fb_terms = [str(x).lower() for x in ((fr_cfg.get("text_fallback_contains") or {}).get(topic) or []) if x]
                def _is_fluoride_doc(d: Dict[str, Any]) -> bool:
                    meta = d.get("metadata", {}) or {}
                    tags = str(meta.get("tags") or "").lower()
                    if "fluoride" in tags:
                        return True
                    txt = (d.get("text") or "").lower()
                    return any(t in txt for t in fb_terms)

                if not any(_is_fluoride_doc(d) for d in (docs or [])):
                    forced_query = str(((fr_cfg.get("queries") or {}).get(topic)) or "").strip()
                    if forced_query:
                        forced_hits = self.vs.search_similar(
                            forced_query,
                            k=max(80, top_k * 20),
                            score_threshold=0.0,
                            use_mmr=False,
                            query_prefix="solid-state battery",
                            kw_query=question,
                        )
                        forced_hits = [h for h in forced_hits if not self.vs._is_reference_like(h.get("text") or "")]
                        forced_hits = [h for h in forced_hits if _is_fluoride_doc(h)]
                        if forced_hits:
                            docs = (forced_hits[:1] + (docs or []))
                            seen = set()
                            uniq = []
                            for d in docs:
                                t = "".join(((d.get("text") or "").lower().split()))
                                if not t:
                                    continue
                                if t in seen:
                                    continue
                                seen.add(t)
                                uniq.append(d)
                            docs = uniq[:top_k]
                            log.info("Final fluoride guarantee: injected 1 fluoride evidence chunk.")
        except Exception as e:
            log.warning(f"Final fluoride guarantee failed: {e}")

        log.info(f"Retrieved {len(docs)} documents")
        
        if not docs:
            log.warning("No documents retrieved from vector database")
            log.warning(f"Database path: {self.vs.persist_dir}")
            log.warning(f"Collection count: {db_stats.get('count', 'unknown')}")

            answer = (
                "The current literature index did not return chunks relevant enough to this question, "
                "so a reliable, evidence-based answer cannot be given from the available corpus."
            )
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "research_question": question,
                "rewritten_question": rewritten_question,
                "expert_analysis": answer,
                "retrieved_documents": [],
                "enhanced_documents": [],
                "enhanced_context": include_context,
                "context_window": context_window,
                "database_stats": db_stats,
                "context_length": 0,
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            return {
                "expert_analysis": answer,
                "scientific_context": "",
                "retrieved_documents": [],
                "enhanced_documents": [],
            }
        
        q_len = len(question)
        if q_len < 200:
            complexity = "short"
        elif q_len < 800:
            complexity = "medium"
        else:
            complexity = "long"
        budget_cfg = RAG_CONFIG.get("context_budget_multipliers", {})
        mul = float(budget_cfg.get(complexity, 1.0))
        base_max_chars = int(RAG_CONFIG.get("enhanced_context_max_chars", 3000))
        adaptive_max_chars = int(base_max_chars * mul)

        if docs and parent_context_used:
            enhanced_docs = docs
            ctx = self._pack_enhanced_context(enhanced_docs, max_chars=adaptive_max_chars)
        elif include_context and docs:
            enhanced_docs = self._get_enhanced_context(docs, context_window)
            log.info(f"Enhanced context with {len(enhanced_docs)} document chunks")
            ctx = self._pack_enhanced_context(enhanced_docs, max_chars=adaptive_max_chars)
        else:
            enhanced_docs = docs
            ctx = self._pack_context(docs)
        
        log.info(f"Context length: {len(ctx)} characters")

        summ_thr = int(RAG_CONFIG.get("context_summarization_threshold", 16000))
        if len(ctx) > summ_thr and bool(RAG_CONFIG.get("enable_context_summarization", False)):
            log.info(f"Context exceeds {summ_thr} chars, summarizing with LLM...")
            ctx = self.llm.summarize_context(ctx)
            log.info(f"Summarized context length: {len(ctx)} characters")

        try:
            llm_prompt = self.llm._build_prompt(question, ctx)
        except Exception:
            llm_prompt = f"Scientific Literature Context:\n{ctx}\n\nResearch Question:\n{question}"
        
        max_out = int(RAG_CONFIG.get("max_output_tokens", 32768))
        effective_max = max_out if max_out > 0 else self.max_new_tokens
        answer = self.llm.generate(question, ctx, max_new_tokens=effective_max, temperature=self.temperature)

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "research_question": question,
            "rewritten_question": rewritten_question,
            "subqueries": subqueries,
            "llm_prompt": llm_prompt,
            "expert_analysis": answer,
            "retrieved_documents": docs[:top_k],
            "enhanced_documents": [
                {
                    "metadata": d.get("metadata", {}),
                    "score": d.get("score", 0.0),
                    "excerpt": d.get("excerpt") or (d.get("text", "")[:200]),
                }
                for d in (enhanced_docs[: top_k * (context_window * 2 + 1)] if enhanced_docs else [])
            ],
            "enhanced_context": include_context,
            "context_window": context_window,
            "database_stats": db_stats,
            "context_length": len(ctx)
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "expert_analysis": answer,
            "scientific_context": ctx,
            "llm_prompt": llm_prompt,
            "retrieved_documents": docs,
            "enhanced_documents": enhanced_docs,
        }



def main():
    parser = argparse.ArgumentParser(description="RAG System for Solid-State Battery")

    parser.add_argument("--db-path",     type=str, default=str(PATHS["chroma_db"]),  help="ChromaDB persistent directory (read/write)")
    parser.add_argument("--runs-dir",    type=str, default=str(PATHS["runs_dir"]),   help="Q&A log save directory")
    
    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model (Sentence-Transformers name or path)")
    
    # Intern S1 API parameters
    parser.add_argument("--api-key", type=str, default=None, help="Intern S1 API key (if None, loads from API/api_keys.json)")
    parser.add_argument("--base-url", type=str, default="https://chat.intern-ai.org.cn/api/v1/", help="Intern S1 API base URL")
    parser.add_argument("--model", type=str, default="intern-s1", help="Intern S1 model name")
    parser.add_argument("--thinking-mode", action="store_true", help="Enable thinking mode for Intern S1")
    
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved documents to return")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens to generate (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (lower = more focused, higher = more creative)")
    
    parser.add_argument("--no-context", action="store_true", help="Disable enhanced context (use simple context only)")
    parser.add_argument("--context-window", type=int, default=1, help="Number of adjacent chunks to include for context (default: 1)")
    
    parser.add_argument("--question", type=str, default=None, help="Ask a question directly (non-interactive)")
    args = parser.parse_args()

    db_path     = Path(args.db_path)
    runs_dir    = Path(args.runs_dir)

    rag = RAG(
        db_path=db_path,
        runs_dir=runs_dir,
        embed_model_id=args.embed_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        thinking_mode=args.thinking_mode
    )

    if args.question:
        include_context = not args.no_context
        context_window = args.context_window
        
        print(f"\n🔍 Question: {args.question}")
        print("=" * 80)
        
        out = rag.ask(
            args.question, 
            top_k=args.top_k,
            include_context=include_context,
            context_window=context_window
        )
        
        print("\n💡 Expert analysis:")
        print("-" * 40)
        print(out["expert_analysis"])
        
        print(f"\n📄 Retrieved documents ({len(out['retrieved_documents'])}):")
        print("-" * 40)
        for i, doc in enumerate(out["retrieved_documents"], 1):
            print(f"  [{i}]")
        
        print(f"\n📊 Context stats:")
        print(f"  Context length: {len(out['scientific_context'])} chars")
        print(f"  Enhanced context: {include_context}")
        print(f"  Context window: {context_window}")
        
    else:
        print("✅ RAG System Ready. Usage Examples:")
        print(f"python rag_system.py --question \"What are the key factors for solid-state electrolyte conductivity?\"")
        print(f"python rag_system.py --question \"What factors influence the interface stability of solid-state batteries?\" --thinking-mode")
        print(f"python rag_system.py --question \"How to maximize the S2 optimization indicator?\" --max-new-tokens 4096")
        print(f"python rag_system.py --question \"What are the key parameters for solid-state battery stability?\" --temperature 0.5")
        print("\n🔧 Context Control Options:")
        print(f"  --no-context           : Use simple context only")
        print(f"  --context-window N     : Include N adjacent chunks (default: 1)")
        print(f"  Example: --context-window 2 --no-context")
        print("\n🔧 Intern S1 API Options:")
        print(f"  --api-key KEY         : API key (default: loads from API/api_keys.json)")
        print(f"  --base-url URL        : API base URL")
        print(f"  --model MODEL         : Model name (default: intern-s1)")
        print(f"  --thinking-mode       : Enable thinking mode")


RAGSystem = RAG

if __name__ == "__main__":
    main()
