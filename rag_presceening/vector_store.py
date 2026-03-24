"""
vector_store.py
Ingest processed paper text into ChromaDB and expose search.
- Load processed paper JSON
- Embeddings: sentence-transformers all-MiniLM-L6-v2 (normalized, cosine)
- Persist to the vector DB
- Search returns text + metadata + similarity + excerpt
"""

from __future__ import annotations
import json
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Union

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from .rag_config import RAG_CONFIG
except ImportError:
    from rag_config import RAG_CONFIG

_BASE_DIR = Path(__file__).parent.resolve()
MODEL_BASE_PATH = str(_BASE_DIR / "models")
VECTOR_DB_PATH = str(_BASE_DIR / "data" / "chroma_db")
PROCESSED_DOCS_PATH = str(_BASE_DIR / "data")
PAPERS_DIR = str(_BASE_DIR / "docs" / "papers")

from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None
from tqdm import tqdm

import chromadb
try:
    from chromadb import PersistentClient as _PersistentClient  # 0.5+
except Exception:  # pragma: no cover
    _PersistentClient = None

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_dir: str = VECTOR_DB_PATH,
        collection_name: str = "papers",
        model_name: str = "all-MiniLM-L6-v2",
        reset_collection: bool = False,
        chunking_strategy: str = "semantic",
    ) -> None:
        """
        Args:
            persist_dir: Chroma persist directory
            collection_name: Collection name
            model_name: Sentence-Transformers embedding model
            reset_collection: If True, drop and recreate (default False)
            chunking_strategy: One of "semantic", "smart", "section", "paragraph", "fixed"
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.chunking_strategy = chunking_strategy
        self.enable_parent_child = bool(RAG_CONFIG.get("enable_parent_child", True))

        if _PersistentClient is not None:
            self.client = _PersistentClient(path=str(self.persist_dir))
            self.child_collection_name = f"{self.collection_name}__child" if self.enable_parent_child else self.collection_name
            self.parent_collection_name = f"{self.collection_name}__parent"

            self.collection = self.client.get_or_create_collection(
                name=self.child_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.parent_collection = self.client.get_or_create_collection(
                name=self.parent_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._has_upsert = hasattr(self.collection, "upsert")
            chroma_ver = "0.5+"
        else:
            from chromadb.config import Settings
            self.client = chromadb.Client(
                Settings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=str(self.persist_dir),
                    chroma_db_impl="duckdb+parquet",
                )
            )
            existing = [c.name for c in self.client.list_collections()]
            self.child_collection_name = f"{self.collection_name}__child" if self.enable_parent_child else self.collection_name
            self.parent_collection_name = f"{self.collection_name}__parent"
            if reset_collection and self.collection_name in existing:
                self.client.delete_collection(self.collection_name)
            if self.child_collection_name not in [c.name for c in self.client.list_collections()]:
                self.collection = self.client.create_collection(
                    name=self.child_collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                self.collection = self.client.get_collection(self.child_collection_name)

            if self.parent_collection_name not in [c.name for c in self.client.list_collections()]:
                self.parent_collection = self.client.create_collection(
                    name=self.parent_collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                self.parent_collection = self.client.get_collection(self.parent_collection_name)
            self._has_upsert = False
            chroma_ver = "0.4.x"

        logger.info(f"[VectorStore] ChromaDB = {chroma_ver}, persist_dir = {self.persist_dir}")
        logger.info(
            f"[VectorStore] collections: child={getattr(self, 'child_collection_name', self.collection_name)}, "
            f"parent={getattr(self, 'parent_collection_name', 'N/A')}"
        )

        logger.info(f"[VectorStore] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("[VectorStore] Embedding model is ready.")

        self._cross_encoder = None
        if bool(RAG_CONFIG.get("enable_cross_encoder_rerank", False)) and CrossEncoder is not None:
            try:
                ce_name = RAG_CONFIG.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info(f"[VectorStore] Loading cross-encoder reranker: {ce_name}")
                self._cross_encoder = CrossEncoder(ce_name)
            except Exception as e:
                logger.warning(f"[VectorStore] Cross-encoder init failed, fallback to rule rerank: {e}")

        self.default_chunk_size = 1000
        self.default_overlap = 200
        self.batch_size = 256
        
        logger.info(f"[VectorStore] ready, ChromaDB: {chroma_ver}")
        logger.info(f"[VectorStore] embedding model: {model_name}")
        logger.info(f"[VectorStore] chunking: {chunking_strategy}")
        logger.info(f"[VectorStore] persist_dir: {self.persist_dir}")

    @staticmethod
    def _hash_id(*parts: str) -> str:
        h = hashlib.md5()
        for p in parts:
            h.update(str(p).encode("utf-8", errors="ignore"))
        return h.hexdigest()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Sentence split: .?! and CJK sentence-ending punctuation; weak breaks on ; / fullwidth ; / enumeration mark.
        """
        strong_pattern = r'[。！？.!?]\s*'
        parts = re.split(strong_pattern, text)
        sentences = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) > 800:
                weak_parts = re.split(r'[;；]\s*|、\s*', p)
                seg = ""
                for wp in weak_parts:
                    wp = wp.strip()
                    if not wp:
                        continue
                    if len(seg) + len(wp) <= 500:
                        seg += wp + "；"
                    else:
                        if seg:
                            sentences.append(seg.strip())
                        seg = wp + "；"
                if seg:
                    sentences.append(seg.strip())
            else:
                sentences.append(p)
        return sentences

    _SECTION_PATTERN = re.compile(
        r'(?m)^('
        r'Abstract|Introduction|Background|'
        r'Methods?|Materials?|Experimental|'
        r'Results?|Discussion|Conclusions?|'
        r'References?|Appendix|Supporting Information|'
        r'\d+\.\s+[^\n]+|\d+\.\d+(?:\.\d+)?\s+[^\n]+'  # numbered headings e.g. 1. / 2.1.1
        r')\s*$',
        re.IGNORECASE
    )

    @staticmethod
    def _is_reference_like(text: str) -> bool:
        """
        Heuristic: detect bibliography-style paragraphs without relying on section/type.
        Typical signals: citation-like line starts, years, bib terms, few body keywords.
        """
        if not text:
            return False
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return False

        import re as _re
        year_pattern = _re.compile(r"\b(19|20)\d{2}\b")
        cite_head_pattern = _re.compile(r"^(\[\d+\]|\d+\.\s+|\(\d+\)\s+)")

        ref_like_lines = 0
        year_hits = 0
        for l in lines[: min(20, len(lines))]:
            if cite_head_pattern.match(l):
                ref_like_lines += 1
            if year_pattern.search(l):
                year_hits += 1

        text_lower = text.lower()
        bib_hits = 0
        for kw in [" et al.", "doi", "vol.", "volume", "pp.", "pages", "journal", "proc.", "proceedings", "isbn", "editor"]:
            if kw in text_lower:
                bib_hits += 1

        line_ratio = ref_like_lines / max(1, min(20, len(lines)))

        if line_ratio >= 0.2 or year_hits >= 2 or bib_hits >= 2:
            return True
        return False

    @staticmethod
    def _get_chunk_tags(text: str) -> set:
        """
        Tag chunks from keywords/regex (halogens, conductivity, SEI, argyrodite, P-site, etc.).
        Stored as semicolon-separated string in Chroma metadata.
        """
        if not text:
            return set()
        t = text.lower()
        tags = set()
        if re.search(r"\b(inf3|in f3|in-f3)\b", t) or re.search(r"\bf[-−]\b|fluoride|fluorination|lif\b", t):
            tags.add("Fluoride")
        if re.search(r"\bcl[-−]\b|chloride|licl\b|argyrodite.*cl", t):
            tags.add("Chloride")
        if re.search(r"\bbr[-−]\b|bromide|libr\b", t):
            tags.add("Bromide")
        if re.search(r"\b[Ii][-−]\b|iodide|lii\b", t):
            tags.add("Iodide")
        if re.search(r"\bo2[-−]\b|oxygen-containing|\boxide\b|\boxygen\b", t):
            tags.add("Oxide")
        if re.search(r"\b(ionic|ion)\s+conductivity|conductivity\s*\(|σ\s*=|\bsigma\b", t, re.I):
            tags.add("Ionic conductivity")
        if re.search(r"\bsei\b|interphase|interfacial|interface\s", t, re.I):
            tags.add("SEI/Interface")
        if "argyrodite" in t:
            tags.add("Argyrodite")
        if re.search(r"\bp[- ]?site|phosphorus\s+site|substitution\s+on\s+p\b", t, re.I):
            tags.add("P-site")
        return tags

    @staticmethod
    def _infer_section_type(section: str) -> str:
        """
        Coarse section label from heading text for retrieval weighting.
        """
        if not section:
            return "unknown"
        s = section.lower()
        if "abstract" in s:
            return "abstract"
        if any(k in s for k in ["introduction", "background"]):
            return "introduction"
        if any(k in s for k in ["method", "methods", "experiment", "experimental"]):
            return "methods"
        if any(k in s for k in ["result", "discussion"]):
            return "results"
        if "conclusion" in s:
            return "conclusion"
        if any(k in s for k in ["reference", "references"]):
            return "references"
        return "other"

    @staticmethod
    def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Chunk by sentence boundaries; weak breaks on CJK/fullwidth punctuation; split very long sentences.
        """
        blocks: List[Dict[str, Any]] = []
        if not text:
            return blocks

        size = max(1, size)
        overlap = max(0, min(overlap, size - 1))

        sentences = VectorStore._split_sentences(text)
        current_chunk = ""
        char_start = 0
        if len(text) > 10000:
            pbar = tqdm(total=max(1, len(text) // (size - overlap)), desc="chunk_text", unit="chunk", leave=False)

        for sentence in sentences:
            if not sentence:
                continue
            sep = "。" if not sentence.rstrip().endswith(("。", "！", "？", ".", "!", "?")) else ""
            seg = sentence + sep

            if len(current_chunk) + len(seg) <= size:
                current_chunk += seg
            else:
                if current_chunk:
                    char_end = char_start + len(current_chunk)
                    blocks.append({"text": current_chunk.strip(), "char_start": char_start, "char_end": char_end})
                    char_start = char_end - overlap
                current_chunk = seg
                if len(text) > 10000:
                    pbar.update(1)

        if current_chunk:
            char_end = char_start + len(current_chunk)
            blocks.append({"text": current_chunk.strip(), "char_start": char_start, "char_end": char_end})
        if len(text) > 10000:
            pbar.close()
        return blocks

    @staticmethod
    def chunk_text_section(text: str, size: int = 1000, overlap: int = 150) -> List[Dict[str, Any]]:
        """
        Chunk by academic section headings; split long parts with chunk_text.
        """
        blocks: List[Dict[str, Any]] = []
        if not text:
            return blocks
        parts = VectorStore._SECTION_PATTERN.split(text)
        current_chunk = ""
        char_start = 0
        run_offset = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(current_chunk) + len(part) <= size:
                current_chunk += part + "\n\n"
            else:
                if current_chunk:
                    char_end = char_start + len(current_chunk)
                    blocks.append({"text": current_chunk.strip(), "char_start": char_start, "char_end": char_end})
                    run_offset = char_end
                    char_start = char_end - overlap
                if len(part) > size:
                    sub_chunks = VectorStore.chunk_text(part, size=size, overlap=overlap)
                    for sc in sub_chunks:
                        sc["char_start"] += run_offset
                        sc["char_end"] += run_offset
                        blocks.append(sc)
                    run_offset = blocks[-1]["char_end"]
                    char_start = run_offset - overlap
                    current_chunk = ""
                else:
                    current_chunk = part + "\n\n"
        if current_chunk:
            char_end = char_start + len(current_chunk)
            blocks.append({"text": current_chunk.strip(), "char_start": char_start, "char_end": char_end})
        return blocks

    @staticmethod
    def chunk_text_paragraph(text: str, max_size: int = 1000) -> List[Dict[str, Any]]:
        """Chunk by paragraph (double newlines)."""
        blocks: List[Dict[str, Any]] = []
        if not text:
            return blocks
        
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        char_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) <= max_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    char_end = char_start + len(current_chunk)
                    blocks.append({
                        "text": current_chunk.strip(),
                        "char_start": char_start,
                        "char_end": char_end
                    })
                    char_start = char_end
                
                current_chunk = para + "\n\n"
        
        if current_chunk:
            char_end = char_start + len(current_chunk)
            blocks.append({
                "text": current_chunk.strip(),
                "char_start": char_start,
                "char_end": char_end
            })
            
        return blocks

    @classmethod
    def chunk_text_semantic(cls, text: str, size: int = 800, overlap: int = 150, min_size: int = 50) -> List[Dict[str, Any]]:
        """
        Semantic chunking: section → paragraph → sentence; drops very short chunks.
        Emits parent blocks and child retrieval blocks with section metadata.
        """
        blocks: List[Dict[str, Any]] = []
        if not text or len(text.strip()) < min_size:
            return blocks
        cfg_size = RAG_CONFIG.get("chunk_semantic_size", size)
        cfg_overlap = RAG_CONFIG.get("chunk_semantic_overlap", overlap)
        base_size = max(cfg_size, min_size * 2)
        base_overlap = max(0, min(cfg_overlap, base_size - min_size))

        parts = cls._SECTION_PATTERN.split(text)
        current_section = ""
        char_start = 0
        parent_idx = 0

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 1:
                current_section = part[:80]
                continue
            paragraphs = [p.strip() for p in part.split("\n\n") if len(p.strip()) >= min_size]
            for para in paragraphs:
                para_start = char_start
                section_type = cls._infer_section_type(current_section)
                factors = RAG_CONFIG.get("section_size_factors", {})
                factor = factors.get(section_type, factors.get("default", 1.0))
                local_size = int(base_size * factor)
                local_overlap = int(base_overlap * (0.8 if section_type in ("results", "conclusion") else 1.0))

                local_size = max(min_size * 2, local_size)
                local_overlap = max(0, min(local_overlap, local_size - min_size))

                para_end = para_start + len(para)
                parent_key = f"{parent_idx}:{para_start}:{para_end}"
                parent_idx += 1
                blocks.append(
                    {
                        "text": para,
                        "char_start": para_start,
                        "char_end": para_end,
                        "section": current_section,
                        "section_type": section_type,
                        "parent_key": parent_key,
                        "parent_char_start": para_start,
                        "parent_char_end": para_end,
                        "is_parent": True,
                    }
                )

                if len(para) <= local_size:
                    blocks.append(
                        {
                            "text": para,
                            "char_start": para_start,
                            "char_end": para_end,
                            "section": current_section,
                            "section_type": section_type,
                            "parent_key": parent_key,
                            "parent_char_start": para_start,
                            "parent_char_end": para_end,
                            "is_parent": False,
                        }
                    )
                    char_start = para_end - local_overlap
                else:
                    sub_chunks = cls.chunk_text(para, size=local_size, overlap=local_overlap)
                    last_end = para_end
                    for sc in sub_chunks:
                        if len(sc["text"]) < min_size:
                            continue
                        sc_start = para_start + sc["char_start"]
                        sc_end = para_start + sc["char_end"]
                        blocks.append(
                            {
                                "text": sc["text"],
                                "char_start": sc_start,
                                "char_end": sc_end,
                                "section": current_section,
                                "section_type": section_type,
                                "parent_key": parent_key,
                                "parent_char_start": para_start,
                                "parent_char_end": para_end,
                                "is_parent": False,
                            }
                        )
                        last_end = sc_end
                    char_start = last_end - local_overlap
        return blocks

    @staticmethod
    def _chunk_text_fixed(text: str, size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Fixed-size overlapping windows (fallback)."""
        blocks: List[Dict[str, Any]] = []
        if not text:
            return blocks
        
        step = max(1, size - overlap)
        n = len(text)
        i = 0
        
        while i < n:
            j = min(i + size, n)
            chunk = text[i:j].strip()
            if chunk:
                blocks.append({"text": chunk, "char_start": i, "char_end": j})
            if j == n:
                break
            i += step
            
        return blocks

    def _iter_docs(self, processed_json_path: str) -> Iterable[Dict[str, Any]]:
        p = Path(processed_json_path)
        if not p.exists():
            raise FileNotFoundError(f"Processed papers JSON not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            docs = json.load(f)
        logger.info(f"[VectorStore] documents to ingest: {len(docs)}")
        for d in docs:
            yield d

    def _flush_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        if not documents:
            return 0
        embeddings = self.model.encode(
            documents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        try:
            if self._has_upsert:
                self.collection.upsert(
                    ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist()
                )
            else:
                self.collection.add(
                    ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist()
                )
        except Exception as ex:
            msg = str(ex).lower()
            if any(k in msg for k in ("exists", "duplicate", "already")):
                logger.warning("[VectorStore] duplicate IDs in batch, skipped.")
            else:
                logger.error(f"[VectorStore] batch write failed: {ex}")
                raise
        return len(documents)

    def _flush_batch_parent(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        if not documents:
            return 0
        embeddings = self.model.encode(
            documents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        try:
            if hasattr(self.parent_collection, "upsert"):
                self.parent_collection.upsert(
                    ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist()
                )
            else:
                self.parent_collection.add(
                    ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist()
                )
        except Exception as ex:
            msg = str(ex).lower()
            if any(k in msg for k in ("exists", "duplicate", "already")):
                logger.warning("[VectorStore] parent collection duplicate IDs skipped.")
            else:
                logger.error(f"[VectorStore] parent write failed: {ex}")
                raise
        return len(documents)

    def ingest_processed_json(
        self,
        processed_json_path: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read processed_papers.json, chunk, embed, write to the vector store (streaming batches).
        """
        if processed_json_path is None:
            processed_json_path = str(Path(PROCESSED_DOCS_PATH) / "processed_papers.json")
        csize = int(chunk_size or self.default_chunk_size)
        ovlp = int(overlap or self.default_overlap)

        total_added = 0
        total_parents_added = 0
        ids_buf: List[str] = []
        docs_buf: List[str] = []
        metas_buf: List[Dict[str, Any]] = []
        p_ids_buf: List[str] = []
        p_docs_buf: List[str] = []
        p_metas_buf: List[Dict[str, Any]] = []

        for d in tqdm(self._iter_docs(processed_json_path), desc="ingest docs", unit="doc"):
            meta = d.get("metadata", {})
            filename = meta.get("filename", "unknown.pdf")
            original_title = meta.get("original_title") or meta.get("title") or Path(filename).stem
            title = original_title
            full_text = d.get("full_text", "") or ""
            doc_keywords = d.get("keywords", []) or []
            
            if self.chunking_strategy == "semantic":
                chunks = self.chunk_text_semantic(full_text, size=csize, overlap=ovlp, min_size=50)
            elif self.chunking_strategy == "smart":
                chunks = self.chunk_text(full_text, size=csize, overlap=ovlp)
            elif self.chunking_strategy == "section":
                chunks = self.chunk_text_section(full_text, size=csize, overlap=ovlp)
            elif self.chunking_strategy == "paragraph":
                chunks = self.chunk_text_paragraph(full_text, max_size=csize)
            else:  # "fixed"
                chunks = self._chunk_text_fixed(full_text, size=csize, overlap=ovlp)

            if self.enable_parent_child and self.chunking_strategy == "semantic":
                parent_counter = 0
                for i, ch in enumerate(chunks):
                    parent_key = ch.get("parent_key")
                    is_ref = self._is_reference_like(ch["text"])
                    p_start = ch.get("parent_char_start", ch.get("char_start"))
                    p_end = ch.get("parent_char_end", ch.get("char_end"))
                    parent_id = self._hash_id(filename, "parent", str(parent_key or ""), str(p_start), str(p_end))

                    if ch.get("is_parent"):
                        p_ids_buf.append(parent_id)
                        p_docs_buf.append(ch["text"])
                        p_meta = {
                            "filename": filename,
                            "title": title,
                            "original_title": original_title,
                            "parent_id": parent_id,
                            "parent_key": parent_key,
                            "char_start": p_start,
                            "char_end": p_end,
                            "extraction_method": meta.get("extraction_method", ""),
                        }
                        if ch.get("section"):
                            p_meta["section"] = ch["section"]
                        if ch.get("section_type"):
                            p_meta["section_type"] = ch["section_type"]
                        p_meta["is_reference"] = is_ref
                        p_meta["tags"] = "; ".join(sorted(self._get_chunk_tags(ch["text"])))
                        if doc_keywords:
                            p_meta["keywords"] = "; ".join(doc_keywords)
                        p_metas_buf.append(p_meta)
                        parent_counter += 1

                        # flush parent batch
                        if len(p_docs_buf) >= self.batch_size:
                            total_parents_added += self._flush_batch_parent(p_ids_buf, p_docs_buf, p_metas_buf)
                            p_ids_buf.clear(); p_docs_buf.clear(); p_metas_buf.clear()
                        continue

                    doc_id = self._hash_id(filename, str(i), str(ch["char_start"]), str(ch["char_end"]))
                    ids_buf.append(doc_id)
                    docs_buf.append(ch["text"])
                    meta_entry = {
                        "filename": filename,
                        "title": title,
                        "original_title": original_title,
                        "chunk_id": i,
                        "char_start": ch["char_start"],
                        "char_end": ch["char_end"],
                        "extraction_method": meta.get("extraction_method", ""),
                    }
                    if ch.get("section"):
                        meta_entry["section"] = ch["section"]
                    if ch.get("section_type"):
                        meta_entry["section_type"] = ch["section_type"]
                    if parent_key is not None:
                        meta_entry["parent_id"] = parent_id
                        meta_entry["parent_key"] = parent_key
                    meta_entry["tags"] = "; ".join(sorted(self._get_chunk_tags(ch["text"])))
                    if doc_keywords:
                        meta_entry["keywords"] = "; ".join(doc_keywords)
                    meta_entry["is_reference"] = is_ref
                    metas_buf.append(meta_entry)

                    if len(docs_buf) >= self.batch_size:
                        total_added += self._flush_batch(ids_buf, docs_buf, metas_buf)
                        ids_buf.clear(); docs_buf.clear(); metas_buf.clear()
            else:
                for i, ch in enumerate(chunks):
                    is_ref = self._is_reference_like(ch["text"])
                    doc_id = self._hash_id(filename, str(i), str(ch["char_start"]), str(ch["char_end"]))
                    ids_buf.append(doc_id)
                    docs_buf.append(ch["text"])
                    meta_entry = {
                        "filename": filename,
                        "title": title,
                        "original_title": original_title,
                        "chunk_id": i,
                        "char_start": ch["char_start"],
                        "char_end": ch["char_end"],
                        "extraction_method": meta.get("extraction_method", ""),
                    }
                    if ch.get("section"):
                        meta_entry["section"] = ch["section"]
                    if ch.get("section_type"):
                        meta_entry["section_type"] = ch["section_type"]
                    meta_entry["tags"] = "; ".join(sorted(self._get_chunk_tags(ch["text"])))
                    if doc_keywords:
                        meta_entry["keywords"] = "; ".join(doc_keywords)
                    meta_entry["is_reference"] = is_ref
                    metas_buf.append(meta_entry)

                if len(docs_buf) >= self.batch_size:
                    total_added += self._flush_batch(ids_buf, docs_buf, metas_buf)
                    ids_buf.clear(); docs_buf.clear(); metas_buf.clear()

        total_added += self._flush_batch(ids_buf, docs_buf, metas_buf)
        if p_docs_buf:
            total_parents_added += self._flush_batch_parent(p_ids_buf, p_docs_buf, p_metas_buf)

        try:
            if hasattr(self.client, "persist"):
                self.client.persist()
        except Exception:
            pass

        logger.info(f"[VectorStore] ingest done, child vectors ~ {total_added}")
        if self.enable_parent_child:
            logger.info(f"[VectorStore] ingest parents ~ {total_parents_added}")
        return {
            "added": int(total_added),
            "collection": self.collection.name,
            "persist_dir": str(self.persist_dir),
        }

    def get_parent_by_id(self, parent_id: str, excerpt_chars: int = 400) -> Optional[Dict[str, Any]]:
        try:
            res = self.parent_collection.get(
                where={"parent_id": {"$eq": parent_id}},
                include=["documents", "metadatas"],
            )
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            if not docs:
                return None
            doc = docs[0]
            meta = metas[0] or {}
            txt = (doc or "").replace("\n", " ").strip()
            excerpt = txt[:excerpt_chars] + ("…" if len(txt) > excerpt_chars else "")
            return {"text": doc, "metadata": meta, "score": 1.0, "excerpt": excerpt}
        except Exception as e:
            logger.debug(f"[VectorStore] get_parent_by_id failed: {e}")
            return None

    def get_chunk_by_metadata(
        self,
        filename: str,
        chunk_id: int,
        excerpt_chars: int = 180,
    ) -> Optional[Dict[str, Any]]:
        """
        Exact chunk lookup by metadata (neighbor expansion; faster than semantic search).
        """
        try:
            where_filter = {
                "$and": [
                    {"filename": {"$eq": filename}},
                    {"chunk_id": {"$eq": chunk_id}},
                ]
            }
            res = self.collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            if not docs:
                return None
            doc = docs[0]
            meta = metas[0]
            txt = (doc or "").replace("\n", " ").strip()
            excerpt = txt[:excerpt_chars] + ("…" if len(txt) > excerpt_chars else "")
            return {
                "text": doc,
                "metadata": meta,
                "score": 1.0,
                "excerpt": excerpt,
            }
        except Exception as e:
            logger.debug(f"[VectorStore] get_chunk_by_metadata failed: {e}")
            return None

    def search_similar(
        self,
        query: str,
        k: int = 5,
        use_query_embeddings: bool = True,
        excerpt_chars: int = 180,
        score_threshold: Optional[float] = 0.25,
        where: Optional[Dict[str, Any]] = None,
        use_mmr: bool = True,
        mmr_lambda: float = 0.6,
        fetch_k: Optional[int] = None,
        query_prefix: Optional[str] = None,
        kw_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cosine similarity search; returns text, metadata, score in [0,1], excerpt.
        """
        mult = RAG_CONFIG.get("fetch_k_multiplier", 6)
        fetch_k = fetch_k or (k * mult)
        query_params: Dict[str, Any] = {
            "n_results": min(fetch_k, 100),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        embed_query = query
        if query_prefix:
            embed_query = f"{query_prefix}: {query}"
        if use_query_embeddings:
            qvec = self.model.encode([embed_query], convert_to_numpy=True, normalize_embeddings=True)
            query_params["query_embeddings"] = qvec.tolist()
        else:
            query_params["query_texts"] = [query]

        res = self.collection.query(**query_params)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        results: List[Dict[str, Any]] = []
        score_threshold = score_threshold if score_threshold is not None else RAG_CONFIG.get("vector_score_threshold", 0.25)
        for doc, meta, dist in zip(docs, metas, dists):
            try:
                dist_f = float(dist)
            except Exception:
                dist_f = 1.0
            sim = max(0.0, min(1.0, 1.0 - dist_f))
            if score_threshold is not None and sim < score_threshold:
                continue
            if (meta or {}).get("is_reference"):
                continue
            if self._is_reference_like(doc or ""):
                continue
            txt = (doc or "").replace("\n", " ").strip()
            excerpt = txt[:excerpt_chars] + ("…" if len(txt) > excerpt_chars else "")
            results.append({"text": doc, "metadata": meta or {}, "score": sim, "excerpt": excerpt})

        kw_q = (kw_query or query or "").strip()
        def _keyword_overlap_score(q: str, meta: Dict[str, Any], text_snippet: str) -> float:
            q_lower = q.lower()
            import re as _re
            q_tokens = [t for t in _re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", q_lower) if len(t) >= 3]
            if not q_tokens:
                return 0.0
            score = 0.0
            kw_raw = meta.get("keywords") or []
            if isinstance(kw_raw, str):
                kw_list = [x.strip() for x in kw_raw.split(";") if x.strip()]
            else:
                kw_list = list(kw_raw) if kw_raw else []
            kw_text = " ".join(kw_list).lower()
            snippet_lower = text_snippet.lower()
            for tok in q_tokens:
                if tok in kw_text:
                    score += 1.0
                elif tok in snippet_lower:
                    score += 0.5
            return min(1.0, score / max(3.0, len(q_tokens)))

        def _section_weight(meta: Dict[str, Any]) -> float:
            sec_type = (meta.get("section_type") or "").lower()
            if sec_type == "results":
                return 1.0
            if sec_type == "conclusion":
                return 0.9
            if sec_type in ("abstract", "introduction"):
                return 0.7
            if sec_type == "methods":
                return 0.5
            return 0.4  # other/unknown

        def _tag_overlap_score(q: str, meta: Dict[str, Any]) -> float:
            """Boost chunks whose tags match question themes (halogens, SEI, etc.)."""
            q_lower = q.lower()
            tags_raw = meta.get("tags")
            if not tags_raw:
                return 0.0
            chunk_tags = set()
            if isinstance(tags_raw, str):
                chunk_tags = {x.strip() for x in tags_raw.split(";") if x.strip()}
            elif isinstance(tags_raw, (list, tuple)):
                chunk_tags = set(str(x).strip() for x in tags_raw if x)
            if not chunk_tags:
                return 0.0
            score = 0.0
            if ("fluoride" in q_lower or "f-" in q_lower or "inf3" in q_lower) and "Fluoride" in chunk_tags:
                score += 1.0
            if ("chloride" in q_lower or "cl-" in q_lower) and "Chloride" in chunk_tags:
                score += 1.0
            if ("bromide" in q_lower or "br-" in q_lower) and "Bromide" in chunk_tags:
                score += 1.0
            if ("iodide" in q_lower or "i-" in q_lower) and "Iodide" in chunk_tags:
                score += 1.0
            if ("oxygen" in q_lower or "oxide" in q_lower or "o2-" in q_lower) and "Oxide" in chunk_tags:
                score += 1.0
            if ("conductivity" in q_lower or "ionic" in q_lower or "ion transport" in q_lower) and "Ionic conductivity" in chunk_tags:
                score += 1.0
            if ("sei" in q_lower or "interphase" in q_lower or "interface" in q_lower) and "SEI/Interface" in chunk_tags:
                score += 1.0
            if "argyrodite" in q_lower and "Argyrodite" in chunk_tags:
                score += 1.0
            if ("p-site" in q_lower or "p site" in q_lower or "phosphorus site" in q_lower) and "P-site" in chunk_tags:
                score += 1.0
            return min(1.0, score)

        reranked: List[Dict[str, Any]] = []
        w_cfg = RAG_CONFIG.get("ranking_weights", {})
        w_sim = float(w_cfg.get("sim", 0.6))
        w_kw = float(w_cfg.get("keywords", 0.2))
        w_sec = float(w_cfg.get("section", 0.1))
        w_tag = float(w_cfg.get("tags", 0.1))
        for r in results:
            meta = r.get("metadata", {}) or {}
            base_sim = float(r.get("score", 0.0))
            kw_score = _keyword_overlap_score(kw_q, meta, r.get("text") or "")
            sec_w = _section_weight(meta)
            tag_score = _tag_overlap_score(kw_q, meta)
            combined = w_sim * base_sim + w_kw * kw_score + w_sec * sec_w + w_tag * tag_score
            r["ranking_score"] = combined
            reranked.append(r)

        if self._cross_encoder is not None and reranked:
            try:
                max_cand = int(RAG_CONFIG.get("rerank_max_candidates", 40))
                cand = reranked[:max_cand]
                pairs = [(kw_q, (c.get("text") or "")[:1200]) for c in cand]
                ce_scores = self._cross_encoder.predict(pairs)
                ce_w = float(RAG_CONFIG.get("cross_encoder_weight", 0.35))
                import numpy as np
                ce_scores = 1 / (1 + np.exp(-np.array(ce_scores)))
                for c, s in zip(cand, ce_scores.tolist()):
                    c["cross_encoder_score"] = float(s)
                    c["ranking_score"] = (1 - ce_w) * float(c.get("ranking_score", 0.0)) + ce_w * float(s)
                reranked = sorted(reranked, key=lambda x: x.get("ranking_score", 0.0), reverse=True)
            except Exception as e:
                logger.warning(f"[VectorStore] cross-encoder rerank failed, fallback: {e}")

        reranked.sort(key=lambda x: x.get("ranking_score", x.get("score", 0.0)), reverse=True)

        safety_min_sim = float(RAG_CONFIG.get("safety_min_sim", 0.0))
        if safety_min_sim > 0 and reranked:
            max_sim = max(float(r.get("score", 0.0)) for r in reranked)
            if max_sim < safety_min_sim:
                logger.warning(
                    f"[VectorStore] All candidates below safety_min_sim={safety_min_sim:.3f}, "
                    f"max_sim={max_sim:.3f}. Returning no results."
                )
                return []

        return reranked[:k]

    def _mmr_diversify(
        self,
        query: str,
        results: List[Dict[str, Any]],
        k: int,
        lambda_: float,
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance: balance relevance vs diversity.
        """
        import numpy as np
        if not results or k >= len(results):
            return results[:k]
        selected: List[Dict[str, Any]] = []
        selected_vecs: List[Any] = []
        remaining = list(results)
        qvec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        rvecs = self.model.encode([r["text"] for r in remaining], convert_to_numpy=True, normalize_embeddings=True)

        for _ in range(k):
            if not remaining:
                break
            best_idx = -1
            best_mmr = -1e9
            for idx in range(len(remaining)):
                rel = float(np.dot(qvec, rvecs[idx]))
                if selected_vecs:
                    max_sim = max(float(np.dot(rvecs[idx], sv)) for sv in selected_vecs)
                else:
                    max_sim = 0.0
                mmr = lambda_ * rel - (1 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            if best_idx >= 0:
                chosen = remaining.pop(best_idx)
                chosen_vec = rvecs[best_idx:best_idx + 1].copy()
                rvecs = np.delete(rvecs, best_idx, axis=0)
                selected.append(chosen)
                selected_vecs.append(chosen_vec[0])
        return selected

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
        except Exception:
            count = None
        return {
            "collection": self.collection.name,
            "collection_name": self.collection.name,
            "persist_dir": str(self.persist_dir),
            "count": count,
            "total_chunks": count,
            "embedding_model": getattr(self.model, "name", "all-MiniLM-L6-v2") if self.model else "N/A",
        }


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    vs = VectorStore(
        persist_dir=VECTOR_DB_PATH,
        collection_name="papers",
        reset_collection=False,
        chunking_strategy="smart",
    )

    processed_path = str(Path(PROCESSED_DOCS_PATH) / "processed_papers.json")
    stats_before = vs.get_collection_stats()
    if Path(processed_path).exists() and (stats_before.get("count") or 0) == 0:
        info = vs.ingest_processed_json(processed_path)
        print("ingest info:", info)

    print("stats:", vs.get_collection_stats())

    demo_qs = [
        "solid-state electrolyte ion conductivity",
        "solid-state battery interface stability",
        "solid electrolyte material composition",
    ]
    for q in demo_qs:
        print("\n" + "="*80)
        print(f"🔍 Query: {q}")
        print("="*80)
        hits = vs.search_similar(q, k=5)
        for i, h in enumerate(hits, 1):
            meta = h["metadata"]
            print(f"\n📄 [{i}] file: {meta.get('filename')}")
            print(f"   📍 chunk_id: {meta.get('chunk_id')} | score: {h['score']:.3f}")
            print(f"   📍 char range: {meta.get('char_start')}-{meta.get('char_end')}")
            print(f"   📝 full text:")
            print("-" * 60)
            print(h["text"])
            print("-" * 60)
