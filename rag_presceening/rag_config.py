"""
Core hyperparameters for the RAG system.
Centralized for tuning and experiment comparison.
"""

RAG_CONFIG = {
    # ---------- Rule-enforced sub-queries (must-have queries) ----------
    # When the question names key entities/formulas, append retrieval sub-queries even if
    # LLM decomposition misses them, to improve recall (especially for F-, Br-, I- drowned in long questions).
    #
    # Rule format:
    # - name: label (logging only)
    # - patterns: regexes matched against the lowercased raw question
    # - queries: extra retrieval sub-queries to append when matched (English, short)
    "must_have_query_rules": [
        {
            "name": "F/fluoride",
            "patterns": [r"\binf3\b", r"\bin[-\s]?f3\b", r"fluoride", r"f[-−⁻]"],
            "queries": [
                "F- doping in Li6PS5Cl argyrodite solid electrolyte ionic conductivity",
                "fluoride (F-) substitution or precursor effects in Li6PS5Cl / Li6PS5X argyrodite",
            ],
        },
        {
            "name": "Br/bromide",
            "patterns": [r"\bbr[-\s]?\b", r"bromide", r"\blibr\b"],
            "queries": [
                "Br- doping in Li6PS5Cl / Li6PS5Br argyrodite solid electrolytes and its impact on Li+ ionic conductivity",
            ],
        },
        {
            "name": "I/iodide",
            "patterns": [r"iodide", r"\blii\b", r"\bi[-−⁻]\b", r"i[-−⁻]\s"],
            "queries": [
                "I- (iodide) substitution in Li6PS5Cl / Li6PS5I argyrodite and its effect on anion disorder and Li+ transport",
            ],
        },
        # O/oxide rules removed; only F/Cl/Br/I halogen doping considered here
        {
            "name": "SEI/interface",
            "patterns": [r"sei", r"interphase", r"interface", r"li\s*[\|]\s*electrolyte", r"ccd", r"impedance"],
            "queries": [
                "SEI solid electrolyte interphase formation stability Li6PS5Cl argyrodite with Li metal",
                "halogen dopant F- Cl- Br- I- effect on Li metal interface stability CCD impedance Li6PS5Cl",
            ],
        },
        {
            "name": "LiF",
            "patterns": [r"\blif\b"],
            "queries": [
                "LiF formation and interphase (SEI) in sulfide argyrodite electrolytes with Li metal",
            ],
        },
        {
            "name": "argyrodite/Li6PS5Cl",
            "patterns": [r"argyrodite", r"li6ps5cl", r"li6ps5x"],
            "queries": [
                "Li6PS5Cl argyrodite halogen F Cl Br I anion disorder S X site mixing",
            ],
        },
    ],

    # ---------- Forced recall fallback (when key entities are named, ensure evidence is retrieved) ----------
    # Rule sub-queries may still miss due to vector thresholds or long-question semantics; for matched
    # themes, run one low-threshold, large-pool retrieval and optionally merge after tag/text filtering.
    "forced_recall": {
        "enabled": True,
        "queries": {
            "Fluoride": "fluoride F- InF3 SnF2 doping Li6PS5Cl Li6PS5X argyrodite ionic conductivity",
            "Bromide": "Br- bromide LiBr Br substitution in Li6PS5Cl Li6PS5Br argyrodite ionic conductivity",
            "Iodide": "I- iodide LiI I substitution in Li6PS5Cl Li6PS5I argyrodite Li+ transport",
        },
        "score_threshold": 0.20,
        "pool_multiplier": 12,
        "min_k": 30,
        # If True, keep only entries whose tags include the theme; else query-only recall without hard tag filter
        # Old stores without tags: set False or fallback recall may empty.
        "require_tag_match": False,
        "text_fallback_contains": {
            "Fluoride": ["inf3", "fluoride", "f-", "f\u207b", "lif"],
        },
    },

    # ---------- Anion scope filter (when the question scopes a halogen, drop non-target anion systems) ----------
    # If the question clearly scopes F/Cl/Br/I doping, exclude Se/Te/O and other non-target systems.
    # excluded_anions: key = anion name; value = text fragments and regexes to detect that system in a doc
    "anion_scope_filter": {
        "Se": {
            "text_patterns": ["li6ps5se", "se substitution", "se-doped", "se-substituted", "se doping", "ps5se", "li6pse5"],
            "regex_patterns": [r"\bse(?!i)\b"],  # standalone Se, not SEI
        },
        "Te": {
            "text_patterns": ["li6ps5te", "te substitution", "te-doped", "te-substituted", "ps5te", "li6pte5", "te doping"],
            "regex_patterns": [r"te[- ]?(?:dop|substitut)"],
        },
        "O": {
            "text_patterns": ["li6ps5o", "o2- substitution", "oxide-doped", "oxygen substitution", "o-substituted", "ps5o"],
            "regex_patterns": [r"li6p[so]5o\b", r"argyrodite.*o2[-−⁻].*substitut"],
        },
    },

    # ---------- Title-locked papers → best chunk per paper (halogen evidence recall) ----------
    # When F/Cl/Br/I are mentioned, match original_title in processed_papers.json, then search chunks with where=filename
    # and inject the best chunk into the candidate / final top_k.
    "title_guided_recall": {
        "enabled": True,
        "trigger_terms": ["fluoride", "f-", "f\u207b", "chloride", "cl-", "bromide", "br-", "iodide", "i-"],
        "max_papers": 3,
        "title_keywords": {
            "Fluoride": ["fluoride", "inf3", "f-", "f\u207b"],
            "Chloride": ["chloride", "cl-", "licl"],
            "Bromide": ["bromide", "br-", "libr"],
            "Iodide": ["iodide", "i-", "lii"],
            "Core": ["li6ps5cl", "li6ps5x", "argyrodite"],
        },
        "per_paper": {
            "score_threshold": 0.0,
            "k": 10
        }
    },

    # ---------- Chunking ----------
    "chunk_semantic_size": 800,
    "chunk_semantic_overlap": 150,
    "chunk_min_size": 50,

    "section_size_factors": {
        "results": 0.8,
        "conclusion": 0.8,
        "methods": 1.1,
        "default": 1.0,
    },

    # ---------- Vector retrieval ----------
    "vector_score_threshold": 0.4,
    "fetch_k_multiplier": 6,
    "safety_min_sim": 0.2,

    "ranking_weights": {
        "sim": 0.6,
        "keywords": 0.2,
        "section": 0.1,
        "tags": 0.1,
    },

    # ---------- Context enrichment ----------
    "max_chunks_per_file": 4,
    "no_snippet_truncation": True,
    "retrieved_snippet_chars": 800,
    "context_snippet_chars": 400,
    "enhanced_context_max_chars": 32000,
    "enable_context_summarization": True,
    "context_summarization_threshold": 16000,

    "context_budget_multipliers": {
        "short": 0.6,
        "medium": 1.0,
        "long": 1.4,
    },

    # ---------- Query rewrite (for retrieval) ----------
    "enable_query_rewrite": True,
    "min_rewrite_length": 80,
    "rewrite_max_new_tokens": 128,

    "enable_query_expansion": True,

    # ---------- Cross-encoder rerank (optional) ----------
    "enable_cross_encoder_rerank": True,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank_max_candidates": 40,
    "cross_encoder_weight": 0.35,

    # ---------- Parent–child (retrieve small chunks, group by parent for context) ----------
    "enable_parent_child": True,
    "use_parent_for_context": True,
    "max_children_per_parent": 2,

    # ---------- Candidate pool and dedup / coverage ----------
    "candidate_pool_multiplier": 6,
    "dedup_by_parent": True,
    "enable_text_dedup": True,
    "text_dedup_jaccard_threshold": 0.85,
    "enable_coverage_selection": True,
    "max_per_aspect": 2,

    # ---------- Generation length ----------
    "max_output_tokens": 32768,

    # ---------- Citations ----------
    "require_citations": True,
    "min_citations": 2,
    "citation_retry_max": 1,

    # ---------- Question decomposition (multi-query) ----------
    "enable_decomposition": True,
    "min_decompose_length": 250,
    "max_subqueries": 4,
    "per_subquery_top_k": 6,
}
