#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch RAG over preset questions.
Runs each preset question through retrieval and answering without interactive input.

From project root (D:\\formation_E):
  python -m rag_presceening.batch_qa
or:
  python rag_presceening/batch_qa.py
"""

import sys
import json
from datetime import datetime
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = [
    # Q1: Halogen doping and ionic conductivity (F/Cl/Br/I only)
    """In Li₆PS₅Cl argyrodite solid electrolytes, evaluate how different dopant-related halogen anion chemistries introduced via substitution (e.g., F⁻, Cl⁻, Br⁻, I⁻) affect Li⁺ ionic transport.
    The scope of this question is STRICTLY limited to F⁻, Cl⁻, Br⁻, and I⁻ in Li₆PS₅Cl / Li₆PS₅X argyrodite systems.
    Make an evidence-backed judgment on:
    1) Which anion chemistries are MOST compatible with maintaining or improving ionic conductivity, and
    2) Which anion chemistries are LEAST compatible (i.e., tend to reduce conductivity).

    REQUIRED DECISION OUTPUT:
    - Provide a ranked list of the considered anion chemistries from most to least favorable for high Li⁺ conductivity in argyrodite compositions.
    - For each ranked item, cite representative evidence.

    In your rationale, explicitly address how each anion chemistry influences:
    (i) anion-site disorder and S/X mixing,
    (ii) Li sublattice occupancy, vacancy/interstitial formation, and percolation pathways,
    (iii) activation energy and room-temperature conductivity trends.

    If evidence is mixed or indirect, state the uncertainty and distinguish direct measurements from hypothesized mechanisms.""",
    
    # Q2: Doping vs SEI / interface (F/Cl/Br/I only; no Se systems)
    """For Li₆PS₅Cl argyrodite solid electrolytes in direct contact with Li metal, evaluate how different dopant-related halogen anion chemistries introduced via substitution and/or anion-containing dopant precursors (e.g., F⁻, Cl⁻, Br⁻, I⁻) influence the formation and stability of the solid–electrolyte interphase (SEI) / interphase.
    The scope of this question is STRICTLY limited to F⁻, Cl⁻, Br⁻, and I⁻ in Li₆PS₅Cl / Li₆PS₅X argyrodite systems.
    Based strictly on reported experimental evidence (primarily in Li₆PS₅Cl; using closely related Li₆PS₅X argyrodites only when directly informative), make an evidence-backed judgment on:
    1) Which dopant anion chemistries MOST consistently promote a more stable SEI/interphase and improved Li|electrolyte compatibility, and
    2) Which chemistries are LEAST effective or detrimental (e.g., unstable interphase growth, large impedance rise, poor cycling, low CCD).

    REQUIRED DECISION OUTPUT:
    - Provide a ranked list of the considered anion chemistries from most to least favorable for stable Li-metal interfacial behavior in argyrodite electrolytes.
    - Identify any chemistries that should be considered unsuitable, and state the primary failure mode(s) for each.
    - For each ranked item, cite representative evidence with quantitative or clearly defined metrics, such as:
    • interfacial resistance (R_int) evolution vs time/cycles,
    • symmetric-cell cycling stability (current density, areal capacity, duration, shorting behavior),
    • critical current density (CCD),
    • interphase composition (e.g., XPS/ToF-SIMS/TEM-EDS) and thickness/morphology trends,
    • any reported changes in reduction products or interphase growth kinetics.

    In your rationale, explicitly address:
    (i) interphase chemistry and morphology (identified phases/species, spatial distribution, thickness, uniformity),
    (ii) electrochemical signatures of interfacial stability (R_int trends, polarization, cycling lifetime, CCD),
    (iii) mechanistic links between dopant chemistry and suppressed parasitic reduction and/or mechanically/chemically robust interphase formation.

    If evidence is mixed or indirect, state the uncertainty and clearly separate direct Li-metal contact experiments from proxy measurements or hypothesized mechanisms.
    
    """,
    # Q3: P-site doping and oxidation states (recommend a valence range, e.g. +3 to +6)
    """
    Within Li₆PS₅Cl argyrodite frameworks, assess the feasibility of substituting metal cations onto the P site (i.e., within PS₄ tetrahedral units), and determine a broad, evidence-backed range of cation oxidation states that can be considered feasible or potentially feasible for P-site substitution. 
    The goal is not to define a narrowly optimized valence window, but rather to identify an inclusive and practically reasonable oxidation-state range that is unlikely to miss plausible dopant candidates.

    Make an evidence-backed judgment on:
    1) Which cation oxidation-state ranges are clearly feasible for P-site substitution (i.e., supported by experimental reports of phase-pure or predominantly solid-solution argyrodite),
    2) Which oxidation-state ranges are potentially feasible but condition-dependent, uncertain, or only weakly supported,
    3) Which oxidation-state ranges are generally unsuitable or frequently associated with impurity phases / phase separation / failed incorporation, and
    4) Under what compositional windows or processing conditions the feasibility changes.

    REQUIRED DECISION OUTPUT
    - Provide an inclusive oxidation-state feasibility map for P-site substitution in argyrodite, dividing oxidation states into three categories:
    (A) clearly feasible,
    (B) potentially feasible / conditionally feasible,
    (C) generally unsuitable.
    - Based on this map, give a recommended broad oxidation-state range (for example, +3 to +6) that should be retained for dopant prescreening in order to avoid missing plausible candidates. The final answer MUST explicitly state such a range in the form of \"+x to +y\" (e.g., +3 to +6).
    - For oxidation states outside this recommended range, state whether they should be excluded entirely or only treated as low-confidence possibilities.
    - For each oxidation-state category, cite representative evidence including: substituted element examples, intended site assignment, nominal substitution level, phase analysis (XRD/Rietveld/impurity fraction), and any reported structural indicators consistent with P-site incorporation.

    In your rationale, explicitly address the constraints governing feasibility:
    (i) local coordination and structural compatibility with PS₄ tetrahedral environments (e.g., retention or distortion of tetrahedra),
    (ii) size and field-strength / bond ionicity considerations as supported by literature discussion,
    (iii) experimentally observed phase stability limits (solid-solution range vs impurity formation),
    (iv) charge-compensation mechanisms reported or inferred (Li vacancies/interstitials, anion vacancies, mixed occupancy, coupled substitutions) and how these correlate with phase stability and transport properties.

    If site occupancy is ambiguous in the literature (e.g., possible substitution on Li, S, or halide sites instead of P), explicitly flag this and separate:
    - confirmed P-site substitution,
    - suggested or uncertain P-site substitution,
    with corresponding confidence levels.

    IMPORTANT SCREENING PRINCIPLE
    - Use an inclusion-oriented standard: when literature evidence suggests that an oxidation state may be feasible under certain compositions or synthesis conditions, include it in the “potentially feasible / conditionally feasible” category rather than excluding it.
    - Only classify an oxidation state as “generally unsuitable” when there is consistent evidence of failure, dominant impurity formation, or clear structural incompatibility with P-site substitution.
    - The final recommended oxidation-state range should therefore be intentionally broad and suitable for prescreening, rather than overly restrictive.

    """,
]


def main():
    from rag_presceening import SolidStateRAGSystem

    base_dir = Path(__file__).parent.resolve()
    runs_dir = base_dir / "data" / "rag_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rag = SolidStateRAGSystem(
        papers_dir=str(base_dir / "docs" / "papers"),
        data_dir=str(base_dir / "data"),
        api_key=None,
    )
    rag.setup_components()

    batch_records = []

    for i, q in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(QUESTIONS)}] Question: {q}")
        print("=" * 80)

        result = rag.rag_system.ask(q, top_k=5, include_context=True, context_window=1)

        docs = result.get("retrieved_documents", [])
        enhanced_docs = result.get("enhanced_documents", [])
        answer = result.get("expert_analysis", "")

        print(f"\n📚 Retrieved {len(docs)} relevant documents")
        print(f"\n💡 Answer:\n{answer}")

        if docs:
            print(f"\n📖 Related papers:")
            for j, doc in enumerate(docs[:3], 1):
                meta = doc.get("metadata", {})
                display_name = meta.get("original_title") or meta.get("filename", "N/A")
                print(f"  {j}. {display_name} (similarity: {doc.get('score', 0):.3f})")

        if enhanced_docs:
            print("\n" + "-" * 80)
            print("📌 Retrieved context (chunks and neighbors)")
            print("-" * 80)

            enhanced_sorted = sorted(
                enhanced_docs,
                key=lambda d: (
                    d.get("metadata", {}).get("filename", ""),
                    d.get("metadata", {}).get("chunk_id", 0),
                ),
            )

            current_file = None
            for doc in enhanced_sorted:
                meta = doc.get("metadata", {}) or {}
                filename = meta.get("filename", "unknown")
                chunk_id = meta.get("chunk_id", "-")
                doc_type = meta.get("type", "retrieved")
                char_start = meta.get("char_start")
                char_end = meta.get("char_end")

                if filename != current_file:
                    current_file = filename
                    display_name = meta.get("original_title") or filename
                    print("\n" + "=" * 80)
                    print(f"📄 File: {display_name}")
                    print("=" * 80)

                if doc_type == "context":
                    head = f"📍 Context chunk_id={chunk_id}"
                else:
                    head = f"🔍 Retrieved chunk_id={chunk_id} (score={doc.get('score', 0.0):.3f})"

                print("\n" + head)
                if char_start is not None and char_end is not None:
                    print(f"   Position: char[{char_start} - {char_end}]")

                text = (doc.get("text") or "").strip()
                if not text:
                    text = doc.get("excerpt", "").strip()
                print("-" * 40)
                print(text)
                print("-" * 40)

        batch_records.append(
            {
                "index": i,
                "question": q,
                "answer": answer,
                "retrieved_documents": docs,
                "scientific_context": result.get("scientific_context", ""),
                "llm_prompt": result.get("llm_prompt", ""),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = runs_dir / f"batch_qa_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(batch_records, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 80)
    print("All questions completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
