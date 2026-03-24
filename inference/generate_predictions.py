"""
Batch predictions for (1) simple metal binaries and (2) Li6PS5Cl-based MFn solid solutions.

Outputs:
  inference/metal_binaries_predictions.csv
  inference/li6ps5cl_mfn_doping_predictions.csv

Run from repo root: python -m inference.generate_predictions
"""

import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from inference.predictor import FormationEnergyPredictor
from data.dataset import get_element_vocabulary_from_mat2vec


# Metal set and oxidation states (see inorganic chemistry references)
_ALL_METALS = {
    "Na", "K", "Rb", "Cs", "Fr",
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Al", "Ga", "In", "Tl", "Nh",
    "Ge", "Sn", "Pb", "Fl",
    "Sb", "Bi", "Mc",
    "Po", "Lv",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
}

METAL_OXIDATION_STATES: Dict[str, List[int]] = {
    "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1], "Fr": [1],
    "Be": [2], "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2], "Ra": [2],
    "Sc": [3], "Y": [3], "La": [3],
    "Ti": [2, 3, 4], "Zr": [2, 3, 4], "Hf": [2, 3, 4], "Rf": [4],
    "V": [2, 3, 4, 5], "Nb": [2, 3, 4, 5], "Ta": [2, 3, 4, 5], "Db": [5],
    "Cr": [2, 3, 4, 5, 6], "Mo": [2, 3, 4, 5, 6], "W": [2, 3, 4, 5, 6], "Sg": [6],
    "Mn": [2, 3, 4, 5, 6, 7], "Tc": [4, 5, 6, 7], "Re": [2, 4, 5, 6, 7], "Bh": [7],
    "Fe": [2, 3, 4, 5, 6], "Ru": [2, 3, 4, 5, 6, 7, 8], "Os": [2, 3, 4, 5, 6, 7, 8], "Hs": [6, 8],
    "Co": [2, 3, 4], "Rh": [2, 3, 4, 5, 6], "Ir": [2, 3, 4, 5, 6], "Mt": [3],
    "Ni": [2, 3, 4], "Pd": [2, 4], "Pt": [2, 4], "Ds": [4],
    "Cu": [1, 2], "Ag": [1], "Au": [1, 3], "Rg": [3],
    "Zn": [2], "Cd": [2], "Hg": [1, 2], "Cn": [2],
    "Al": [3], "Ga": [1, 3], "In": [1, 3], "Tl": [1, 3], "Nh": [3],
    "Ge": [2, 4], "Sn": [2, 4], "Pb": [2, 4], "Fl": [2],
    "Sb": [3, 5], "Bi": [3, 5], "Mc": [3],
    "Po": [2, 4], "Lv": [2],
    "Ce": [3, 4], "Pr": [3, 4], "Nd": [2, 3], "Pm": [3],
    "Sm": [2, 3], "Eu": [2, 3], "Gd": [3], "Tb": [3, 4],
    "Dy": [2, 3], "Ho": [3], "Er": [3], "Tm": [2, 3],
    "Yb": [2, 3], "Lu": [3],
    "Ac": [3], "Th": [4], "Pa": [3, 4, 5], "U": [3, 4, 5, 6],
    "Np": [3, 4, 5, 6, 7], "Pu": [3, 4, 5, 6, 7], "Am": [2, 3, 4, 5, 6],
    "Cm": [3, 4], "Bk": [3, 4], "Cf": [2, 3, 4], "Es": [2, 3],
    "Fm": [2, 3], "Md": [2, 3], "No": [2, 3], "Lr": [3],
}

_DEFAULT_OXIDATION_BY_GROUP: Dict[int, List[int]] = {
    1: [1], 2: [2], 3: [3], 4: [2, 3, 4], 5: [2, 3, 4, 5], 6: [2, 3, 4, 5, 6],
    7: [2, 3, 4, 5, 6, 7], 8: [2, 3, 4, 5, 6, 7, 8], 9: [2, 3, 4],
    10: [2, 3, 4], 11: [1, 2], 12: [2], 13: [3], 14: [2, 4], 15: [3, 5], 16: [2, 4],
}

_ELEMENT_GROUP: Dict[str, int] = {
    "H": 1, "He": 18, "Li": 1, "Be": 2, "B": 13, "C": 14, "N": 15, "O": 16, "F": 17,
    "Ne": 18, "Na": 1, "Mg": 2, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12, "Ga": 13, "Ge": 14, "As": 15, "Se": 16, "Br": 17, "Kr": 18,
    "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10,
    "Ag": 11, "Cd": 12, "In": 13, "Sn": 14, "Sb": 15, "Te": 16, "I": 17, "Xe": 18,
    "Cs": 1, "Ba": 2, "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 3, "Eu": 3,
    "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10, "Au": 11, "Hg": 12,
    "Tl": 13, "Pb": 14, "Bi": 15, "Po": 16, "At": 17, "Rn": 18,
    "Fr": 1, "Ra": 2, "Ac": 3, "Th": 3, "Pa": 3, "U": 3, "Np": 3, "Pu": 3, "Am": 3,
    "Cm": 3, "Bk": 3, "Cf": 3, "Es": 3, "Fm": 3, "Md": 3, "No": 3, "Lr": 3,
    "Rf": 4, "Db": 5, "Sg": 6, "Bh": 7, "Hs": 8, "Mt": 9, "Ds": 10, "Rg": 11, "Cn": 12,
    "Nh": 13, "Fl": 14, "Mc": 15, "Lv": 16,
}


def _get_oxidation_states_for_metal(metal: str) -> List[int]:
    """Oxidation states: explicit table first, else group fallback."""
    if metal in METAL_OXIDATION_STATES:
        return METAL_OXIDATION_STATES[metal]
    group = _ELEMENT_GROUP.get(metal)
    if group is not None and group in _DEFAULT_OXIDATION_BY_GROUP:
        return _DEFAULT_OXIDATION_BY_GROUP[group]
    return [2, 3]


def get_metal_elements_in_vocab() -> List[Tuple[str, List[int]]]:
    """Metals present in mat2vec vocabulary with their oxidation state lists."""
    element_to_idx = get_element_vocabulary_from_mat2vec()
    vocab_elements = set(element_to_idx.keys())
    available_metals = []
    for metal in _ALL_METALS:
        if metal in vocab_elements:
            states = _get_oxidation_states_for_metal(metal)
            available_metals.append((metal, states))
    
    return sorted(available_metals, key=lambda x: x[0])


def build_oxide_formula(metal: str, n: int) -> str:
    """Neutral oxide from valence n (even n -> MO_{n/2}, odd -> M2O_n)."""
    if n <= 0:
        raise ValueError(f"Invalid oxidation state n={n} for metal {metal}")

    if n % 2 == 0:
        m_M = 1
        m_O = n // 2
    else:
        m_M = 2
        m_O = n

    return f"{metal}{'' if m_M == 1 else m_M}O{'' if m_O == 1 else m_O}"


def build_halide_formula(metal: str, n: int, halogen: str) -> str:
    """Neutral halide MX_n."""
    if n <= 0:
        raise ValueError(f"Invalid oxidation state n={n} for metal {metal}")
    return f"{metal}{halogen}{'' if n == 1 else n}"


def build_hydroxide_formula(metal: str, n: int) -> str:
    """Expanded M(OH)_n as M O_n H_n (parser has no parentheses)."""
    if n <= 0:
        raise ValueError(f"Invalid oxidation state n={n} for metal {metal}")
    return f"{metal}O{'' if n == 1 else n}H{'' if n == 1 else n}"


def generate_metal_binary_formulas(metal_oxidation_pairs: List[Tuple[str, List[int]]]) -> List[Dict]:
    """Halides, oxides, hydroxides for each metal and valence."""
    records: List[Dict] = []
    halogens = ["Cl", "F", "Br", "I"]

    for metal, oxidation_states in metal_oxidation_pairs:
        for n in oxidation_states:
            oxide_formula = build_oxide_formula(metal, n)
            records.append(
                {
                    "formula": oxide_formula,
                    "metal": metal,
                    "anion": "O",
                    "type": "oxide",
                    "oxidation_state": n,
                }
            )

            hydroxide_formula = build_hydroxide_formula(metal, n)
            records.append(
                {
                    "formula": hydroxide_formula,
                    "metal": metal,
                    "anion": "OH",
                    "type": "hydroxide",
                    "oxidation_state": n,
                }
            )

            for X in halogens:
                halide_formula = build_halide_formula(metal, n, X)
                records.append(
                    {
                        "formula": halide_formula,
                        "metal": metal,
                        "anion": X,
                        "type": "halide",
                        "oxidation_state": n,
                    }
                )

    return records


def generate_li6ps5cl_mfn_formulas(
    metal_oxidation_pairs: List[Tuple[str, List[int]]],
    x_min: float = 0.0,
    x_max: float = 0.1,
    x_step: float = 0.01,
) -> List[Dict]:
    """
    Li6PS5Cl-type MFn doping: integer-scaled formulas with x = k/100.
    See code for Li/P/S/Cl/F counts; non-negative counts only; x=0 -> Li6PS5Cl.
    """
    records: List[Dict] = []
    k_values: List[int] = []
    k_min = int(round(x_min * 100))
    k_max = int(round(x_max * 100))
    k_step = int(round(x_step * 100))
    for k in range(k_min, k_max + 1, max(1, k_step)):
        if k < 0:
            continue
        k_values.append(k)

    for metal, oxidation_states in metal_oxidation_pairs:
        for n in oxidation_states:
            k_upper_by_cl = int(math.floor(100.0 / float(n))) if n > 0 else 0

            for k in k_values:
                if k == 0:
                    formula = "Li6PS5Cl"
                    records.append(
                        {
                            "metal": metal,
                            "oxidation_state": n,
                            "x": 0.0,
                            "k": 0,
                            "formula": formula,
                            "Li_count": 6,
                            "M_count": 0,
                            "P_count": 1,
                            "S_count": 5,
                            "Cl_count": 1,
                            "F_count": 0,
                        }
                    )
                    continue

                if k > k_upper_by_cl:
                    continue

                li_count = 600 + 5 * k - n * k
                m_count = k
                p_count = 100 - k
                s_count = 500
                cl_count = 100 - n * k
                f_count = n * k

                if min(li_count, m_count, p_count, s_count, cl_count, f_count) < 0:
                    continue

                counts = [li_count, m_count, p_count, s_count, cl_count, f_count]
                nonzero_counts = [c for c in counts if c > 0]
                if nonzero_counts:
                    gcd = int(nonzero_counts[0])
                    for c in nonzero_counts[1:]:
                        gcd = math.gcd(gcd, int(c))
                    if gcd > 1:
                        li_int = li_count // gcd
                        m_int = m_count // gcd
                        p_int = p_count // gcd
                        s_int = s_count // gcd
                        cl_int = cl_count // gcd
                        f_int = f_count // gcd
                    else:
                        li_int = li_count
                        m_int = m_count
                        p_int = p_count
                        s_int = s_count
                        cl_int = cl_count
                        f_int = f_count
                else:
                    continue

                parts: List[str] = []
                if li_int > 0:
                    parts.append(f"Li{'' if li_int == 1 else li_int}")
                if m_int > 0:
                    parts.append(f"{metal}{'' if m_int == 1 else m_int}")
                if p_int > 0:
                    parts.append(f"P{'' if p_int == 1 else p_int}")
                if s_int > 0:
                    parts.append(f"S{'' if s_int == 1 else s_int}")
                if cl_int > 0:
                    parts.append(f"Cl{'' if cl_int == 1 else cl_int}")
                if f_int > 0:
                    parts.append(f"F{'' if f_int == 1 else f_int}")

                if not parts:
                    continue

                formula = "".join(parts)
                x = k / 100.0

                records.append(
                    {
                        "metal": metal,
                        "oxidation_state": n,
                        "x": x,
                        "k": k,
                        "formula": formula,
                        "Li_count": li_int,
                        "M_count": m_int,
                        "P_count": p_int,
                        "S_count": s_int,
                        "Cl_count": cl_int,
                        "F_count": f_int,
                    }
                )

    return records


def predict_for_formulas(
    predictor: FormationEnergyPredictor,
    records: List[Dict],
    formula_key: str = "formula",
    batch_size: int = 1024,
) -> pd.DataFrame:
    """Batch predict formulas; add pred, contribution string, and Li/P/S/Cl/M columns."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    formulas: List[str] = df[formula_key].astype(str).tolist()

    preds: List[float] = []
    all_contributions: List[List[Tuple[str, float, float]]] = []
    
    for start in range(0, len(formulas), batch_size):
        batch_formulas = formulas[start: start + batch_size]
        batch_preds, batch_contributions = predictor.predict_batch_with_contributions(batch_formulas)
        preds.extend(batch_preds)
        all_contributions.extend(batch_contributions)

    df["pred_formation_energy"] = np.array(preds, dtype=float)

    contribution_strings = []
    for contributions in all_contributions:
        if contributions:
            parts = [f"{elem}:{frac:.6f}:{contrib:.6f}" for elem, frac, contrib in contributions]
            contribution_strings.append("|".join(parts))
        else:
            contribution_strings.append("")
    
    df["element_contributions"] = contribution_strings

    BASE_ELEMENTS_FOR_COLUMNS = {"Li", "P", "S", "Cl"}

    for elem in sorted(BASE_ELEMENTS_FOR_COLUMNS):
        fractions_col = []
        contributions_col = []
        for contributions in all_contributions:
            found = False
            for e, frac, contrib in contributions:
                if e == elem:
                    fractions_col.append(frac)
                    contributions_col.append(contrib)
                    found = True
                    break
            if not found:
                fractions_col.append(np.nan)
                contributions_col.append(np.nan)
        
        df[f"{elem}_fraction"] = fractions_col
        df[f"{elem}_contribution"] = contributions_col

    M_fractions_col = []
    M_contributions_col = []
    
    for contributions in all_contributions:
        metal_fractions = []
        metal_contributions = []
        for elem, frac, contrib in contributions:
            if elem not in BASE_ELEMENTS_FOR_COLUMNS:
                metal_fractions.append(frac)
                metal_contributions.append(contrib)
        
        if metal_fractions:
            total_metal_fraction = sum(metal_fractions)
            if total_metal_fraction > 0:
                weighted_contrib = sum(f * c for f, c in zip(metal_fractions, metal_contributions)) / total_metal_fraction
                M_fractions_col.append(total_metal_fraction)
                M_contributions_col.append(weighted_contrib)
            else:
                M_fractions_col.append(0.0)
                M_contributions_col.append(0.0)
        else:
            M_fractions_col.append(np.nan)
            M_contributions_col.append(np.nan)
    
    df["M_fraction"] = M_fractions_col
    df["M_contribution"] = M_contributions_col
    
    return df


def main(
    model_path: str = "checkpoints/best_model_final.pt",
    output_dir: str = "inference",
) -> None:
    """Load model, generate both CSV outputs."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading formation energy predictor...")
    predictor = FormationEnergyPredictor(model_path=model_path)

    print("\nCollecting metal elements from mat2vec vocabulary...")
    metal_oxidation_pairs = get_metal_elements_in_vocab()
    print(f"  Found {len(metal_oxidation_pairs)} metal elements in vocabulary")
    
    total_states = sum(len(states) for _, states in metal_oxidation_pairs)
    print(f"  Total oxidation states: {total_states}")
    print(f"  Example: {metal_oxidation_pairs[:5]}")

    print("\nGenerating simple binary metal compounds (halides + oxides + hydroxides)...")
    binary_records = generate_metal_binary_formulas(metal_oxidation_pairs)

    baseline_formulas = [
        {"formula": "LiCl",   "metal": "Li", "anion": "Cl", "type": "baseline", "oxidation_state": 1},
        {"formula": "Li2S",   "metal": "Li", "anion": "S",  "type": "baseline", "oxidation_state": 1},
        {"formula": "Li3PO4", "metal": "Li", "anion": "O",  "type": "baseline", "oxidation_state": 1},
        {"formula": "LiF",    "metal": "Li", "anion": "F",  "type": "baseline", "oxidation_state": 1},
        {"formula": "LiOH",   "metal": "Li", "anion": "OH", "type": "baseline", "oxidation_state": 1},
    ]
    binary_records.extend(baseline_formulas)

    print(f"  Total binary compounds (including baselines): {len(binary_records)}")

    if binary_records:
        df_temp = pd.DataFrame(binary_records)
        type_counts = df_temp["type"].value_counts()
        print(f"  Breakdown by type:")
        for comp_type, count in type_counts.items():
            print(f"    {comp_type}: {count}")

    print("Predicting formation energies for binary compounds...")
    binary_df = predict_for_formulas(predictor, binary_records)

    binary_out_path = os.path.join(output_dir, "metal_binaries_predictions.csv")
    binary_df.to_csv(binary_out_path, index=False)
    print(f"  Saved binary predictions to: {binary_out_path}")

    print("\nGenerating Li6PS5Cl-based MFn doping compositions (all oxidation states)...")
    doping_records = generate_li6ps5cl_mfn_formulas(metal_oxidation_pairs)
    print(f"  Total Li6PS5Cl-based compositions: {len(doping_records)}")

    print("Predicting formation energies for Li6PS5Cl-based doped systems...")
    doping_df = predict_for_formulas(predictor, doping_records)

    doping_out_path = os.path.join(output_dir, "li6ps5cl_mfn_doping_predictions.csv")
    doping_df.to_csv(doping_out_path, index=False)
    print(f"  Saved doping predictions to: {doping_out_path}")

    print("\nAll predictions finished.")


if __name__ == "__main__":
    main()


