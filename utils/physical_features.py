# -*- coding: utf-8 -*-
"""
Build 11-D physical feature vectors for elements 1–118 via mendeleev and save
to data/met2vec.json in the same dict format as the rest of the project:

    {"H": [11 floats], "He": [...], ...}
"""

import json
import os
from collections import defaultdict

import numpy as np
from mendeleev import element


# Same 1–118 order as utils/mat2vec.py (if present elsewhere)
PERIODIC_TABLE = [
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
    'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]


def generate_physical_properties_for_met2vec(
    save_path: str = "data/met2vec.json",
    log_dir: str = "logs"
):
    """
    Generate 11-D physical features for elements 1–118 and write JSON.

    Feature order:
        0: atomic_number
        1: atomic_weight
        2: Pauling electronegativity (en_pauling)
        3: electron_affinity
        4: atomic_radius
        5: van der Waals radius
        6: group_id
        7: period
        8: first ionization energy (ionenergies[1])
        9: block code (s=0, p=1, d=2, f=3)
       10: nvalence()

    Args:
        save_path: Output JSON path (default data/met2vec.json).
        log_dir: Directory for error logs.

    Returns:
        dict mapping element symbol to feature list.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "physical_features_errors.log")
    error_log = defaultdict(list)

    element_properties = {}
    block_encoding = {"s": 0, "p": 1, "d": 2, "f": 3}

    print("=" * 60)
    print("Building element physical features (data/met2vec.json)")
    print("=" * 60)
    print(f"Output: {save_path}")
    print(f"Log dir: {log_dir}")
    print()

    print(f"Generating 11-D features for {len(PERIODIC_TABLE)} elements...")

    for symbol in PERIODIC_TABLE:
        try:
            elem = element(symbol)

            props = [
                elem.atomic_number,
                elem.atomic_weight,
                elem.en_pauling,
                elem.electron_affinity,
                elem.atomic_radius,
                elem.vdw_radius,
                elem.group_id,
                elem.period,
                elem.ionenergies.get(1, np.nan),
                block_encoding.get(elem.block, -1),
                elem.nvalence()
            ]

            props = [float(p) if p is not None else np.nan for p in props]
            element_properties[symbol] = props
            print(f"  OK {symbol}")

        except Exception as e:
            print(f"  WARN failed {symbol}: {e}")
            error_log["failed_elements"].append({"symbol": symbol, "error": str(e)})
            element_properties[symbol] = [np.nan] * 11

    print("\nImputing NaNs with column means...")
    props_array = np.array(list(element_properties.values()), dtype=np.float64)
    mean = np.nanmean(props_array, axis=0)

    for symbol in element_properties:
        props = np.array(element_properties[symbol], dtype=np.float64)
        props = np.where(np.isnan(props), mean, props)
        element_properties[symbol] = props.astype(float).tolist()

    sorted_elements = sorted(element_properties.keys())
    sorted_properties = {el: element_properties[el] for el in sorted_elements}

    print(f"\nSaving to {save_path}")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sorted_properties, f, indent=2)

    if error_log["failed_elements"]:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(error_log, f, indent=2, ensure_ascii=False)
        print(f"WARN {len(error_log['failed_elements'])} elements failed; see {log_file}")

    print("\nDone.")
    print(f"   Elements: {len(sorted_properties)}")
    print(f"   Dim: 11")
    print("=" * 60)

    return sorted_properties


def main():
    save_path = os.path.join("data", "met2vec.json")
    log_dir = "logs"
    generate_physical_properties_for_met2vec(save_path=save_path, log_dir=log_dir)


if __name__ == "__main__":
    main()
