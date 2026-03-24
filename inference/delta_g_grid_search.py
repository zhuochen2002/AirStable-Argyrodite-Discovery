"""
Grid search over (alpha, beta, gamma) for hydrolysis-style reactions of doped LPSC.

Solid G uses predicted formation energy per formula unit; gas mu from standard state + ln(p).
See project docs for stoichiometry details.
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.formula_parser import parse_formula  # type: ignore


DEBUG_MISSING_MAX_PRINT = 50
DEBUG_MISSING_COUNT = 0

METALS_MAXIMIZE_DG_PER_H2O = {
    # e.g. "Mg", "Al",
}


@dataclass
class GasChemPotentials:
    """Gas chemical potentials (eV per molecule)."""

    mu0_H2O: -228.500/96.485332123
    mu0_H2S: -33.408/96.485332123
    T: float = 300.0
    p_H2O: float = 0.001
    p_H2S: float = 1e-6

    def mu_H2O(self) -> float:
        k_B_eV_per_K = 8.617333e-5
        return self.mu0_H2O + k_B_eV_per_K * self.T * math.log(self.p_H2O / 1.0)

    def mu_H2S(self) -> float:
        k_B_eV_per_K = 8.617333e-5
        return self.mu0_H2S + k_B_eV_per_K * self.T * math.log(self.p_H2S / 1.0)


def count_atoms_in_formula(formula: str) -> int:
    """Atom count for eV/atom -> eV per formula unit."""
    elements, counts = parse_formula(formula)
    return int(sum(counts))


def build_mcl_formula(metal: str, n: int) -> str:
    """MCl_n formula string."""
    return f"{metal}Cl{'' if n == 1 else n}"


def build_mf_formula(metal: str, n: int) -> str:
    """MF_n formula string."""
    return f"{metal}F{'' if n == 1 else n}"


def build_m_hydroxide_expanded_formula(metal: str, n: int) -> str:
    """Expanded M(OH)_n as M O_n H_n."""
    return f"{metal}O{'' if n == 1 else n}H{'' if n == 1 else n}"


def build_mo_n_over_2_formula_and_factor(metal: str, n: int) -> Tuple[str, float]:
    """MO_{n/2} stoichiometry: even n -> MO_{n/2}, odd n -> M2O_n with factor 0.5 on G."""
    if n <= 0:
        raise ValueError(f"Invalid oxidation state n={n} for metal {metal}")

    if n % 2 == 0:
        formula = f"{metal}O{n // 2 if n // 2 != 1 else ''}"
        factor = 1.0
    else:
        formula = f"{metal}2O{n if n != 1 else ''}"
        factor = 0.5

    return formula, factor


@dataclass
class ReactionSearchConfig:
    """Grid step sizes for alpha, beta, gamma."""

    alpha_step: float = 0.01
    beta_step: float = 0.01
    gamma_step: float = 0.01
    nonneg_tolerance: float = 1e-8


@dataclass
class ReactionResult:
    """Best feasible reaction at one grid point."""

    alpha: float
    beta: float
    gamma: float
    delta_G: float
    delta_G_norm: Optional[float]
    coeffs: Dict[str, float]


def _frange(start: float, stop: float, step: float) -> List[float]:
    """Inclusive float range with small epsilon."""
    if step <= 0:
        raise ValueError("step must be positive")
    values: List[float] = []
    x = start
    eps = step * 0.5
    while x <= stop + eps:
        values.append(round(x, 10))
        x += step
    return values


def compute_delta_g_for_grid_point(
    *,
    alpha: float,
    beta: float,
    gamma: float,
    x_doping: float,
    n_valence: int,
    metal: str,
    G_solid_per_fu: Dict[str, float],
    gas_mu: GasChemPotentials,
    nonneg_tol: float = 1e-8,
    reactant_formula: str,
) -> Optional[ReactionResult]:
    """Single (alpha,beta,gamma) evaluation; None if infeasible or missing G."""
    n = float(n_valence)
    x = float(x_doping)

    x1 = 2.0 - alpha - beta + (8.0 - n) * x        # ν(LiOH)
    x2 = -x - alpha / n                            # ν(MCl_n)
    x3 = x - beta / n                              # ν(MF_n)
    x4 = x + (alpha + beta) / n - gamma           # ν(MO_{n/2})
    omega = (12.0 - alpha - beta + (8.0 - n) * x + n * gamma) / 2.0  # ν(H2O)

    coeffs = {
        "LiCl": 1.0 + alpha,
        "LiF": beta,
        "Li3PO4": 1.0 - x,
        "H2S": 5.0,
        "LiOH": x1,
        "MCl_n": x2,
        "MF_n": x3,
        "M(OH)_n": gamma,
        "MO_n_over_2": x4,
        "H2O": omega,
    }

    for name, v in coeffs.items():
        if v < -nonneg_tol:
            return None

    for k, v in list(coeffs.items()):
        if abs(v) < nonneg_tol:
            coeffs[k] = 0.0

    global DEBUG_MISSING_COUNT

    if reactant_formula not in G_solid_per_fu:
        if DEBUG_MISSING_COUNT < DEBUG_MISSING_MAX_PRINT:
            print(
                "[DEBUG] Missing G for reactant",
                f"formula='{reactant_formula}', metal={metal}, n={n_valence}, x={x_doping},",
                f"alpha={alpha}, beta={beta}, gamma={gamma}",
            )
            DEBUG_MISSING_COUNT += 1
        return None

    G_react_raw = G_solid_per_fu[reactant_formula]
    try:
        elems_r, counts_r = parse_formula(reactant_formula)
        elem_count_map = {e: c for e, c in zip(elems_r, counts_r)}
        s_count = float(elem_count_map.get("S", 0.0))
    except Exception:
        s_count = 0.0

    if s_count > 0.0:
        scale_react = 5.0 / s_count
    else:
        scale_react = 1.0

    G_react = G_react_raw * scale_react

    needed_formulas: Dict[str, float] = {}
    needed_formulas["LiCl"] = coeffs["LiCl"]
    needed_formulas["LiF"] = coeffs["LiF"]
    needed_formulas["Li3PO4"] = coeffs["Li3PO4"]
    needed_formulas["LiOH"] = coeffs["LiOH"]

    formula_MCln = build_mcl_formula(metal, int(n_valence))
    formula_MFn = build_mf_formula(metal, int(n_valence))
    formula_MOHn = build_m_hydroxide_expanded_formula(metal, int(n_valence))
    formula_MOn2, mo_energy_factor = build_mo_n_over_2_formula_and_factor(metal, int(n_valence))

    needed_formulas[formula_MCln] = coeffs["MCl_n"]
    needed_formulas[formula_MFn] = coeffs["MF_n"]
    needed_formulas[formula_MOHn] = coeffs["M(OH)_n"]
    needed_formulas[formula_MOn2] = coeffs["MO_n_over_2"]

    for formula, nu in needed_formulas.items():
        if nu > 0.0 and formula not in G_solid_per_fu:
            if DEBUG_MISSING_COUNT < DEBUG_MISSING_MAX_PRINT:
                print(
                    "[DEBUG] Missing G for product",
                    f"formula='{formula}', metal={metal}, n={n_valence}, x={x_doping},",
                    f"alpha={alpha}, beta={beta}, gamma={gamma}, nu={nu}",
                )
                DEBUG_MISSING_COUNT += 1
            return None

    delta_G_solid = 0.0
    delta_G_solid -= G_react

    for formula, nu in needed_formulas.items():
        if nu <= 0.0:
            continue
        if formula == formula_MOn2:
            G_fu = G_solid_per_fu[formula] * mo_energy_factor
        else:
            G_fu = G_solid_per_fu[formula]
        delta_G_solid += nu * G_fu

    mu_H2O = gas_mu.mu_H2O()
    mu_H2S = gas_mu.mu_H2S()
    nu_H2O = coeffs["H2O"]
    nu_H2S = coeffs["H2S"]

    delta_G_gas = nu_H2S * mu_H2S - nu_H2O * mu_H2O

    delta_G_total = delta_G_solid + delta_G_gas

    if nu_H2O > nonneg_tol:
        delta_G_norm = delta_G_total / nu_H2O
    else:
        delta_G_norm = None

    coeffs_expanded: Dict[str, float] = {
        reactant_formula: -1.0,
        "H2O": -nu_H2O,
        "H2S": nu_H2S,
        "LiCl": coeffs["LiCl"],
        "LiF": coeffs["LiF"],
        "Li3PO4": coeffs["Li3PO4"],
        "LiOH": coeffs["LiOH"],
        formula_MCln: coeffs["MCl_n"],
        formula_MFn: coeffs["MF_n"],
        formula_MOHn: coeffs["M(OH)_n"],
        f"MO_{n_valence}/2 ({formula_MOn2})": coeffs["MO_n_over_2"],
    }

    return ReactionResult(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta_G=delta_G_total,
        delta_G_norm=delta_G_norm,
        coeffs=coeffs_expanded,
    )


def search_best_reaction_for_doped_lpsc(
    *,
    metal: str,
    n_valence: int,
    x_doping: float,
    reactant_formula: str,
    G_solid_per_fu: Dict[str, float],
    gas_mu: GasChemPotentials,
    config: ReactionSearchConfig,
) -> Optional[ReactionResult]:
    """Grid search (alpha,beta,gamma); minimize delta_G_norm (or maximize if metal in METALS_MAXIMIZE_DG_PER_H2O)."""
    n = float(n_valence)
    x = float(x_doping)

    alpha_min = -1.0
    alpha_max = -n * x
    beta_min = 0.0
    beta_max = n * x
    gamma_min = 0.0
    gamma_max = x

    if alpha_max < alpha_min:
        alpha_max, alpha_min = alpha_min, alpha_max

    alphas = _frange(alpha_min, alpha_max, config.alpha_step)
    betas = _frange(beta_min, beta_max, config.beta_step)
    gammas = _frange(gamma_min, gamma_max, config.gamma_step)

    best_result: Optional[ReactionResult] = None

    maximize = metal in METALS_MAXIMIZE_DG_PER_H2O

    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                res = compute_delta_g_for_grid_point(
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    x_doping=x,
                    n_valence=n_valence,
                    metal=metal,
                    G_solid_per_fu=G_solid_per_fu,
                    gas_mu=gas_mu,
                    nonneg_tol=config.nonneg_tolerance,
                    reactant_formula=reactant_formula,
                )
                if res is None or res.delta_G_norm is None:
                    continue

                if best_result is None:
                    best_result = res
                else:
                    if not maximize:
                        if res.delta_G_norm < best_result.delta_G_norm:  # type: ignore[arg-type]
                            best_result = res
                        elif math.isclose(
                            res.delta_G_norm, best_result.delta_G_norm  # type: ignore[arg-type]
                        ) and res.delta_G < best_result.delta_G:
                            best_result = res
                    else:
                        if res.delta_G_norm > best_result.delta_G_norm:  # type: ignore[arg-type]
                            best_result = res
                        elif math.isclose(
                            res.delta_G_norm, best_result.delta_G_norm  # type: ignore[arg-type]
                        ) and res.delta_G > best_result.delta_G:
                            best_result = res

    return best_result


def build_G_solid_from_predictions(
    prediction_csv_paths: List[str],
    energy_column: str = "pred_formation_energy",
) -> Dict[str, float]:
    """Build {formula: G per formula unit} from CSVs; energy column is eV/atom -> multiply by atom count."""
    import pandas as pd

    G_solid: Dict[str, float] = {}

    for path in prediction_csv_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "formula" not in df.columns or energy_column not in df.columns:
            continue

        for _, row in df.iterrows():
            formula = str(row["formula"])
            if formula in G_solid:
                continue
            try:
                e_form_per_atom = float(row[energy_column])
            except Exception:
                continue
            try:
                n_atoms = count_atoms_in_formula(formula)
            except Exception:
                continue
            G_solid[formula] = e_form_per_atom * n_atoms

    return G_solid


if __name__ == "__main__":
    """Batch over li6ps5cl_mfn_doping_predictions.csv -> li6ps5cl_mfn_deltaG_results.csv."""
    import pandas as pd

    prediction_files = [
        os.path.join(_project_root, "inference", "li6ps5cl_mfn_doping_predictions.csv"),
        os.path.join(_project_root, "inference", "metal_binaries_predictions.csv"),
    ]
    G_solid = build_G_solid_from_predictions(prediction_files)

    gas_mu = GasChemPotentials(
        mu0_H2O=-228.500/96.485332123,
        mu0_H2S=-33.408/96.485332123,
        T=300.0,
        p_H2O=1e-3,
        p_H2S=1e-6,
    )

    doping_path = os.path.join(_project_root, "inference", "li6ps5cl_mfn_doping_predictions.csv")
    if not os.path.exists(doping_path):
        print(f"Doping prediction file not found: {doping_path}")
        sys.exit(0)

    df_doping = pd.read_csv(doping_path)
    required_cols = {"metal", "oxidation_state", "x", "formula"}
    if not required_cols.issubset(df_doping.columns):
        print(f"Doping CSV missing required columns: {required_cols}")
        sys.exit(0)

    search_config = ReactionSearchConfig(
        alpha_step=0.01,
        beta_step=0.01,
        gamma_step=0.01,
        nonneg_tolerance=1e-8,
    )

    results_rows: List[Dict[str, object]] = []

    print(f"Total doped samples: {len(df_doping)}")

    for idx, row in df_doping.iterrows():
        metal = str(row["metal"])
        n_valence = int(row["oxidation_state"])
        x_doping = float(row["x"])
        reactant_formula = str(row["formula"])

        best = search_best_reaction_for_doped_lpsc(
            metal=metal,
            n_valence=n_valence,
            x_doping=x_doping,
            reactant_formula=reactant_formula,
            G_solid_per_fu=G_solid,
            gas_mu=gas_mu,
            config=search_config,
        )

        if best is None:
            results_rows.append(
                {
                    "metal": metal,
                    "oxidation_state": n_valence,
                    "x": x_doping,
                    "formula": reactant_formula,
                    "delta_G_min": np.nan,
                    "delta_G_norm_min": np.nan,
                    "alpha_opt": np.nan,
                    "beta_opt": np.nan,
                    "gamma_opt": np.nan,
                    "nu_LPSC_doped": -1.0,
                    "nu_H2O": np.nan,
                    "nu_LiCl": np.nan,
                    "nu_LiF": np.nan,
                    "nu_Li3PO4": np.nan,
                    "nu_H2S": np.nan,
                    "nu_LiOH": np.nan,
                    "nu_MCl_n": np.nan,
                    "nu_MF_n": np.nan,
                    "nu_MOH_n": np.nan,
                    "nu_MO_n_over_2": np.nan,
                    "mu_LPSC_doped": np.nan,
                    "mu_H2O": np.nan,
                    "mu_LiCl": np.nan,
                    "mu_LiF": np.nan,
                    "mu_Li3PO4": np.nan,
                    "mu_H2S": np.nan,
                    "mu_LiOH": np.nan,
                    "mu_MCl_n": np.nan,
                    "mu_MF_n": np.nan,
                    "mu_MOH_n": np.nan,
                    "mu_MO_n_over_2": np.nan,
                    "contrib_LPSC_doped": np.nan,
                    "contrib_H2O": np.nan,
                    "contrib_LiCl": np.nan,
                    "contrib_LiF": np.nan,
                    "contrib_Li3PO4": np.nan,
                    "contrib_H2S": np.nan,
                    "contrib_LiOH": np.nan,
                    "contrib_MCl_n": np.nan,
                    "contrib_MF_n": np.nan,
                    "contrib_MOH_n": np.nan,
                    "contrib_MO_n_over_2": np.nan,
                }
            )
        else:
            alpha_opt = best.alpha
            beta_opt = best.beta
            gamma_opt = best.gamma
            n = float(n_valence)
            x = float(x_doping)

            x1 = 2.0 - alpha_opt - beta_opt + (8.0 - n) * x        # ν(LiOH)
            x2 = -x - alpha_opt / n                                # ν(MCl_n)
            x3 = x - beta_opt / n                                  # ν(MF_n)
            x4 = x + (alpha_opt + beta_opt) / n - gamma_opt        # ν(MO_{n/2})
            omega = (12.0 - alpha_opt - beta_opt + (8.0 - n) * x + n * gamma_opt) / 2.0  # ν(H2O)

            nu_LPSC = -1.0
            nu_H2O = -omega
            nu_LiCl = 1.0 + alpha_opt
            nu_LiF = beta_opt
            nu_Li3PO4 = 1.0 - x
            nu_H2S = 5.0
            nu_LiOH = x1
            nu_MCl_n = x2
            nu_MF_n = x3
            nu_MOH_n = gamma_opt
            nu_MO_n_over_2 = x4

            G_react_raw = G_solid.get(reactant_formula, np.nan)
            try:
                elems_r, counts_r = parse_formula(reactant_formula)
                elem_count_map = {e: c for e, c in zip(elems_r, counts_r)}
                s_count = float(elem_count_map.get("S", 0.0))
            except Exception:
                s_count = 0.0
            if s_count > 0.0 and not np.isnan(G_react_raw):
                mu_LPSC = G_react_raw * (5.0 / s_count)
            else:
                mu_LPSC = G_react_raw

            mu_LiCl = G_solid.get("LiCl", np.nan)
            mu_LiF = G_solid.get("LiF", np.nan)
            mu_Li3PO4 = G_solid.get("Li3PO4", np.nan)
            mu_LiOH = G_solid.get("LiOH", np.nan)

            formula_MCln = build_mcl_formula(metal, n_valence)
            formula_MFn = build_mf_formula(metal, n_valence)
            formula_MOHn = build_m_hydroxide_expanded_formula(metal, n_valence)
            formula_MOn2, mo_energy_factor = build_mo_n_over_2_formula_and_factor(metal, n_valence)

            mu_MCl_n = G_solid.get(formula_MCln, np.nan)
            mu_MF_n = G_solid.get(formula_MFn, np.nan)
            mu_MOH_n = G_solid.get(formula_MOHn, np.nan)
            mu_MO_n_over_2 = np.nan
            if formula_MOn2 in G_solid:
                mu_MO_n_over_2 = G_solid[formula_MOn2] * mo_energy_factor

            mu_H2O = gas_mu.mu_H2O()
            mu_H2S = gas_mu.mu_H2S()

            contrib_LPSC = nu_LPSC * mu_LPSC if not np.isnan(mu_LPSC) else np.nan
            contrib_H2O = nu_H2O * mu_H2O
            contrib_LiCl = nu_LiCl * mu_LiCl if not np.isnan(mu_LiCl) else np.nan
            contrib_LiF = nu_LiF * mu_LiF if not np.isnan(mu_LiF) else np.nan
            contrib_Li3PO4 = nu_Li3PO4 * mu_Li3PO4 if not np.isnan(mu_Li3PO4) else np.nan
            contrib_H2S = nu_H2S * mu_H2S
            contrib_LiOH = nu_LiOH * mu_LiOH if not np.isnan(mu_LiOH) else np.nan
            contrib_MCl_n = nu_MCl_n * mu_MCl_n if not np.isnan(mu_MCl_n) else np.nan
            contrib_MF_n = nu_MF_n * mu_MF_n if not np.isnan(mu_MF_n) else np.nan
            contrib_MOH_n = nu_MOH_n * mu_MOH_n if not np.isnan(mu_MOH_n) else np.nan
            contrib_MO_n_over_2 = (
                nu_MO_n_over_2 * mu_MO_n_over_2 if not np.isnan(mu_MO_n_over_2) else np.nan
            )

            results_rows.append(
                {
                    "metal": metal,
                    "oxidation_state": n_valence,
                    "x": x_doping,
                    "formula": reactant_formula,
                    "delta_G_min": best.delta_G,
                    "delta_G_norm_min": best.delta_G_norm if best.delta_G_norm is not None else np.nan,
                    "alpha_opt": alpha_opt,
                    "beta_opt": beta_opt,
                    "gamma_opt": gamma_opt,
                    "nu_LPSC_doped": nu_LPSC,
                    "nu_H2O": nu_H2O,
                    "nu_LiCl": nu_LiCl,
                    "nu_LiF": nu_LiF,
                    "nu_Li3PO4": nu_Li3PO4,
                    "nu_H2S": nu_H2S,
                    "nu_LiOH": nu_LiOH,
                    "nu_MCl_n": nu_MCl_n,
                    "nu_MF_n": nu_MF_n,
                    "nu_MOH_n": nu_MOH_n,
                    "nu_MO_n_over_2": nu_MO_n_over_2,
                    "mu_LPSC_doped": mu_LPSC,
                    "mu_H2O": mu_H2O,
                    "mu_LiCl": mu_LiCl,
                    "mu_LiF": mu_LiF,
                    "mu_Li3PO4": mu_Li3PO4,
                    "mu_H2S": mu_H2S,
                    "mu_LiOH": mu_LiOH,
                    "mu_MCl_n": mu_MCl_n,
                    "mu_MF_n": mu_MF_n,
                    "mu_MOH_n": mu_MOH_n,
                    "mu_MO_n_over_2": mu_MO_n_over_2,
                    "contrib_LPSC_doped": contrib_LPSC,
                    "contrib_H2O": contrib_H2O,
                    "contrib_LiCl": contrib_LiCl,
                    "contrib_LiF": contrib_LiF,
                    "contrib_Li3PO4": contrib_Li3PO4,
                    "contrib_H2S": contrib_H2S,
                    "contrib_LiOH": contrib_LiOH,
                    "contrib_MCl_n": contrib_MCl_n,
                    "contrib_MF_n": contrib_MF_n,
                    "contrib_MOH_n": contrib_MOH_n,
                    "contrib_MO_n_over_2": contrib_MO_n_over_2,
                }
            )

        if (idx + 1) % 100 == 0 or idx == len(df_doping) - 1:
            print(f"Processed {idx + 1}/{len(df_doping)} samples")

    df_results = pd.DataFrame(results_rows)
    out_path = os.path.join(_project_root, "inference", "li6ps5cl_mfn_deltaG_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved ΔG results to: {out_path}")



