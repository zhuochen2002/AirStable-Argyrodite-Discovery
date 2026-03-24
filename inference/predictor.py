"""Formation energy inference: load checkpoint, encode formulas, batch predict and evaluate CSVs."""
import os
import sys
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import MODEL_CONFIG
from models.model import FormationEnergyModel
from utils.mat2vec import load_embeddings_matrix
from data.dataset import get_element_vocabulary_from_mat2vec
from data.formula_parser import parse_formula_to_unique_elements


class FormationEnergyPredictor:
    """Inference entry aligned with training (same model and vocabulary)."""

    def __init__(
        self,
        model_path: str = "checkpoints/best_model_final.pt",
        device: str = None,
        n_max: int = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        embeddings_matrix, _, _ = load_embeddings_matrix()
        mat2vec_tensor = torch.from_numpy(embeddings_matrix).float()

        self.model = FormationEnergyModel(MODEL_CONFIG, mat2vec_tensor)
        self.model.to(self.device)

        # Some code versions save checkpoints as `best_model.pt` while others
        # expect `best_model_final.pt`. Here we transparently fall back.
        resolved_path: Optional[str] = None
        candidates: List[str] = []
        if model_path:
            candidates.append(model_path)
        ckpt_dir = os.path.dirname(model_path) if model_path else "checkpoints"
        candidates.extend([
            os.path.join(ckpt_dir, "best_model.pt"),
            os.path.join(ckpt_dir, "best_model_final.pt"),
        ])
        seen = set()
        for p in candidates:
            if not p or p in seen:
                continue
            seen.add(p)
            if os.path.exists(p):
                resolved_path = p
                break
        if resolved_path is None:
            raise FileNotFoundError(
                f"Model checkpoint not found. Tried: {candidates}. "
                f"(Given model_path={model_path})"
            )

        checkpoint = torch.load(resolved_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.element_to_idx: Dict[str, int] = get_element_vocabulary_from_mat2vec()
        self.n_max: int = n_max if n_max is not None else MODEL_CONFIG["n_max"]

    def _encode_formula(
        self, formula: str
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
        """Encode formula to (element_indices, fractions, mask) like FormationEnergyDataset."""
        elements, fractions = parse_formula_to_unique_elements(formula)

        indices: List[int] = []
        valid_fractions: List[float] = []
        unknown: List[str] = []

        for element, fraction in zip(elements, fractions):
            element_norm = (
                element[0].upper() + element[1:].lower()
                if len(element) > 1
                else element.upper()
            )
            if element_norm in self.element_to_idx:
                indices.append(self.element_to_idx[element_norm])
                valid_fractions.append(float(fraction))
            else:
                unknown.append(element)

        if unknown:
            print(f"Warning: Unknown elements {unknown} in formula '{formula}' (skipped)")

        n_elements = len(indices)
        if n_elements == 0:
            raise ValueError(
                f"All elements in formula '{formula}' are unknown or invalid. "
                f"Parsed elements: {elements}, Unknown: {unknown}"
            )

        if n_elements > self.n_max:
            indices = indices[: self.n_max]
            valid_fractions = valid_fractions[: self.n_max]
            n_elements = self.n_max

        element_indices = np.zeros(self.n_max, dtype=np.int64)
        fractions_arr = np.zeros(self.n_max, dtype=np.float32)
        mask = np.zeros(self.n_max, dtype=np.float32)

        element_indices[:n_elements] = indices
        fractions_arr[:n_elements] = valid_fractions
        mask[:n_elements] = 1.0

        total = fractions_arr[:n_elements].sum()
        if total > 0:
            fractions_arr[:n_elements] /= total

        return (
            torch.LongTensor(element_indices),
            torch.FloatTensor(fractions_arr),
            torch.FloatTensor(mask),
        )

    def predict_batch(self, formulas: List[str]) -> List[float]:
        """Predict formation energy for a list of formulas."""
        if not formulas:
            return []

        encoded = [self._encode_formula(f) for f in formulas]
        element_indices = torch.stack([e[0] for e in encoded], dim=0).to(self.device)
        fractions = torch.stack([e[1] for e in encoded], dim=0).to(self.device)
        mask = torch.stack([e[2] for e in encoded], dim=0).to(self.device)

        with torch.no_grad():
            preds = self.model(element_indices, fractions, mask)

        return preds.cpu().numpy().astype(float).tolist()

    def predict_batch_with_contributions(
        self, formulas: List[str]
    ) -> Tuple[List[float], List[List[Tuple[str, float, float]]]]:
        """
        Predict formation energy and per-slot element contributions.

        Returns:
            predictions: one float per formula
            contributions: list of (element_symbol, fraction, contribution_value) per formula
        """
        if not formulas:
            return [], []

        encoded = [self._encode_formula(f) for f in formulas]
        element_indices = torch.stack([e[0] for e in encoded], dim=0).to(self.device)
        fractions = torch.stack([e[1] for e in encoded], dim=0).to(self.device)
        mask = torch.stack([e[2] for e in encoded], dim=0).to(self.device)

        with torch.no_grad():
            tokens = self.model.composition_embedding(element_indices, fractions, mask)
            encoded_t = self.model.encoder(tokens, mask)
            encoded_norm = self.model.pre_head_norm(encoded_t)
            element_contributions = self.model.element_head(encoded_norm).squeeze(-1)
            formation_energy = (fractions * element_contributions).sum(dim=1)

        preds = formation_energy.cpu().numpy().astype(float).tolist()
        contributions_tensor = element_contributions.cpu().numpy()
        fractions_tensor = fractions.cpu().numpy()
        element_indices_tensor = element_indices.cpu().numpy()
        mask_tensor = mask.cpu().numpy()

        idx_to_element = {idx: elem for elem, idx in self.element_to_idx.items()}

        all_contributions = []
        for i in range(len(formulas)):
            sample_contributions = []
            for j in range(self.n_max):
                if mask_tensor[i, j] > 0.5:
                    elem_idx = int(element_indices_tensor[i, j])
                    if elem_idx > 0 and elem_idx in idx_to_element:
                        element_symbol = idx_to_element[elem_idx]
                        fraction = float(fractions_tensor[i, j])
                        contribution = float(contributions_tensor[i, j])
                        sample_contributions.append((element_symbol, fraction, contribution))
            all_contributions.append(sample_contributions)

        return preds, all_contributions

    def get_attention_weights(self, formula: str) -> Tuple[np.ndarray, List[str]]:
        """Last layer, heads averaged: attention [n_valid, n_valid], element labels."""
        element_indices, fractions, mask = self._encode_formula(formula)
        element_indices = element_indices.unsqueeze(0).to(self.device)
        fractions = fractions.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            attn = self.model.get_attention_weights(element_indices, fractions, mask)

        attn_np = attn[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()

        valid_idx = np.where(mask_np > 0.5)[0]
        attn_valid = attn_np[np.ix_(valid_idx, valid_idx)]

        idx_to_element = {idx: elem for elem, idx in self.element_to_idx.items()}
        elements = [
            idx_to_element[int(element_indices[0, j].item())]
            for j in valid_idx
            if element_indices[0, j].item() > 0
        ]

        return attn_valid, elements

    def get_attention_weights_per_head(
        self, formula: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Last layer, per head: [n_heads, n_valid, n_valid]."""
        element_indices, fractions, mask = self._encode_formula(formula)
        element_indices = element_indices.unsqueeze(0).to(self.device)
        fractions = fractions.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            attn = self.model.get_attention_weights_per_head(
                element_indices, fractions, mask
            )

        attn_np = attn[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()

        valid_idx = np.where(mask_np > 0.5)[0]
        n_heads = attn_np.shape[0]
        attn_per_head = np.array(
            [attn_np[h][np.ix_(valid_idx, valid_idx)] for h in range(n_heads)]
        )

        idx_to_element = {idx: elem for elem, idx in self.element_to_idx.items()}
        elements = [
            idx_to_element[int(element_indices[0, j].item())]
            for j in valid_idx
        ]

        return attn_per_head, elements

    def predict_file(
        self,
        csv_path: str,
        formula_col: str = "formula",
        target_col: Optional[str] = "formation_energy",
        batch_size: int = 1024,
        output_path: Optional[str] = None,
        plot_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Batch-predict CSV; optional metrics and scatter plot when target column exists."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if formula_col not in df.columns:
            raise ValueError(f"Column '{formula_col}' not found in {csv_path}")

        formulas = df[formula_col].astype(str).tolist()
        preds: List[float] = []

        for start in range(0, len(formulas), batch_size):
            batch_formulas = formulas[start : start + batch_size]
            batch_preds = self.predict_batch(batch_formulas)
            preds.extend(batch_preds)

        df["pred_formation_energy"] = preds

        metrics: Dict[str, float] = {}
        if target_col is not None and target_col in df.columns:
            y_true = df[target_col].astype(float).values
            y_pred = np.array(preds, dtype=float)
            errors = y_pred - y_true
            abs_errors = np.abs(errors)

            mae = float(abs_errors.mean())
            mse = float((errors ** 2).mean())
            rmse = float(np.sqrt(mse))

            y_mean = float(y_true.mean())
            ss_res = float(((y_true - y_pred) ** 2).sum())
            ss_tot = float(((y_true - y_mean) ** 2).sum())
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            df["error"] = errors
            df["abs_error"] = abs_errors

            metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

            print(f"[{os.path.basename(csv_path)}]")
            print(f"  MAE : {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MSE : {mse:.6f}")
            print(f"  R2  : {r2:.6f}")

            if plot_path is not None:
                plt.figure(figsize=(5, 5))
                plt.scatter(y_true, y_pred, s=5, alpha=0.5)
                y_min = float(min(y_true.min(), y_pred.min()))
                y_max = float(max(y_true.max(), y_pred.max()))
                plt.plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=1)
                plt.xlabel("True formation energy")
                plt.ylabel("Predicted formation energy")
                plt.tight_layout()
                plt.savefig(plot_path, dpi=300)
                plt.close()

        if output_path is None:
            root, ext = os.path.splitext(csv_path)
            output_path = root + "_pred" + ext
        df.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")

        return metrics

    def evaluate_standard_splits(
        self,
        train_path: str = "data/train.csv",
        val_path: str = "data/val.csv",
        test_path: str = "data/test.csv",
        formula_col: str = "formula",
        target_col: str = "formation_energy",
        batch_size: int = 1024,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate train/val/test CSVs; writes pair plots next to each split file."""
        results: Dict[str, Dict[str, float]] = {}

        if os.path.exists(train_path):
            print("\nEvaluating on train set...")
            results["train"] = self.predict_file(
                train_path,
                formula_col=formula_col,
                target_col=target_col,
                batch_size=batch_size,
                output_path=None,
                plot_path=os.path.join(os.path.dirname(train_path), "train_pair_plot.png"),
            )
        else:
            print(f"Train file not found: {train_path}")

        if os.path.exists(val_path):
            print("\nEvaluating on val set...")
            results["val"] = self.predict_file(
                val_path,
                formula_col=formula_col,
                target_col=target_col,
                batch_size=batch_size,
                output_path=None,
                plot_path=os.path.join(os.path.dirname(val_path), "val_pair_plot.png"),
            )
        else:
            print(f"Val file not found: {val_path}")

        if os.path.exists(test_path):
            print("\nEvaluating on test set...")
            results["test"] = self.predict_file(
                test_path,
                formula_col=formula_col,
                target_col=target_col,
                batch_size=batch_size,
                output_path=None,
                plot_path=os.path.join(os.path.dirname(test_path), "test_pair_plot.png"),
            )
        else:
            print(f"Test file not found: {test_path}")

        return results


if __name__ == "__main__":
    """
    No args: evaluate data/train.csv, data/val.csv, data/test.csv and append predictions.
    With CSV paths: predict each file; metrics if formation_energy column exists.

        python -m inference.predictor
        python -m inference.predictor data/custom.csv
    """
    predictor = FormationEnergyPredictor()

    if len(sys.argv) == 1:
        predictor.evaluate_standard_splits()
    else:
        for path in sys.argv[1:]:
            print(f"\nPredicting for file: {path}")
            predictor.predict_file(path)
