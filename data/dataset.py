"""PyTorch Dataset for formation energy: load/split CSV, pad sequences, masks."""
import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.formula_parser import parse_formula_to_unique_elements
from data.data_split import load_dataset, split_dataset
from utils.mat2vec import load_embeddings_matrix


def add_processed_elements_to_dataframe(data: pd.DataFrame, element_to_idx: Dict[str, int]) -> pd.DataFrame:
    """
    Add columns: elements, fractions, element_indices (comma-separated strings).
    Indices use 1-based vocabulary; 0 is reserved for padding in the model.
    """
    elements_list = []
    fractions_list = []
    indices_list = []

    print("Processing formulas and extracting elements...")
    parse_errors = 0

    for _, row in data.iterrows():
        formula = row['formula']
        try:
            elements, fractions = parse_formula_to_unique_elements(formula)
            valid_elements = []
            valid_fractions = []
            valid_indices = []

            for element, fraction in zip(elements, fractions):
                element_normalized = (
                    element[0].upper() + element[1:].lower() if len(element) > 1 else element.upper()
                )
                if element_normalized in element_to_idx:
                    valid_elements.append(element_normalized)
                    valid_fractions.append(fraction)
                    valid_indices.append(element_to_idx[element_normalized])

            if len(valid_elements) == 0:
                parse_errors += 1
                elements_list.append("")
                fractions_list.append("")
                indices_list.append("")
            else:
                total_fraction = sum(valid_fractions)
                if total_fraction > 0:
                    valid_fractions = [f / total_fraction for f in valid_fractions]
                elements_str = ",".join(valid_elements)
                fractions_str = ",".join([f"{f:.8f}" for f in valid_fractions])
                indices_str = ",".join(str(ei) for ei in valid_indices)
                elements_list.append(elements_str)
                fractions_list.append(fractions_str)
                indices_list.append(indices_str)

        except Exception as e:
            parse_errors += 1
            elements_list.append("")
            fractions_list.append("")
            indices_list.append("")
            if parse_errors <= 5:
                print(f"  Warning: Failed to parse formula '{formula}': {e}")

    if parse_errors > 0:
        print(f"  Warning: {parse_errors} formulas had parsing issues")

    data = data.copy()
    data['elements'] = elements_list
    data['fractions'] = fractions_list
    data['element_indices'] = indices_list
    return data


def _compute_element_sets(df: pd.DataFrame) -> Dict[int, set]:
    """Map row index -> set of element symbols for coverage checks."""
    element_sets: Dict[int, set] = {}
    for idx, row in df.iterrows():
        formula = row.get('formula', None)
        if not isinstance(formula, str) or not formula:
            continue
        try:
            elements, _ = parse_formula_to_unique_elements(formula)
            element_sets[idx] = set(elements)
        except Exception as e:
            print(f"  Warning: Failed to parse formula '{formula}' when computing element sets: {e}")
    return element_sets


def ensure_element_coverage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ensure every element that appears anywhere in the global pool appears in train
    at least once by moving representative rows from val/test to train when needed.
    """
    if train_df.empty:
        print("Warning: train_df is empty when ensuring element coverage; skip coverage adjustment.")
        return train_df, val_df, test_df

    print("Ensuring element coverage in train set...")
    train_elem_sets = _compute_element_sets(train_df)
    val_elem_sets = _compute_element_sets(val_df)
    test_elem_sets = _compute_element_sets(test_df)

    all_elements = set()
    for es in list(train_elem_sets.values()) + list(val_elem_sets.values()) + list(test_elem_sets.values()):
        all_elements.update(es)

    train_elements = set()
    for es in train_elem_sets.values():
        train_elements.update(es)

    missing_elements = sorted(all_elements - train_elements)
    if not missing_elements:
        print("  All elements already appear in train set. No adjustment needed.")
        return train_df, val_df, test_df

    print(f"  Elements missing in train set: {missing_elements}")
    train_median_fe = float(train_df['formation_energy'].median())
    move_from_val = set()
    move_from_test = set()

    for elem in missing_elements:
        val_candidates = [i for i, es in val_elem_sets.items() if elem in es]
        test_candidates = [i for i, es in test_elem_sets.items() if elem in es]
        chosen_idx = None
        chosen_from = None

        def _select_best(candidates, df_local):
            if not candidates:
                return None
            best_idx = None
            best_dist = None
            for i in candidates:
                fe = float(df_local.loc[i, 'formation_energy'])
                dist = abs(fe - train_median_fe)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = i
            return best_idx

        if val_candidates:
            chosen_idx = _select_best(val_candidates, val_df)
            chosen_from = "val"
        elif test_candidates:
            chosen_idx = _select_best(test_candidates, test_df)
            chosen_from = "test"

        if chosen_idx is None:
            print(f"  Warning: Element '{elem}' does not appear in val/test; cannot ensure coverage.")
            continue

        if chosen_from == "val":
            move_from_val.add(chosen_idx)
        elif chosen_from == "test":
            move_from_test.add(chosen_idx)

    if move_from_val:
        rows = val_df.loc[list(move_from_val)]
        train_df = pd.concat([train_df, rows], axis=0)
        val_df = val_df.drop(index=list(move_from_val))
    if move_from_test:
        rows = test_df.loc[list(move_from_test)]
        train_df = pd.concat([train_df, rows], axis=0)
        test_df = test_df.drop(index=list(move_from_test))

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"  Moved {len(move_from_val)} samples from val and {len(move_from_test)} samples from test to train.")
    print(f"  New sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def prepare_datasets(csv_path: str,
                   output_dir: str = "data",
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   random_seed: int = 42,
                   split_method: str = "stratified") -> Tuple[str, str, str]:
    """Load CSV, split, add element columns, write train/val/test CSV paths."""
    print("=" * 60)
    print("Preparing datasets: loading and splitting")
    print("=" * 60)

    data = load_dataset(csv_path)
    print()

    train_data, val_data, test_data = split_dataset(
        data,
        method=split_method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    print()

    train_data, val_data, test_data = ensure_element_coverage(train_data, val_data, test_data)
    print()

    print("Loading element vocabulary for index mapping...")
    element_to_idx = get_element_vocabulary_from_mat2vec()
    print()

    print("Adding processed elements to train set...")
    train_data = add_processed_elements_to_dataframe(train_data, element_to_idx)
    print(f"  Train set: {len(train_data)} samples")

    print("Adding processed elements to validation set...")
    val_data = add_processed_elements_to_dataframe(val_data, element_to_idx)
    print(f"  Validation set: {len(val_data)} samples")

    print("Adding processed elements to test set...")
    test_data = add_processed_elements_to_dataframe(test_data, element_to_idx)
    print(f"  Test set: {len(test_data)} samples")
    print()

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Saved train set to: {train_path}")
    print(f"  Columns: {list(train_data.columns)}")
    print(f"Saved val set to:   {val_path}")
    print(f"  Columns: {list(val_data.columns)}")
    print(f"Saved test set to:  {test_path}")
    print(f"  Columns: {list(test_data.columns)}")
    print()

    return train_path, val_path, test_path


def get_element_vocabulary_from_mat2vec() -> Dict[str, int]:
    """
    Build element -> index from mat2vec ordering; indices are 1-based (0 = padding).
    """
    _, mat2vec_element_to_idx, _ = load_embeddings_matrix()
    element_to_idx = {element: idx + 1 for element, idx in mat2vec_element_to_idx.items()}

    print(f"Loaded element vocabulary from mat2vec: {len(element_to_idx)} elements")
    print(f"  Index range: [1, {len(element_to_idx)}], 0 reserved for padding")
    print(f"  Example mappings: {dict(list(element_to_idx.items())[:5])}")
    return element_to_idx


class FormationEnergyDataset(Dataset):
    """Formation energy regression: padded element indices, fractions, mask."""

    def __init__(self,
                 data,
                 element_to_idx: Dict[str, int],
                 n_max: int = 20):
        self.data = data.reset_index(drop=True)
        self.element_to_idx = element_to_idx
        self.n_max = n_max
        self._parse_all_formulas()
        print(f"Dataset initialized: {len(self.data)} samples, n_max={n_max}")

    def _parse_all_formulas(self):
        parse_errors = 0
        max_elements = 0
        for formula in self.data['formula']:
            try:
                elements, _ = parse_formula_to_unique_elements(formula)
                max_elements = max(max_elements, len(elements))
                for element in elements:
                    if element not in self.element_to_idx:
                        parse_errors += 1
                        break
            except Exception:
                parse_errors += 1
        if parse_errors > 0:
            print(f"Warning: {parse_errors} formulas have parsing issues")
        if max_elements > self.n_max:
            print(f"Warning: Maximum elements ({max_elements}) exceeds n_max ({self.n_max})")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        formula = row['formula']
        formation_energy = float(row['formation_energy'])

        try:
            elements, fractions = parse_formula_to_unique_elements(formula)
        except Exception as e:
            raise ValueError(f"Failed to parse formula '{formula}': {e}") from e

        element_indices = []
        valid_fractions = []
        unknown_elements = []

        for element, fraction in zip(elements, fractions):
            element_normalized = (
                element[0].upper() + element[1:].lower() if len(element) > 1 else element.upper()
            )
            if element_normalized in self.element_to_idx:
                element_indices.append(self.element_to_idx[element_normalized])
                valid_fractions.append(fraction)
            else:
                unknown_elements.append(element)

        if unknown_elements:
            print(f"Warning: Unknown elements {unknown_elements} in formula '{formula}' (skipped)")

        n_elements = len(element_indices)
        if n_elements == 0:
            raise ValueError(
                f"All elements in formula '{formula}' are unknown or invalid. "
                f"Parsed elements: {elements}, Unknown: {unknown_elements}"
            )

        if n_elements > self.n_max:
            element_indices = element_indices[:self.n_max]
            valid_fractions = valid_fractions[:self.n_max]
            n_elements = self.n_max

        element_indices_padded = np.zeros(self.n_max, dtype=np.int64)
        fractions_padded = np.zeros(self.n_max, dtype=np.float32)
        mask = np.zeros(self.n_max, dtype=np.float32)

        element_indices_padded[:n_elements] = element_indices
        fractions_padded[:n_elements] = valid_fractions
        mask[:n_elements] = 1.0

        total = fractions_padded[:n_elements].sum()
        if total > 0:
            fractions_padded[:n_elements] /= total

        return {
            'element_indices': torch.LongTensor(element_indices_padded),
            'fractions': torch.FloatTensor(fractions_padded),
            'mask': torch.FloatTensor(mask),
            'formation_energy': torch.FloatTensor([formation_energy])
        }

    def get_statistics(self) -> Dict:
        element_counts = []
        formation_energies = []
        for i in range(len(self)):
            sample = self[i]
            n_elements = int(sample['mask'].sum().item())
            element_counts.append(n_elements)
            formation_energies.append(sample['formation_energy'].item())
        return {
            'n_samples': len(self),
            'avg_elements_per_sample': np.mean(element_counts),
            'max_elements': np.max(element_counts),
            'min_elements': np.min(element_counts),
            'formation_energy_mean': np.mean(formation_energies),
            'formation_energy_std': np.std(formation_energies),
            'formation_energy_min': np.min(formation_energies),
            'formation_energy_max': np.max(formation_energies)
        }


if __name__ == "__main__":
    from config import DATA_PATH, TRAIN_CONFIG

    print("\n" + "=" * 60)
    print("Dataset Preparation")
    print("=" * 60 + "\n")

    train_path, val_path, test_path = prepare_datasets(
        csv_path=DATA_PATH,
        output_dir="data",
        train_ratio=TRAIN_CONFIG['train_ratio'],
        val_ratio=TRAIN_CONFIG['val_ratio'],
        test_ratio=TRAIN_CONFIG['test_ratio'],
        random_seed=42,
        split_method="stratified"
    )

    print("=" * 60)
    print("Dataset preparation completed.")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    print("\nYou can now use these files to create PyTorch datasets.")
