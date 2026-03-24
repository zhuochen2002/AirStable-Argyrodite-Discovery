"""Load CSV datasets and split into train / validation / test sets."""
import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, List

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.formula_parser import parse_formula_to_unique_elements


def _round_bucket(x: float, step: float = 0.05) -> float:
    """Quantize a value into buckets of width `step` (e.g. 0.05 for 5% mole fraction bins)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return round(float(x) / step) * step


def build_comp_group(elements: List[str], fractions: List[float], step: float = 0.05) -> str:
    """
    Build a composition group key: normalized mole fractions (binned), sorted by element.

    Example: "Li:0.50|P:0.10|S:0.40"
    """
    if not elements or not fractions:
        return "EMPTY"
    total = sum(abs(f) for f in fractions) or 1.0
    norm = {el: f / total for el, f in zip(elements, fractions)}
    items = sorted(
        [(el, _round_bucket(v, step)) for el, v in norm.items()],
        key=lambda x: x[0]
    )
    return "|".join([f"{el}:{v:.2f}" for el, v in items])


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load CSV with required columns material_id, formula, formation_energy."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    required_columns = ['material_id', 'formula', 'formation_energy']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    initial_count = len(data)
    data = data.dropna(subset=['formula', 'formation_energy'])
    if len(data) < initial_count:
        print(f"Removed {initial_count - len(data)} rows with missing values")

    print("Building composition groups (comp_group) from formulas...")
    comp_groups = []
    elements_sets = []
    parse_errors = 0

    for _, row in data.iterrows():
        formula = row['formula']
        try:
            elements, fractions = parse_formula_to_unique_elements(formula)
            if not elements:
                comp_groups.append("EMPTY")
                elements_sets.append(set())
                continue
            cg = build_comp_group(elements, fractions, step=0.05)
            comp_groups.append(cg)
            elements_sets.append(set(elements))
        except Exception as e:
            parse_errors += 1
            comp_groups.append("PARSE_ERROR")
            elements_sets.append(set())
            if parse_errors <= 5:
                print(f"  Warning: Failed to parse formula '{formula}': {e}")

    if parse_errors > 0:
        print(f"  Warning: {parse_errors} formulas had parsing issues when building comp_group.")

    data = data.copy()
    data['comp_group'] = comp_groups
    data['elements_set'] = elements_sets

    print(f"Loaded {len(data)} samples from {csv_path}")
    print(f"  Unique composition groups (comp_group): {data['comp_group'].nunique()}")
    return data


def split_dataset_random(data: pd.DataFrame,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1,
                        random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random shuffle split."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    print("Splitting dataset (random)...")
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(data))

    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train + n_val]
    test_indices = shuffled_indices[n_train + n_val:]

    train_data = data.iloc[train_indices].copy().reset_index(drop=True)
    val_data = data.iloc[val_indices].copy().reset_index(drop=True)
    test_data = data.iloc[test_indices].copy().reset_index(drop=True)

    print(f"Train: {len(train_data)} samples ({train_ratio*100:.1f}%)")
    print(f"Val:   {len(val_data)} samples ({val_ratio*100:.1f}%)")
    print(f"Test:  {len(test_data)} samples ({test_ratio*100:.1f}%)")

    return train_data, val_data, test_data


def split_dataset_stratified(data: pd.DataFrame,
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1,
                            test_ratio: float = 0.1,
                            random_seed: int = 42,
                            n_bins: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split on formation_energy bins (quantile-based)."""
    from sklearn.model_selection import train_test_split

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    print("Splitting dataset (stratified by formation energy)...")
    formation_energies = data['formation_energy'].values

    actual_n_bins = n_bins
    labels = None

    for _ in range(10):
        try:
            labels = pd.qcut(formation_energies, q=actual_n_bins, labels=False, duplicates='drop')
            unique_labels = np.unique(labels[~np.isnan(labels)])
            if len(unique_labels) == 0:
                raise ValueError("No valid labels")
            label_counts = pd.Series(labels).value_counts()
            min_count = label_counts.min()
            if min_count >= 2:
                if actual_n_bins < n_bins:
                    print(f"  Adjusted n_bins from {n_bins} to {len(unique_labels)} to ensure each bin has at least 2 samples")
                label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
                labels = np.array([label_map.get(l, -1) for l in labels])
                valid_mask = labels >= 0
                if not valid_mask.all():
                    data = data[valid_mask].reset_index(drop=True)
                    labels = labels[valid_mask]
                break
            actual_n_bins = max(2, actual_n_bins - 1)
        except (ValueError, TypeError):
            actual_n_bins = max(2, actual_n_bins - 1)

        if actual_n_bins < 2:
            print("  Warning: Cannot create stratified split with sufficient samples per bin. Falling back to random split.")
            return split_dataset_random(data, train_ratio, val_ratio, test_ratio, random_seed)

    label_counts = pd.Series(labels).value_counts().sort_index()
    while label_counts.min() < 2 and len(label_counts) > 1:
        min_label = label_counts.idxmin()
        sorted_labels = sorted(label_counts.index)
        min_idx = sorted_labels.index(min_label)
        if min_idx == 0:
            merge_to = sorted_labels[1]
        elif min_idx == len(sorted_labels) - 1:
            merge_to = sorted_labels[min_idx - 1]
        else:
            prev_count = label_counts[sorted_labels[min_idx - 1]]
            next_count = label_counts[sorted_labels[min_idx + 1]]
            merge_to = sorted_labels[min_idx - 1] if prev_count > next_count else sorted_labels[min_idx + 1]
        labels[labels == min_label] = merge_to
        label_counts = pd.Series(labels).value_counts().sort_index()

    unique_labels = sorted(np.unique(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

    temp_ratio = val_ratio + test_ratio
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels,
        test_size=temp_ratio,
        stratify=labels,
        random_state=random_seed
    )

    val_ratio_in_temp = val_ratio / temp_ratio
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        test_size=(1 - val_ratio_in_temp),
        stratify=temp_labels,
        random_state=random_seed
    )

    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    print(f"Train: {len(train_data)} samples ({train_ratio*100:.1f}%)")
    print(f"  Formation energy: mean={train_data['formation_energy'].mean():.4f}, std={train_data['formation_energy'].std():.4f}")
    print(f"Val:   {len(val_data)} samples ({val_ratio*100:.1f}%)")
    print(f"  Formation energy: mean={val_data['formation_energy'].mean():.4f}, std={val_data['formation_energy'].std():.4f}")
    print(f"Test:  {len(test_data)} samples ({test_ratio*100:.1f}%)")
    print(f"  Formation energy: mean={test_data['formation_energy'].mean():.4f}, std={test_data['formation_energy'].std():.4f}")

    return train_data, val_data, test_data


def split_dataset_by_energy_range(data: pd.DataFrame,
                                  train_ratio: float = 0.8,
                                  val_ratio: float = 0.1,
                                  test_ratio: float = 0.1,
                                  random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by formation energy, split into blocks, assign within each block to
    encourage coverage of the energy range in each split.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    print("Splitting dataset (by energy range, ensuring coverage)...")
    sorted_data = data.sort_values('formation_energy').reset_index(drop=True)

    n_total = len(sorted_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    n_blocks = 10
    block_size = n_total // n_blocks

    train_indices = []
    val_indices = []
    test_indices = []

    np.random.seed(random_seed)

    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < n_blocks - 1 else n_total
        block_indices = np.arange(start_idx, end_idx)
        shuffled_block = np.random.permutation(block_indices)
        block_n_train = int(len(shuffled_block) * train_ratio)
        block_n_val = int(len(shuffled_block) * val_ratio)
        train_indices.extend(shuffled_block[:block_n_train])
        val_indices.extend(shuffled_block[block_n_train:block_n_train + block_n_val])
        test_indices.extend(shuffled_block[block_n_train + block_n_val:])

    train_indices = np.random.permutation(train_indices)
    val_indices = np.random.permutation(val_indices)
    test_indices = np.random.permutation(test_indices)

    train_data = sorted_data.iloc[train_indices].copy().reset_index(drop=True)
    val_data = sorted_data.iloc[val_indices].copy().reset_index(drop=True)
    test_data = sorted_data.iloc[test_indices].copy().reset_index(drop=True)

    print(f"Train: {len(train_data)} samples ({train_ratio*100:.1f}%)")
    print(f"  Formation energy range: [{train_data['formation_energy'].min():.4f}, {train_data['formation_energy'].max():.4f}]")
    print(f"Val:   {len(val_data)} samples ({val_ratio*100:.1f}%)")
    print(f"  Formation energy range: [{val_data['formation_energy'].min():.4f}, {val_data['formation_energy'].max():.4f}]")
    print(f"Test:  {len(test_data)} samples ({test_ratio*100:.1f}%)")
    print(f"  Formation energy range: [{test_data['formation_energy'].min():.4f}, {test_data['formation_energy'].max():.4f}]")

    return train_data, val_data, test_data


def split_dataset(data: pd.DataFrame,
                 method: str = "stratified",
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_seed: int = 42,
                 **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test.

    method: "stratified" | "random" | "energy_range"
    Extra kwargs: n_bins for stratified (default 10).
    """
    if method == "random":
        return split_dataset_random(data, train_ratio, val_ratio, test_ratio, random_seed)
    if method == "stratified":
        n_bins = kwargs.get('n_bins', 10)
        return split_dataset_stratified(data, train_ratio, val_ratio, test_ratio, random_seed, n_bins)
    if method == "energy_range":
        return split_dataset_by_energy_range(data, train_ratio, val_ratio, test_ratio, random_seed)
    raise ValueError(f"Unknown split method: {method}. Choose from: 'random', 'stratified', 'energy_range'")
