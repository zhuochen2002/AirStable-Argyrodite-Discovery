"""
Utilities for mat2vec embeddings: extract element vectors from a Word2Vec model and save to JSON.

When run as __main__:
1. Extract embeddings for all 118 periodic elements from the mat2vec model.
2. Save to MAT2VEC_EMBEDDINGS_PATH from config (or default).

Downstream code loads vectors via load_embeddings_from_file().
"""
import os
import sys
import json
import numpy as np
from typing import Dict, Tuple, Optional

# Project root on path for config import
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from config import MAT2VEC_PATH, MAT2VEC_EMBEDDINGS_PATH
except ImportError:
    MAT2VEC_PATH = None
    MAT2VEC_EMBEDDINGS_PATH = "data/met2vec.json"  # fallback if config missing

# Standard periodic table (118 elements)
PERIODIC_TABLE = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]


def _load_word2vec_model(model_path: str):
    """
    Load the mat2vec Word2Vec model from disk.

    Args:
        model_path: Path to the saved Word2Vec model.

    Returns:
        Loaded gensim Word2Vec model.
    """
    try:
        from gensim.models import Word2Vec
        model = Word2Vec.load(model_path)
        return model
    except ImportError:
        raise ImportError(
            "gensim is required to load mat2vec model. "
            "Install it with: pip install gensim"
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load mat2vec model from {model_path}. "
            f"Error: {str(e)}"
        )


def extract_element_embeddings_from_word2vec(model_path: str) -> Dict[str, np.ndarray]:
    """
    Extract one embedding vector per periodic element from the mat2vec Word2Vec model.

    Args:
        model_path: Path to the mat2vec Word2Vec model.

    Returns:
        Mapping from element symbol to embedding vector (float32).
    """
    print(f"Loading Word2Vec model from: {model_path}")
    model = _load_word2vec_model(model_path)
    element_embeddings = {}
    missing_elements = []

    print("Extracting element embeddings...")
    for element in PERIODIC_TABLE:
        try:
            vector = model.wv[element]
            element_embeddings[element] = np.array(vector, dtype=np.float32)
        except KeyError:
            missing_elements.append(element)
            dim = model.wv.vector_size
            element_embeddings[element] = np.zeros(dim, dtype=np.float32)

    if missing_elements:
        print(f"Warning: {len(missing_elements)} elements not in vocabulary, using zero vectors: {missing_elements}")
    else:
        print(f"Successfully extracted embeddings for all {len(PERIODIC_TABLE)} elements")

    return element_embeddings


def save_embeddings_to_json(element_embeddings: Dict[str, np.ndarray], json_path: str):
    """
    Save element embeddings to a JSON file.

    Args:
        element_embeddings: Mapping element symbol -> vector.
        json_path: Output JSON path.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    data = {
        element: vector.tolist()
        for element, vector in element_embeddings.items()
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved embeddings to: {json_path}")


def load_embeddings_from_file(json_path: str = None) -> Dict[str, np.ndarray]:
    """
    Load element embeddings from JSON for modeling.

    Args:
        json_path: Path to JSON (defaults to MAT2VEC_EMBEDDINGS_PATH under project root).

    Returns:
        Mapping element symbol -> embedding [d_mat2vec].
    """
    if json_path is None:
        json_path = os.path.join(_project_root, MAT2VEC_EMBEDDINGS_PATH)

    with open(json_path, 'r') as f:
        data = json.load(f)

    element_embeddings = {
        element: np.array(vector, dtype=np.float32)
        for element, vector in data.items()
    }

    return element_embeddings


def load_embeddings_matrix(
    json_path: str = None
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Load stacked embedding matrix and index maps for modeling.

    Args:
        json_path: Path to JSON (defaults to MAT2VEC_EMBEDDINGS_PATH).

    Returns:
        embeddings: Matrix [n_elements, d_mat2vec].
        element_to_idx: Element symbol -> row index.
        idx_to_element: Row index -> element symbol.
    """
    if json_path is None:
        json_path = os.path.join(_project_root, MAT2VEC_EMBEDDINGS_PATH)

    element_embeddings = load_embeddings_from_file(json_path)

    elements = sorted(element_embeddings.keys())
    element_to_idx = {element: idx for idx, element in enumerate(elements)}
    idx_to_element = {idx: element for element, idx in element_to_idx.items()}

    embeddings_matrix = np.array(
        [element_embeddings[el] for el in elements],
        dtype=np.float32
    )

    return embeddings_matrix, element_to_idx, idx_to_element


def get_element_embedding(
    element_symbol: str,
    embeddings: np.ndarray,
    element_to_idx: Dict[str, int],
    default: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Look up a single element embedding.

    Args:
        element_symbol: e.g. "Li".
        embeddings: Matrix [n_elements, d_mat2vec].
        element_to_idx: Element -> row index.
        default: If set, returned when symbol is missing; else zeros.

    Returns:
        Row vector [d_mat2vec].
    """
    element_symbol = element_symbol.capitalize()

    if element_symbol in element_to_idx:
        idx = element_to_idx[element_symbol]
        return embeddings[idx]
    else:
        if default is not None:
            return default
        else:
            return np.zeros(embeddings.shape[1], dtype=np.float32)


# -----------------------------------------------------------------------------
# CLI: generate and save embeddings
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("mat2vec Element Embeddings Generator")
    print("=" * 60)

    if MAT2VEC_PATH is None:
        print("Error: MAT2VEC_PATH not set in config.py")
        print("Please set MAT2VEC_PATH to the path of mat2vec Word2Vec model")
        sys.exit(1)

    model_path = os.path.abspath(MAT2VEC_PATH)
    output_path = os.path.join(_project_root, MAT2VEC_EMBEDDINGS_PATH)

    if not os.path.exists(model_path):
        print(f"Error: Word2Vec model not found at: {model_path}")
        print("Please check MAT2VEC_PATH in config.py")
        sys.exit(1)

    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    print()

    try:
        element_embeddings = extract_element_embeddings_from_word2vec(model_path)
        save_embeddings_to_json(element_embeddings, output_path)

        print()
        print("Verifying saved file...")
        loaded_embeddings = load_embeddings_from_file(output_path)
        print(f"OK: verified {len(loaded_embeddings)} elements loaded")

        if len(loaded_embeddings) > 0:
            sample_element = list(loaded_embeddings.keys())[0]
            embedding_dim = loaded_embeddings[sample_element].shape[0]
            print(f"  Embedding dimension: {embedding_dim}")
            print(f"  Total elements: {len(loaded_embeddings)}")

        print()
        print("=" * 60)
        print("Embeddings generation completed successfully.")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
