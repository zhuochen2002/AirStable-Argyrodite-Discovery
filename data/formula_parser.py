"""Parse chemical formulas into element symbols and stoichiometric fractions."""
import re
from typing import List, Tuple


def parse_formula(formula: str) -> Tuple[List[str], List[float]]:
    """
    Parse a formula string into element symbols and raw counts (not normalized).

    Examples:
        "Li6PS5Cl" -> ["Li", "P", "S", "Cl"], [6, 1, 5, 1]
        "H2O" -> ["H", "O"], [2, 1]
    """
    formula = formula.strip()
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)

    elements = []
    counts = []
    for element, count_str in matches:
        elements.append(element)
        count = int(count_str) if count_str else 1
        counts.append(count)

    if len(elements) == 0:
        raise ValueError(f"Could not parse formula: {formula}")

    return elements, counts


def normalize_fractions(elements: List[str], fractions: List[float]) -> List[float]:
    """Normalize fractions so they sum to 1."""
    total = sum(fractions)
    if total == 0:
        raise ValueError("Sum of fractions is zero")
    return [f / total for f in fractions]


def parse_and_normalize(formula: str) -> Tuple[List[str], List[float]]:
    """Parse formula and return normalized mole fractions in one step."""
    elements, counts = parse_formula(formula)
    normalized_fractions = normalize_fractions(elements, counts)
    return elements, normalized_fractions


def get_unique_elements(elements: List[str]) -> Tuple[List[str], List[int]]:
    """
    Deduplicate elements while preserving order of first occurrence.

    Example: ["Li", "O", "Li"] -> unique ["Li", "O"], indices [0, 1, 0]
    """
    unique_elements = []
    element_to_idx = {}
    indices = []
    for element in elements:
        if element not in element_to_idx:
            element_to_idx[element] = len(unique_elements)
            unique_elements.append(element)
        indices.append(element_to_idx[element])
    return unique_elements, indices


def aggregate_fractions_by_element(elements: List[str], fractions: List[float]) -> Tuple[List[str], List[float]]:
    """
    Sum fractions for duplicate element symbols, then sort elements alphabetically.

    Example: elements=["Li", "O", "Li"], fractions=[2, 1, 1]
        -> unique ["Li", "O"], aggregated [3, 1]
    """
    element_fractions = {}
    for element, fraction in zip(elements, fractions):
        if element in element_fractions:
            element_fractions[element] += fraction
        else:
            element_fractions[element] = fraction
    unique_elements = sorted(element_fractions.keys())
    aggregated_fractions = [element_fractions[el] for el in unique_elements]
    return unique_elements, aggregated_fractions


def parse_formula_to_unique_elements(formula: str) -> Tuple[List[str], List[float]]:
    """
    Parse formula, merge duplicate elements, normalize mole fractions.

    Returns unique elements in alphabetical order and normalized fractions.
    """
    elements, counts = parse_formula(formula)
    unique_elements, aggregated_fractions = aggregate_fractions_by_element(elements, counts)
    normalized_fractions = normalize_fractions(unique_elements, aggregated_fractions)
    return unique_elements, normalized_fractions
