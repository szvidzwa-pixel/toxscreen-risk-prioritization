from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


MORGAN_GENERATOR_CACHE: dict[tuple[int, int], object] = {}


def get_morgan_generator(radius: int, n_bits: int):
    """Reuse RDKit Morgan generators across calls for cleaner, faster featurization."""
    key = (radius, n_bits)
    if key not in MORGAN_GENERATOR_CACHE:
        MORGAN_GENERATOR_CACHE[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
        )
    return MORGAN_GENERATOR_CACHE[key]


def smiles_to_fingerprint(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    """Convert a SMILES string into a Morgan fingerprint."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    generator = get_morgan_generator(radius=radius, n_bits=n_bits)
    bit_vector = generator.GetFingerprint(mol)
    array = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def featurize_smiles(smiles_values, radius: int, n_bits: int):
    """Featurize a sequence of SMILES strings and keep valid row indices."""
    rows = []
    valid_indices = []

    for idx, smiles in enumerate(smiles_values):
        fingerprint = smiles_to_fingerprint(smiles, radius=radius, n_bits=n_bits)
        if fingerprint is None:
            continue
        rows.append(fingerprint)
        valid_indices.append(idx)

    if not rows:
        raise ValueError("No valid SMILES strings were featurized.")

    return np.asarray(rows), valid_indices
