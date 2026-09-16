#!/usr/bin/env python3
"""
MCES Calculator — Maximum Common Edge Subgraph distance between two molecules.

Usage:
    python mces.py "CCO" "CCCO"
    python mces.py  (prompts for input)
"""

import sys
from rdkit import Chem
from rdkit.Chem import rdFMCS


def mces_distance(smi_a: str, smi_b: str) -> dict:
    mol_a = Chem.MolFromSmiles(smi_a.strip())
    mol_b = Chem.MolFromSmiles(smi_b.strip())

    if mol_a is None:
        raise ValueError(f"Invalid SMILES for molecule A: {smi_a}")
    if mol_b is None:
        raise ValueError(f"Invalid SMILES for molecule B: {smi_b}")

    result = rdFMCS.FindMCS(
        [mol_a, mol_b],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        matchValences=True,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        timeout=10,
    )

    n_a = mol_a.GetNumBonds()
    n_b = mol_b.GetNumBonds()
    mcs_bonds = result.numBonds
    similarity = (2 * mcs_bonds) / (n_a + n_b) if (n_a + n_b) > 0 else 0.0
    distance = 1.0 - similarity

    return {
        "smiles_a":   smi_a,
        "smiles_b":   smi_b,
        "bonds_a":    n_a,
        "bonds_b":    n_b,
        "mcs_bonds":  mcs_bonds,
        "mcs_atoms":  result.numAtoms,
        "mcs_smarts": result.smartsString,
        "similarity": similarity,
        "distance":   distance,
    }


def main():
    if len(sys.argv) == 3:
        smi_a, smi_b = sys.argv[1], sys.argv[2]
    else:
        print("MCES Calculator")
        print("-" * 40)
        smi_a = input("SMILES A: ").strip()
        smi_b = input("SMILES B: ").strip()

    r = mces_distance(smi_a, smi_b)

    print()
    print("=" * 40)
    print(f"  Bonds in A     : {r['bonds_a']}")
    print(f"  Bonds in B     : {r['bonds_b']}")
    print(f"  MCS bonds      : {r['mcs_bonds']}")
    print(f"  MCS atoms      : {r['mcs_atoms']}")
    print(f"  Similarity     : {r['similarity']:.4f}  ({r['similarity']*100:.1f}%)")
    print(f"  MCES distance  : {r['distance']:.4f}")
    print(f"  MCS SMARTS     : {r['mcs_smarts']}")
    print("=" * 40)


if __name__ == "__main__":
    main()