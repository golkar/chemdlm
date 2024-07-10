# %%
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol


def visualize_smiles_3d(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    block = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=400, height=400)
    view.addModel(block, "mol")
    view.setStyle({"stick": {}})
    view.zoomTo()
    view.show()


# visualize_smiles_3d("CC(C)(C)C(=O)Nc1ccc(cc1)C(=O)C(C)C")


# %%
def visualize_smiles_side_by_side_3d(smiles_list):
    num_smiles = len(smiles_list)
    if num_smiles == 0:
        raise ValueError("Please provide at least one SMILES string")

    view = py3Dmol.view(width=800, height=300, linked=False, viewergrid=(1, num_smiles))
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string at index {i}")

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        block = Chem.MolToMolBlock(mol)
        view.addModel(block, "mol", viewer=(0, i))
        view.setStyle({"stick": {}}, viewer=(0, i))
        view.zoomTo(viewer=(0, i))

    view.show()


# Example usage:
# visualize_smiles_side_by_side_3d(["CCO", "CCC", "CCN"])
