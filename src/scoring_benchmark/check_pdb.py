from Bio import PDB
from Bio.PDB.Polypeptide import is_aa

def has_peptide_and_protein(pdb_file, peptide_max_length=50):
    """
    Checks whether the PDB file contains exactly two polypeptide chains:
    one chain with fewer than `peptide_max_length` amino acids (a "peptide"),
    and one chain with >= `peptide_max_length` amino acids (a "protein").

    Returns True if the condition is met, otherwise False.
    """

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("temp_struct", pdb_file)

    chain_residue_counts = []
    model = structure[0]

    for chain in model:
        count_aa = 0
        for residue in chain.get_residues():
            if is_aa(residue, standard=True):
                count_aa += 1
        if count_aa > 0:
            chain_residue_counts.append(count_aa)


    if len(chain_residue_counts) != 2:
        return False


    chain_residue_counts.sort()
    if chain_residue_counts[0] < peptide_max_length and chain_residue_counts[1] >= peptide_max_length:
        return True

    return False


if __name__ == "__main__":
    """
    Run in root.
    """
    pdb_file = "data/test_data/1a22.pdb"
    print(has_peptide_and_protein(pdb_file)) # = False
    pdb_file = "data/test_data/1ssc.pdb"
    print(has_peptide_and_protein(pdb_file)) # = True