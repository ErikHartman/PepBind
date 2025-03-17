from Bio.PDB import PDBParser, PDBIO, Select

def remove_peptide_from_complex(pdb_path, output_pdb_path, protein_chain="A"):
    """
    Remove a specified peptide chain from a PDB file, leaving only the protein
    chain (and any other chains not specified as the peptide).

    """

    # Parse the structure
    parser = PDBParser(QUIET=True)  # QUIET avoids a lot of console output
    structure = parser.get_structure("complex", pdb_path)

    # We will create a custom selection class that excludes the peptide chain
    class PeptideRemoverSelect(Select):
        def accept_chain(self, chain):
            # Keep chain if it is the protein chain
            if chain.id == protein_chain:
                return True  # include
            return False  # exclude

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path, select=PeptideRemoverSelect())

    return output_pdb_path


def get_contact_residues(
    structure, protein_chain_id="A", peptide_chain_id="B", cutoff=4.0
):
    """
    Identify residues on the protein chain that are within 'cutoff' angstroms
    of any atom in the peptide chain.
    """
    # Extract chain objects
    model = structure[0]  # assume only one model
    protein_chain = model[protein_chain_id]
    peptide_chain = model[peptide_chain_id]

    contact_residues = set()

    # Get all atoms from the peptide
    peptide_atoms = list(peptide_chain.get_atoms())

    # For each residue in the protein chain, check if it's within 'cutoff' of the peptide
    for residue in protein_chain.get_residues():
        # Skip heteroatoms or water
        if not residue.id[0] == " ":  # Standard residues have id[0] == " "
            continue
        for atom in residue:
            # If any atom of this residue is within cutoff of any peptide atom, mark contact
            for pep_atom in peptide_atoms:
                dist = atom - pep_atom  # operator- gives distance
                if dist <= cutoff:
                    # We store (chain_id, (resname, resseq, icode)) or something simpler
                    contact_residues.add((protein_chain_id, residue.id[1]))
                    # Break to avoid double-counting the same residue
                    break
            else:
                # If we never broke out, keep checking next residue atom
                continue
            # Once we've added the residue, no need to check additional atoms
            break

    return contact_residues


def compare_binding_site(
    pdb1_path,
    pdb2_path,
    protein_chain="A",
    peptide_chain="B",
    distance_cutoff=4.0,
    overlap_threshold=0.5,
):
    """
    Compare the binding sites of two complexes (same sequences, different poses).
    We define "binding site" as the set of protein residues within 'distance_cutoff'
    angstroms of the peptide.

    Then we compare the overlap of these residues in structure 1 vs. structure 2.
    """

    # Parse PDB structures
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure("complex1", pdb1_path)
    structure2 = parser.get_structure("complex2", pdb2_path)

    # Identify contact residues in each structure
    contact_residues_1 = get_contact_residues(
        structure1,
        protein_chain_id=protein_chain,
        peptide_chain_id=peptide_chain,
        cutoff=distance_cutoff,
    )
    contact_residues_2 = get_contact_residues(
        structure2,
        protein_chain_id=protein_chain,
        peptide_chain_id=peptide_chain,
        cutoff=distance_cutoff,
    )

    # Compute overlap
    intersection = contact_residues_1.intersection(contact_residues_2)
    union = contact_residues_1.union(contact_residues_2)

    if len(union) == 0:
        overlap_fraction = 0.0
    else:
        overlap_fraction = len(intersection) / len(union)
    same_site = overlap_fraction >= overlap_threshold

    return same_site, overlap_fraction
