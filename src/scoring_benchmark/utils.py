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
            for pep_atom in peptide_atoms:
                dist = atom - pep_atom  # operator- gives distance
                if dist <= cutoff:
                    contact_residues.add((protein_chain_id, residue.id[1]))
                    # Break to avoid double-counting the same residue
                    break
            else:
                # If we never broke out, keep checking next residue atom
                continue
            # Once we've added the residue, no need to check additional atoms
            break

    return contact_residues


def get_interface_residues_in_pdb(
    pdb_path,
    protein_chain="A",
    peptide_chain="B",
    cutoff=5.0
):
    """
    Given a PDB file with a protein (chain A) and a peptide (chain B),
    return the residue indices on the protein chain within 'cutoff'
    angstroms of the peptide.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    contact_residues = get_contact_residues(
        structure, 
        protein_chain_id=protein_chain, 
        peptide_chain_id=peptide_chain, 
        cutoff=cutoff
    )
    
    residue_indices = sorted([res_id for (_, res_id) in contact_residues])
    
    return residue_indices


def ensure_peptide_is_chain_b(pdb_path, output_pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    model = structure[0]
    
    # Determine chain lengths (only counting standard residues)
    chain_lengths = {}
    for chain in model:
        length = sum(1 for r in chain.get_residues() if r.id[0] == " ")
        chain_lengths[chain.id] = length
    
    # Identify shortest chain as new 'B' and another chain as 'A'
    sorted_chains = sorted(chain_lengths.items(), key=lambda x: x[1])
    peptide_chain_id = sorted_chains[0][0]
    protein_chain_id = sorted_chains[-1][0]
    
    # Rename chain IDs if needed
    for chain in model:
        if chain.id == peptide_chain_id:
            chain.id = "B"
        elif chain.id == protein_chain_id:
            chain.id = "A"
            
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)
    return output_pdb_path