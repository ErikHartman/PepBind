import requests
import json

# PDB API:n 
def fetch_pdb_metadata(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {pdb_id}")
        return None

# Checkar om ligand är en peptid
def is_peptide_binding_protein(metadata):
    try:
        # Print the entire metadata dictionary
        keywords = metadata["struct_keywords"]["pdbx_keywords"]
        return "PEPTIDE BINDING PROTEIN" in keywords.upper()
    except KeyError:
        print("KeyError: 'pdbx_keywords' key not found in metadata")
        return False

# TODO: Skriv om för att göra detta som en polar/pandas dataframe -> Output i aptitligt format
def filter_peptide_binding_proteins(pdb_ids):
    peptide_binding_proteins = []
    for pdb_id in pdb_ids:
        metadata = fetch_pdb_metadata(pdb_id)
        if metadata and is_peptide_binding_protein(metadata):
            peptide_binding_proteins.append(pdb_id)
    return peptide_binding_proteins

# Testlista som Proof of Concept
pdb_ids = ["2B2V", "1b7h", "1bwa", "1a1e", "1a28", "1a30"]  # Replace with your 5K PDB IDs

peptide_binding_pdbs = filter_peptide_binding_proteins(pdb_ids)

print(f"Peptide Binding Proteins: {len(peptide_binding_pdbs)} found")
print(peptide_binding_pdbs)
