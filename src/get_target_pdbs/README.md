# Fetching target PDBs using reactome and PDB

2 scripts:

1. `mine_reactome.py`: gets the relevant protein IDs from reactome. Requires `0_uniprot2reactome_all_levels_2024_02_04.txt`.
2. `query_pdb.py`: gets the PDB entries for the relevant proteins.

Run `mine_reactome.py` then `query_pdb.py` to generate all files.
