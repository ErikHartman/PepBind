import re
import requests
import concurrent.futures
import csv
import os
import pandas as pd

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

GRAPHQL_QUERY = """
query molecule($id: String!) {
    chem_comp(comp_id: $id) {
        chem_comp {
            id
            name
            type
        }
    }
}
"""

def is_peptide_like(ligand_id):
    try:
        response = requests.post(
            RCSB_GRAPHQL_URL,
            json={"query": GRAPHQL_QUERY, "variables": {"id": ligand_id}},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        chem_comp = data.get("data", {}).get("chem_comp", None)

        if (chem_comp is None):
            return ligand_id, False, None
        chem_type = chem_comp.get("chem_comp", {}).get("type", "").lower()

        if "peptide-like" in chem_type:
            return ligand_id, True, chem_type
        
        return ligand_id, False, chem_type
    
    except requests.exceptions.RequestException as e:
        print(f"Error querying ligand {ligand_id}: {e}")
    return ligand_id, False, None

def process_ligands(df):
    peptide_like_ligands = []
    non_peptide_like_ligands = []
    unique_ligand_names = df['Ligand name'].unique()
    total_ligands = len(unique_ligand_names)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_ligand = {executor.submit(is_peptide_like, ligand_id): ligand_id for ligand_id in unique_ligand_names}
        for index, future in enumerate(concurrent.futures.as_completed(future_to_ligand)):
            ligand_id, is_peptide, chem_type = future.result()
            if is_peptide:
                peptide_like_ligands.append((ligand_id, chem_type))
            else:
                non_peptide_like_ligands.append((ligand_id, chem_type))
            print(f"Processed {index + 1}/{total_ligands} ligands - {ligand_id} is {'peptide-like' if is_peptide else 'not peptide-like'}", end='\r')
    
    peptide_like_df = pd.DataFrame(peptide_like_ligands, columns=["Ligand name", "Type"])
    non_peptide_like_df = pd.DataFrame(non_peptide_like_ligands, columns=["Ligand name", "Type"])
    
    return peptide_like_df, non_peptide_like_df

def convert_to_dataframe(input_file):
    data = []
    with open(input_file, 'r') as infile:
        for line in infile:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            pdb_code = parts[0]
            resolution = parts[1]
            release_year = parts[2]
            binding_data = parts[3]
            reference = parts[5]
            ligand_name = parts[6]
            data.append([pdb_code, resolution, release_year, binding_data, reference, ligand_name])
    
    df = pd.DataFrame(data, columns=['PDB code', 'Resolution', 'Release year', 'Binding data', 'Reference', 'Ligand name'])
    return df




def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, "data/databases/refined-set/index/INDEX_refined_set.2020")
    output_file_peptide = os.path.join(base_dir, "data/databases/benchmark_data/peptide_like_ligands.csv")
    output_file_non_peptide = os.path.join(base_dir, "data/databases/benchmark_data/peptide_like_ligands.csv")

    os.makedirs(os.path.dirname(output_file_peptide), exist_ok=True)

    df = convert_to_dataframe(input_file)
    df['Ligand name'] = df['Ligand name'].str.strip('()')

    peptide_like_df, non_peptide_like_df = process_ligands(df)

    merged_df = df.merge(peptide_like_df, on="Ligand name", how="inner")

    peptide_like_df.to_csv(output_file_peptide, index=False)
    non_peptide_like_df.to_csv(output_file_non_peptide, index=False)
    merged_df.to_csv(os.path.join(base_dir, "output_data/merged_peptide_like_ligands.csv"), index=False)

    print(f"\nPeptide-like ligands have been written to {output_file_peptide}")
    print(f"Non-peptide-like ligands have been written to {output_file_non_peptide}")
    print(f"Merged peptide-like ligands have been written to {os.path.join(base_dir, 'output_data/merged_peptide_like_ligands.csv')}")

if __name__ == "__main__":
    main()

