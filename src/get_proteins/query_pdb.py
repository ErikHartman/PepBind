import requests
import pandas as pd
import os
from tqdm import tqdm
import json

data_dir = os.path.abspath("/srv/data1/general/immunopeptides_data/")

extracellular_df = pd.read_csv(
    os.path.join(data_dir, "databases/1_proteins_extracellular_cell_membrane.txt"),
    sep="\t",
)


def query_pdb_by_uniprot(uniprot_id):
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "exact_match",
                        "value": uniprot_id,
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "exact_match",
                        "value": "UniProt",
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                    },
                },
            ],
        },
        "return_type": "polymer_entity",
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    try:
        response = requests.post(url, json=query)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error querying PDB for {uniprot_id}: {e}")
        return []

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        return []

    result_set = data.get("result_set", [])
    pdb_ids = [
        entry.get("identifier") for entry in result_set if entry.get("identifier")
    ]
    return pdb_ids


pdb_results = []


unique_uniprots = extracellular_df["uniprot"].unique()
for uid in tqdm(unique_uniprots, desc="Querying PDB for structures"):
    pdb_ids = query_pdb_by_uniprot(uid)
    pdb_results.append(
        {"uniprot": uid, "pdb_ids": ";".join(pdb_ids) if pdb_ids else None}
    )

pdb_df = pd.DataFrame(pdb_results)

print(len(pdb_df))
result_df = extracellular_df.merge(pdb_df, on="uniprot", how="left")
result_df = result_df.dropna(subset=["pdb_ids"])
print(len(result_df))

output_file = os.path.join(data_dir, "databases/2_proteins_extracellular_cell_membrane_with_pdb.txt")
result_df.to_csv(output_file, sep="\t", index=False)

print(f"Results saved to {output_file}")
print(result_df.head())
