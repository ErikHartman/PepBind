import requests
import pandas as pd
import time
from tqdm import tqdm

pathways = {
    "tlr cascades": "R-HSA-168898",
    "complement cascade": "R-HSA-166658",
    "neutrophil degranulation": "R-HSA-6798695",
    "interferon signaling": "R-HSA-913531",
    "signaling by interleukins": "R-HSA-449147",
    "TNFR2 non-canonical NF-kB pathway": "R-HSA-5668541",
}

uniprot2reactome = pd.read_csv(
    "/Users/erikhartman/dev/immunopeptides/data/databases/0_uniprot2reactome_all_levels_2024_02_04.txt",
    names=["uniprot", "reactome", "url", "rxn_name", "x", "species"],
    sep="\t",
)

uniprot2reactome = uniprot2reactome[
    uniprot2reactome["reactome"].isin(pathways.values())
]

uniprot_ids = uniprot2reactome["uniprot"].unique()


def is_extracellular(uniprot_id):
    """
    Query UniProt for the given UniProt ID and return:
      - A boolean indicating if the protein is annotated as extracellular.
      - The protein's full name.
      - The protein's short name.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)
    if not response.ok:
        print(f"Warning: Could not retrieve data for {uniprot_id}")
        return False, "", ""

    data = response.json()

    protein_desc = data.get("proteinDescription", {})
    fullName = (
        protein_desc.get("recommendedName", {}).get("fullName", {}).get("value", "")
    )
    uniProtkbId = data.get("uniProtkbId", "")
    comments = data.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "SUBCELLULAR LOCATION":
            locations = comment.get("subcellularLocations", [])
            for loc in locations:
                location_value = loc.get("location", {}).get("value", "")

                if "extracellular" in location_value.lower():
                    return True, fullName, uniProtkbId

                if "cell membrane" in location_value.lower():
                    return True, fullName, uniProtkbId

    return False, fullName, uniProtkbId


extracellular_ids = []
fullNames = []
uniProtkbIds = []

def is_wanted_protein(fullName, uniProtkbId):
    if "immunoglobulin" in fullName.lower():
        return False
    if "hla class" in fullName.lower():
        return False
    if not uniProtkbId:
        return False
    if not fullName:
        return False
    if fullName.lower().startswith("isoform"):
        return False
    return True

for uid in tqdm(uniprot_ids, desc="Processing UniProt IDs"):
    extracellular, fullName, uniProtkbId = is_extracellular(uid)
    if is_wanted_protein(fullName, uniProtkbId):
        if extracellular:
            extracellular_ids.append(uid)
            fullNames.append(fullName)
            uniProtkbIds.append(uniProtkbId)
    #time.sleep(0.1)

print("Extracellular proteins:")
print(uniProtkbIds)


info_df = pd.DataFrame(
    {"uniprot": extracellular_ids, "fullName": fullNames, "uniProtkbId": uniProtkbIds}
)

extracellular_df = uniprot2reactome[
    uniprot2reactome["uniprot"].isin(extracellular_ids)
].copy()

extracellular_df = extracellular_df.merge(info_df, on="uniprot", how="left")

print(extracellular_df)

extracellular_df.to_csv(
    "/Users/erikhartman/dev/immunopeptides/data/databases/1_uniprot2reactome_all_levels_2024_02_04_extracellular_cell_membrane.txt",
    index=False,
    sep="\t",
)

info_df.to_csv(
    "/Users/erikhartman/dev/immunopeptides/data/databases/1_proteins_extracellular_cell_membrane.txt",
    index=False,
    sep="\t",
)
