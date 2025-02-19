import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_dir, "data/databases/refined-set/index/INDEX_general_PP.2020")
output_file_peptide = os.path.join(base_dir, "data/benchmark_data/peptide-like-proteins.csv")

def parse_line(line):
    parts = line.split()
    pdb_code = parts[0]
    resolution = parts[1]
    release_year = parts[2]
    binding_data = parts[3]
    reference = f"{pdb_code}.pdf"
    ligand_info = " ".join(parts[5:])
    ligand_name = ligand_info.split(",")[0]
    return pdb_code, resolution, release_year, binding_data, reference, ligand_name

def is_peptide(ligand_info):
    try:
        length = int(ligand_info.split("-mer")[0].split("(")[-1])
        return length < 50
    except (ValueError, IndexError):
        return False

data = {
    "Ligand name": [],
    "Type": [],
    "PDB code": [],
    "Resolution": [],
    "Release year": [],
    "Binding data": [],
    "Reference": []
}

with open(input_file, 'r') as infile:
    for line in infile:
        if line.startswith("#") or not line.strip():
            continue
        pdb_code, resolution, release_year, binding_data, reference, ligand_info = parse_line(line)
        if is_peptide(ligand_info):
            ligand_name = ligand_info.split(",")[0]
            data["Ligand name"].append(ligand_name)
            data["Type"].append("Peptide")
            data["PDB code"].append(pdb_code)
            data["Resolution"].append(resolution)
            data["Release year"].append(release_year)
            data["Binding data"].append(binding_data)
            data["Reference"].append(reference)

df = pd.DataFrame(data)
df.to_csv(output_file_peptide, index=False)
