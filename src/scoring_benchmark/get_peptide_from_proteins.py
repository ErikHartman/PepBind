import pandas as pd
import os


def parse_line(line):
    parts = line.split()
    pdb_code = parts[0]
    resolution = parts[1]
    release_year = parts[2]
    binding_data = parts[3]
    reference = f"{pdb_code}.pdb"
    ligand_info = " ".join(parts[5:])
    ligand_name = ligand_info.split(",")[0]
    return pdb_code, resolution, release_year, binding_data, reference, ligand_name


def is_peptide(ligand_info):
    try:
        length = int(ligand_info.split("-mer")[0].split("(")[-1])
        return length < 50
    except (ValueError, IndexError):
        return False


if __name__ == "__main__":

    data_dir = os.path.abspath("/srv/data1/general/immunopeptides_data/")

    input_file = os.path.join(
        data_dir, "databases/benchmark_data/INDEX_general_PP.2020"
    )

    output_file_peptide = os.path.join(
        data_dir, "databases/benchmark_data/peptide_like_proteins.csv"
    )

    data = {
        "Ligand name": [],
        "Type": [],
        "PDB code": [],
        "Resolution": [],
        "Release year": [],
        "Binding data": [],
        "Reference": [],
    }

    with open(input_file, "r") as infile:
        for line in infile:
            if line.startswith("#") or not line.strip():
                continue
            pdb_code, resolution, release_year, binding_data, reference, ligand_info = (
                parse_line(line)
            )
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
    print(f"Wrote info for all chains with peptide length < 50 to: {output_file_peptide}")
