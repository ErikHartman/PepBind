import os
from bopep import Scorer

if __name__ == "__main__":

    data_dir = os.path.abspath(
        "/srv/data1/general/immunopeptides_data/databases/benchmark_data"
    )

    pdb_dir = os.path.join(data_dir, "pdbs")

    scorer = Scorer()

    for pdb in os.listdir(pdb_dir):
        pdb_path = os.path.join(pdb_dir, pdb)
        print(pdb_path)
        print(
            scorer.calculate_rosetta_scores(pdb_path),
        )
        break
