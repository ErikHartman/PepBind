import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def get_peptide_lengths(directory):
    lengths = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pdb"):
            with open(os.path.join(directory, filename), 'r') as file:
                residues_by_chain = defaultdict(set)
                for line in file:
                    if line.startswith("ATOM"):
                        residue_id = line[22:26].strip()
                        chain_id = line[21]
                        residues_by_chain[chain_id].add(residue_id)
                if residues_by_chain:
                    lengths[filename] = min(len(residues) for residues in residues_by_chain.values())
    return lengths

def plot_peptide_lengths(directory):

    """ 
    Plots the length of the shortest chain in each PDB file in the given directory.

    :param directory: Directory containing PDB files
    :output: Saves a plot of the distribution of peptide lengths to the directory

    """
    peptide_lengths = get_peptide_lengths(directory)
    length_counts = Counter(peptide_lengths.values())
    
    lengths = list(length_counts.keys())
    counts = list(length_counts.values())
    
    plt.figure(figsize=(10, 5))
    plt.bar(lengths, counts, color='blue')
    plt.xlabel('Length of Peptides (number of residues)')
    plt.ylabel('Count of Peptides')
    plt.title('Distribution of Peptide Lengths in PDB Complex Files')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    output_file = os.path.join(directory, "../peptide_lengths.png")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

