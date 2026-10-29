"""Sequence-defined grouping and a fixed, outcome-blind holdout search.

No affinity values, structural score features, or model predictions are used.
MMseqs2 reports detected sequence relationships; this is not a screen against
the structure predictors' training databases.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


def build_similarity_groups(
    records: pd.DataFrame,
    *,
    mmseqs: str = "mmseqs",
    receptor_identity: float = 0.30,
    receptor_coverage: float = 0.80,
    receptor_evalue: float = 0.001,
    peptide_similarity: float = 0.80,
    threads: int = 2,
) -> dict:
    """Connect rows sharing a PDB, receptor, or sufficiently similar peptide.

    ``records`` has a unique complex_filename index and target_sequence,
    peptide_sequence, pdb_code, and kind columns. Receptor hits must satisfy
    identity and E-value thresholds and cover the specified fraction of either
    receptor. Peptide similarity is 1 - global Levenshtein distance / max length;
    this explicitly includes substitutions, insertions, and deletions. All
    qualifying links are joined transitively, including links through decoys.

    Returns membership, row-level edges, all significant receptor_hits, all
    peptide_hits (for the later distance criterion), and metadata. Edges use
    representative rows for each unique sequence, plus exact-sequence links;
    they therefore encode the same connected components without expanding
    every sequence pair into every corresponding row pair.
    """
    from Bio.Align import PairwiseAligner

    required = {"target_sequence", "peptide_sequence", "pdb_code", "kind"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"Missing sequence metadata columns: {sorted(missing)}")
    if records.empty or not records.index.is_unique or records.index.hasnans:
        raise ValueError("Records need a nonempty, unique complex_filename index.")
    if not records.kind.isin(["real", "shuffle", "random"]).all():
        raise ValueError("kind must be real, shuffle, or random.")
    if not all(0 < value <= 1 for value in
               [receptor_identity, receptor_coverage, peptide_similarity]):
        raise ValueError("Identity, coverage, and peptide thresholds must be in (0, 1].")
    executable = shutil.which(str(mmseqs))
    if executable is None:
        raise FileNotFoundError(f"MMseqs2 executable not found: {mmseqs}")

    membership = records.copy()
    membership.index = membership.index.astype(str)
    membership.index.name = "complex_filename"
    if not membership.index.is_unique:
        raise ValueError("complex_filename identifiers collide after conversion to strings.")
    for column, prefix, id_column in [
        ("target_sequence", "t_", "target_id"),
        ("peptide_sequence", "p_", "peptide_id"),
    ]:
        if membership[column].isna().any():
            raise ValueError(f"Missing {column} values.")
        membership[column] = membership[column].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
        if not membership[column].str.fullmatch(r"[A-Z]+").all():
            raise ValueError(f"{column} must contain nonempty amino-acid sequences.")
        membership[id_column] = membership[column].map(
            lambda sequence: prefix + hashlib.sha256(sequence.encode()).hexdigest()[:20])
        if not membership.groupby(id_column)[column].nunique().eq(1).all():
            raise ValueError("Sequence identifier hash collision.")
    if membership.pdb_code.isna().any():
        raise ValueError("Every complex needs its source pdb_code, including decoys.")
    membership["pdb_code"] = membership.pdb_code.astype(str).str.lower()

    # Union-find over observations preserves links through either molecule.
    names = membership.index.to_numpy()
    positions = {name: i for i, name in enumerate(names)}
    parent = np.arange(len(names))
    edges = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def connect(left, right, source, similarity=1.0):
        if left == right:
            return
        left_root, right_root = find(positions[left]), find(positions[right])
        parent[right_root] = left_root
        edges.append((left, right, source, float(similarity)))

    for column in ["pdb_code", "target_id", "peptide_id"]:
        for _, group in membership.groupby(column, sort=True):
            group_names = sorted(group.index)
            for other in group_names[1:]:
                connect(group_names[0], other, f"same_{column}")

    targets = (membership.reset_index().sort_values("complex_filename")
               .drop_duplicates("target_id").set_index("target_id").sort_index())
    peptides = (membership.reset_index().sort_values("complex_filename")
                .drop_duplicates("peptide_id").set_index("peptide_id").sort_index())
    hit_columns = ["query", "target", "identity", "alignment_length", "query_length",
                   "target_length", "query_coverage", "target_coverage", "evalue", "bits"]
    with tempfile.TemporaryDirectory(prefix="revision2_sequence_split_") as temp:
        temporary = Path(temp)
        fasta = temporary / "receptors.fasta"
        fasta.write_text("".join(f">{key}\n{row.target_sequence}\n"
                                 for key, row in targets.iterrows()))
        # Search permissively so subthreshold hits are also available for the
        # distance-based split criterion. Coverage is imposed below in either
        # direction; no representative-only clustering is used.
        command = [executable, "easy-search", str(fasta), str(fasta),
                   str(temporary / "hits.tsv"), str(temporary / "work"),
                   "--min-seq-id", "0", "-c", "0", "--cov-mode", "0",
                   "-e", str(receptor_evalue), "-s", "7.5", "--alignment-mode", "3",
                   "--max-seqs", str(max(len(targets), 10000)), "--threads", str(threads),
                   "--format-output", "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise RuntimeError(f"MMseqs2 failed:\n{result.stdout[-5000:]}")
        hit_path = temporary / "hits.tsv"
        receptor_hits = (pd.read_csv(hit_path, sep="\t", names=hit_columns)
                         if hit_path.stat().st_size else pd.DataFrame(columns=hit_columns))
    covered = (receptor_hits.query_coverage.ge(receptor_coverage)
               | receptor_hits.target_coverage.ge(receptor_coverage))
    receptor_hits["qualifies"] = (covered & receptor_hits.identity.ge(receptor_identity)
                                  & receptor_hits.evalue.le(receptor_evalue))
    for row in receptor_hits.loc[receptor_hits.qualifies].itertuples():
        connect(targets.at[row.query, "complex_filename"], targets.at[row.target, "complex_filename"],
                "receptor_similarity", row.identity)

    aligner = PairwiseAligner(mode="global", match_score=0, mismatch_score=-1,
                             open_gap_score=-1, extend_gap_score=-1)
    peptide_rows = []
    peptide_items = list(peptides.peptide_sequence.items())
    for i, (left_id, left_sequence) in enumerate(peptide_items):
        for right_id, right_sequence in peptide_items[i + 1:]:
            distance = -aligner.score(left_sequence, right_sequence)
            similarity = 1.0 - distance / max(len(left_sequence), len(right_sequence))
            qualifies = similarity + 1e-12 >= peptide_similarity
            peptide_rows.append((left_id, right_id, similarity, qualifies))
            if qualifies:
                connect(peptides.at[left_id, "complex_filename"], peptides.at[right_id, "complex_filename"],
                        "peptide_similarity", similarity)
    peptide_hits = pd.DataFrame(peptide_rows, columns=["query", "target", "similarity", "qualifies"])

    components = {}
    for name, position in positions.items():
        components.setdefault(find(position), []).append(name)
    group_by_name = {}
    for number, component in enumerate(sorted(components.values(), key=lambda values: min(values))):
        group_by_name.update({name: f"g_{number:04d}" for name in component})
    membership["similarity_group"] = membership.index.map(group_by_name)
    edge_frame = pd.DataFrame(edges, columns=["left_complex", "right_complex", "source", "similarity"])
    metadata = {
        "receptor_identity": receptor_identity, "receptor_coverage": receptor_coverage,
        "receptor_coverage_rule": "either sequence", "receptor_evalue": receptor_evalue,
        "peptide_similarity": peptide_similarity,
        "peptide_similarity_definition": "1 - global Levenshtein distance / max(sequence lengths)",
        "mmseqs_version": subprocess.check_output([executable, "version"], text=True).strip(),
        "mmseqs_command": command, "n_records": len(membership), "n_targets": len(targets),
        "n_peptides": len(peptides), "n_groups": membership.similarity_group.nunique(),
    }
    return {"membership": membership, "edges": edge_frame,
            "receptor_hits": receptor_hits, "peptide_hits": peptide_hits, "metadata": metadata}


def choose_similarity_split(
    grouped: dict,
    *,
    seed: int = 42,
    n_candidates: int = 2000,
    fractions: tuple = (0.60, 0.20, 0.20),
    fraction_tolerance: float = 0.10,
    min_real: int = 10,
    min_decoys: int = 10,
    min_groups: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the most distant admissible split among fixed random candidates.

    Components are indivisible. Candidate generation uses a fixed random seed;
    admissibility uses only real/shuffle/random sample counts. Each kind's
    allocation must be within ``fraction_tolerance`` of its requested fraction
    in all three splits. Minimum numbers of real rows, decoys, and groups apply
    to each split. Empty kinds are omitted from fraction checks.

    Among admissible candidates, minimize the largest test-versus-development
    sequence similarity normalized by its grouping threshold. Ties minimize
    the sum of the two normalized maxima, then allocation error, then candidate
    index. Receptor similarity uses detected significant hits covering either
    sequence; peptide similarity uses all global edit comparisons. Undetected
    receptor relationships contribute zero, which is not proof of no homology.
    This is a finite search, not a claim of a globally most-dissimilar partition.
    """
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError("fractions must contain three positive values summing to one.")
    if n_candidates < 1 or fraction_tolerance < 0 or min(min_real, min_decoys, min_groups) < 0:
        raise ValueError("Invalid split search or minimum-size settings.")
    membership = grouped["membership"].copy()
    groups = sorted(membership.similarity_group.unique())
    if len(groups) < 3 * min_groups:
        raise ValueError(f"Only {len(groups)} components; at least {3 * min_groups} are required.")
    group_index = {group: i for i, group in enumerate(groups)}
    counts = pd.crosstab(membership.similarity_group, membership.kind).reindex(
        index=groups, columns=["real", "shuffle", "random"], fill_value=0).to_numpy()
    total = counts.sum(axis=0)
    if total[0] < 3 * min_real or total[1:].sum() < 3 * min_decoys:
        raise ValueError("Not enough real observations or decoys for the requested split minima.")

    receptor_matrix = np.zeros((len(groups), len(groups)))
    peptide_matrix = np.zeros_like(receptor_matrix)
    target_groups = membership.drop_duplicates("target_id").set_index("target_id").similarity_group.map(group_index)
    peptide_groups = membership.drop_duplicates("peptide_id").set_index("peptide_id").similarity_group.map(group_index)
    hits = grouped["receptor_hits"]
    coverage = grouped["metadata"]["receptor_coverage"]
    hits = hits.loc[(hits.query_coverage.ge(coverage) | hits.target_coverage.ge(coverage))
                    & hits.evalue.le(grouped["metadata"]["receptor_evalue"])]
    for frame, lookup, matrix, score_column in [
        (hits, target_groups, receptor_matrix, "identity"),
        (grouped["peptide_hits"], peptide_groups, peptide_matrix, "similarity"),
    ]:
        if frame.empty:
            continue
        left = frame["query"].map(lookup).to_numpy(dtype=int)
        right = frame["target"].map(lookup).to_numpy(dtype=int)
        values = frame[score_column].to_numpy(dtype=float)
        np.maximum.at(matrix, (left, right), values)
        np.maximum.at(matrix, (right, left), values)
    np.fill_diagonal(receptor_matrix, 0)
    np.fill_diagonal(peptide_matrix, 0)

    random = np.random.default_rng(seed)
    audit_rows, best_key, best_assignment = [], None, None
    split_names = ("train", "val", "test")
    for candidate in range(n_candidates):
        assignment = random.choice(3, size=len(groups), p=fractions)
        allocated = np.array([counts[assignment == i].sum(axis=0) for i in range(3)])
        group_counts = np.bincount(assignment, minlength=3)
        observed = allocated[:, total > 0] / total[total > 0]
        allocation_error = np.abs(observed - np.asarray(fractions)[:, None])
        admissible = (np.all(allocation_error <= fraction_tolerance + 1e-12)
                      and np.all(allocated[:, 0] >= min_real)
                      and np.all(allocated[:, 1:].sum(axis=1) >= min_decoys)
                      and np.all(group_counts >= min_groups))
        entry = {"candidate": candidate, "admissible": bool(admissible),
                 "max_fraction_error": float(allocation_error.max())}
        for i, split in enumerate(split_names):
            entry[f"{split}_groups"] = int(group_counts[i])
            for j, kind in enumerate(["real", "shuffle", "random"]):
                entry[f"{split}_{kind}"] = int(allocated[i, j])
        if admissible:
            test, development = np.where(assignment == 2)[0], np.where(assignment != 2)[0]
            receptor_max = float(receptor_matrix[np.ix_(test, development)].max())
            peptide_max = float(peptide_matrix[np.ix_(test, development)].max())
            normalized = (receptor_max / grouped["metadata"]["receptor_identity"],
                          peptide_max / grouped["metadata"]["peptide_similarity"])
            key = (max(normalized), sum(normalized), float(allocation_error.sum()), candidate)
            entry.update(test_dev_max_detected_receptor_identity=receptor_max,
                         test_dev_max_peptide_similarity=peptide_max,
                         normalized_max_similarity=max(normalized))
            if best_key is None or key < best_key:
                best_key, best_assignment = key, assignment.copy()
        audit_rows.append(entry)
    audit = pd.DataFrame(audit_rows)
    if best_assignment is None:
        largest = pd.DataFrame(counts, index=groups, columns=["real", "shuffle", "random"])
        largest["total"] = largest.sum(axis=1)
        raise ValueError(
            f"No admissible split in {n_candidates} fixed candidates (seed={seed}). "
            "Similarity thresholds and sample-size criteria were not relaxed. "
            f"Largest components:\n{largest.nlargest(10, 'total').to_string()}"
        )
    split_map = {group: split_names[best_assignment[i]] for i, group in enumerate(groups)}
    membership["split"] = membership.similarity_group.map(split_map)
    audit["selected"] = audit.candidate.eq(best_key[-1])
    edges = grouped["edges"]
    if not edges.empty:
        left = edges.left_complex.map(membership.split)
        right = edges.right_complex.map(membership.split)
        if left.isna().any() or right.isna().any() or not left.eq(right).all():
            raise AssertionError("A qualifying similarity edge crosses the new split.")
    for column in ["similarity_group", "target_id", "peptide_id", "pdb_code"]:
        if not membership.groupby(column).split.nunique().eq(1).all():
            raise AssertionError(f"{column} crosses the new split.")
    return membership, audit
