# Immunomodulating peptides

The aim of this project is to get an understanding of the peptides that exist in infected/inflamed conditions and how these interact with immunomodulatory proteins.

To do so we utilize newly developed protein-peptide interaction mining and public data.

The project goes as follows:

1. Curate a dataset of peptidomics datasets. 
    - Combine all datasets and make them comparable in a nice format.
2. Curate a dataset of immunomodulatory proteins.
3. Utilize BoPep (public repo) to dock peptides to proteins.
4. Analyze output.

## Data

All data is in `/srv/data1/general/immunopeptides_data`

## Target proteins: 
All proteins leading in to:
-	TLR cascades
-	Complement cascade
-	Neutrophil degranulation
-	Interferon Signaling
-	Signaling by Interleukins
-	TNFR2 non canonical NFkB…

See `/src/get_proteins`.

## Benchmark data

See `/src/scoring_benchmark`



### Peptide repos

### Target structures

### Scoring functions

Possibly relevant scoring functions:
- pDockQ 
- GDockScore

Bopep scoring functions:
- Interface_dG,
- rosetta_score,
- interface_delta_hbond_unsat,
- packstat

DONE:
- [x] Curate subset of protein-peptide interactions (see **get_protein_peptide.py**)

TODO: 
- [ ] Find appropriate scoring functions: should be implementable (Should they be recent?)
- [ ] Determine functions that are well performing on boolean tasks (e.g. whether a peptide binds or not)
- [ ] Determine functions that are well performing on regression tasks (e.g. how strong a peptide binds - affinity)
