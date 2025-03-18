# Scoring benchmark

[Link to nice Nature paper that does the same thing.](https://www.nature.com/articles/s43016-024-01089-5#MOESM1)

## TODO: @Malcolm

- Clean the script sand combine scripts nicely. Rename things to make the workflow neater etc.
  - Reduce intermediary outputs to a minimum. Only keep what is necessary.
  - Name variables consistently. Including column names. Two columns with the same thing should never have different names.
  - If there are several steps, prefix files with a step index. E.g, `0_*.csv` `1_*.csv` etc. Where 0 is input to produce 1.
- Create a "run script" that runs everything.
- Delete all old files.
- Re-run everything.
- Work on the script `compute_score.py`.
  - The script should take the final output and return linear regression weights.
  - It should also compute metrics.
  - It should also produce plots.

Lastly we'll want to populate this README to explain what we have done.
