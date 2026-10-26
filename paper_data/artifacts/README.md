# Paper inputs

These are the original JSON/JSONL records from the experiment cluster.
There are 300 runs in each dataset: 120 clean and 180 noisy runs, with
117 measurements per run. The 1202 files retain their original bytes.

- `exp_b/`: original trajectories; input for `tab_alerts.tex`.
- `exp_b_rerun/`: repeated runs with gradient norms, validation metrics,
  and timing records; input for the baseline and overhead tables.

Do not mix the two datasets. The original alert table has 16/120 clean
false alarms; the rerun has 22/120 under the same calibration procedure.
The rerun is not claimed to be numerically identical to the original.

Checkpoints, CIFAR downloads, logs, and temporary files are not included.
Historical paths inside the final manifests describe the source run;
those checkpoint paths are not required for the table calculations.

Verify the files with `make data-check`. Rebuild the tables with
`make tables` and compare their values with `make paper-check`.
`SHA256SUMS` lists hashes relative to this directory.
