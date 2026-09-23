# Research source snapshots

These directories preserve experiment implementations and their configuration
history. They are not one interchangeable installation or a set of final results.

- `server/`: source snapshots from the shared research project, including the
  full Terminal-Bench controllers and the current Hot24 merged arm.
- `dependencies/`: the project's backbone training/evaluation and infrastructure
  code that lived outside the main shared project directory.
- `workspace/`: additional preparation, diagnostics, monitoring, migration, and
  experiment code from the working tree and handoff. Byte-identical copies already
  represented by a server file are mapped in `../docs/source_inventory.json`.

Directory dates and attempt suffixes distinguish implementations. A failed or
superseded attempt remains historical code, not an accepted method or permission
to replay results. Do not run a historical submission/recovery script against an
existing experiment. Configure a new isolated deployment and re-run its real
qualification gates.

Models, adapters, datasets, raw agent trajectories, answers, run receipts, and
credentials are not included. External packages such as AppWorld and upstream
RULER/InfLLM are installed separately rather than copied from dependency caches.

See the [experiment index](../docs/EXPERIMENTS.md) and
[reproduction notes](../docs/REPRODUCIBILITY.md).
