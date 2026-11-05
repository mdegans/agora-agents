# Claude Haiku 5.5 probe run, 2026-11-05

The analysis is in `../../analyses/haiku_5_5_2026-11-05.md`.

The contents:
- `ratings/<arm>/{direct,indirect}_repNN.json`: the full `probe_distribution` output of each rep, with every answer, its token counts and its thinking summary when there was one. The `.txt` files hold the printed summary.
- `open_ended/<arm>/{direct,indirect}.{json,md}`: one pass at the `low` and `high` settings.
- `summary.json` and `table.md`: per-item counts, means and modes, set beside the legacy baselines (`scripts/aggregate.py`).
- `scripts/`: the scripts as run (`run_arm.sh`, `run_open.sh`, `cost.py`).

The arms:
- `off`: thinking disabled.
- `low`: adaptive thinking at effort `low`, display summarized.
- `high`: adaptive thinking at effort `high`, display summarized.
