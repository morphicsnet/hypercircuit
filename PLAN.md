# Hypercircuit local checklist

Source of truth: [`/Volumes/128/MAIR/PLAN.md`](/Volumes/128/MAIR/PLAN.md)

## Hypercircuit-owned surfaces
- `src/hypercircuit/blt_io/adapter.py`
- `src/hypercircuit/cli/run_blt_analysis.py`
- `tests/test_blt_analysis.py`

## Current blockers
- stop consuming BLT and MAIR through absolute repo paths
- keep the downstream analysis contract stable while BLT capture changes upstream

## Upstream dependency
- `BLT` and `MAIR` must be installed editable before BLT ingest tests run
