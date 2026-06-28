# Pull Request

## What

<!-- One sentence: what does this PR change? -->

## Why

<!-- What problem does it solve? Link the audit finding, issue, or vendor doc that motivated it. -->

## How

<!-- Highest-level summary of the approach. Algorithm choices, data flow, anything a reviewer needs to understand the diff. -->

## Strategy PRs only — config justification

If this PR adds a new strategy or changes a vendor SDK config, fill in:

- **Strategy name:** `<name>`
- **Vendor / SDK version:** `<sdk==version>`
- **Why this config:** <!-- link to vendor docs page that recommends this default -->
- **Expected accuracy delta:** vs the prior run on the same `--seed N`. Attach the diff between `results/longmemeval-s_<strategy>_summary.json` before and after.
- **Reproduction recipe:** the exact `memory-arena benchmark` command this was tested with.

## Tests

- [ ] `pytest tests/ -x` passes locally
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] If this PR touches a strategy, I added/updated a test under `tests/strategies/`
- [ ] If this PR changes evaluator behavior, the structural checks still pass with the existing fixtures

## Data integrity (required for any PR that produces or consumes result JSONs)

- [ ] Every accuracy / cost / latency claim in this PR has a result JSON in `results/` with the commit SHA stamped in `metadata.commit_sha` matching this PR's HEAD
- [ ] No headline number is from a single seed — at minimum `--seed 0,1,2`
- [ ] If a vendor SDK was changed, the `status` field is not `config-failed-at-default` for the new config
- [ ] No `recall_at_k_measurable=False` strategy is being labeled with a Recall@k score in the table

## Reviewer focus

<!-- Where should the reviewer look first? Files to skim vs files to read carefully. -->
