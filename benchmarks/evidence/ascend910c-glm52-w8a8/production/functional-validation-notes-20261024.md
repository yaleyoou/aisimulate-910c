# Functional delivery validation

The operator-level feature is usable within the published EP16 request/search
presets. Full serving accuracy, EP32 small-batch coverage and new topologies
remain separate work. The formal direct-FPM profile is still unaccepted;
checkpoint revision53 records that workflow separately from the delivered feature.

## Runtime and actual model

`functional-delivery-20261024/summary.json` records the image, wheel/native
checksums, actual model metadata hashes and validation counts. Its manifest
seals103 output/log files. The installed wheel runs without the source tree on
Python's import path; standalone package and engine verification passes.

With `/mnt/raid/user/data/models/glm-q`, all60 finite random-search candidates
complete and match the staged metadata run in every compared metric, maximum
difference0. The saved winning YAML reproduces its metrics. Bayesian search
issues12 suggestions, evaluates11 successfully and reuses one cached result.
These are functional checks and prediction reproducibility, not serving measurements.

## Legacy replay preservation

Baseline `a93487f9938eed211449c6d0a35661dbd711c9a5` is an ancestor of candidate
`f90015dfbdc8e2a66754b22eadd9d396e1e01fb9`. Both use rustc1.99.1,
release builds and an empty explicit feature set. The same skill runner source
and refreshed offline dependency lock are used for both; the original skill
lock cannot build either revision with `--locked`. Exact lockfiles, compiler,
runner/binary checksums, frozen configs and observations are archived in
`replay-validation-20261024/`.

The historical skill golden points target an older revision. They are first
requalified on this baseline, adjusting only capacities where necessary to
exercise one to three preemptions/retractions. The final capacities, workload,
canonical exclusions and all other configuration bytes are identical for the
compared revisions. No new report exclusions are introduced.

All eight vLLM/SGLang/TRT-LLM rows complete5000 ordered Mooncake requests with
full output, reuse, engine waiting and applicable handoffs/pressure recovery.
Each baseline and candidate runs twice in separate processes on the same
physical CPU assigned to its row; all four canonical digests agree per row.
This is semantic byte parity, not a loop-performance benchmark.

GNU `size` reports text3057660/data74968/bss384 bytes for the baseline runner,
and text3066904/data75064/bss384 for the candidate. Build records also retain
the complete binary sizes and checksums.

## Cost boundary

The new PD table loads once per prepared database and retains at most32 points
across eight mappings. Canonical source/destination validation prepares an
immutable shared curve once before replay. Each measured handoff performs one
page rounding and bounded lookup over at most five points. Requests share the
curve without rereading tables or rebuilding models. Non-Ascend databases do
not load this table; the bandwidth path keeps its previous calculation.

CLI wall-clock startup, data/model preparation, virtual-time replay wall time,
predicted serving latency and measured operator latency are distinct quantities.
The archived reports must not be interpreted as real910C service throughput or
as a full host-performance qualification campaign.
