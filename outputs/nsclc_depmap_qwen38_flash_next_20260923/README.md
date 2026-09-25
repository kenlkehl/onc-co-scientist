# NSCLC DepMap: Qwen3.8 Flash Next

Fresh grid: **120 runs** across three workflows, expected/surprising versions, masked/unmasked semantics, and ten repeats per cell. **25 iterations per run**.

Persistent started first. Sequential and deliberative each require the previous phase’s reporting receipt. An hourly follow-up in the originating conversation checks progress, reports phase results, and starts the next authorized phase.

- [Protocol](PROTOCOL.md)
- [Concurrency increased to 40](CONCURRENCY.md)
- [Sampling settings](sampling.json)
- [Preflight and configuration verification](verification.json)
- [Data masking audit](masking_audit.json)
- [Frozen provenance](frozen_manifest.json)
- [Persistent progress](persistent/progress.json) · [Driver log](persistent/driver.log)
- [Persistent results, when ready](persistent/RESULTS.md)
- [Sequential results, when ready](sequential/RESULTS.md)
- [Deliberative results, when ready](deliberative/RESULTS.md)

The served model is `Inferact/Qwen3.8-Flash-Next-NVFP4` at `http://sn4622130540:8001/v1`. Requests use Qwen thinking sampling and `reasoning_effort=xhigh`, including retries. Source and inputs are isolated under this directory; older experiments are untouched.

For operational recovery, run `manage.py status --workflow persistent` using `/home/klkehl/thisenv/bin/python`. A stopped driver supports `run_concurrent.py --workflow persistent --parallel 40 --resume` after verifying that its original process is gone. Calls are replayed from durable records; never launch a duplicate driver. Network access to the model host is required.
