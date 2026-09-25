# Operational concurrency amendment — September 23, 2026

The user requested higher concurrency after observing server headroom, then explicitly confirmed 40. The runtime limit is now **40 concurrent runs**, increased from 10 to 20 and then 40, for persistent and subsequent phases. Each phase contains exactly 40 assigned runs, so phases remain separate. The original frozen scientific configurations still record the original scheduling value; their fingerprints, all scientific settings, source, data, and scoring remain unchanged.

Before the change, vLLM reported ten running requests, no waiting requests, no preemptions, and 34.8% KV-cache use. The original coordinator had a fixed worker count, so its verified process (PID 1730612) was stopped and the endpoint was checked for zero active requests before resuming.

All **190 completed call records** were preserved with SHA-256 hashes. Ten in-flight requests lacked completed response records and are eligible for reissue during exact journal replay. Their partial generation usage is unavailable and must be disclosed separately in final resource accounting.

- [Interruption audit](persistent/concurrency_changes/20260923T135301Z/interruption.json)
- [Initial replacement launch at 20](persistent/launch_parallel20.json)
- [Current replacement launch at 40](persistent/launch_parallel40.json)
- [Second interruption audit](persistent/concurrency_changes/20260923T140149Z/interruption.json)
- [Current driver state](persistent/execution.json)
- [Operational runner](run_concurrent.py)

At the second increase, vLLM reported 20 running requests, no queue or preemptions, and 46.2% KV-cache use. All 205 completed call records were preserved and 20 in-flight requests were interrupted for exact replay. Across both changes, partial token usage is unavailable for 30 interrupted request attempts. The first operational runner is archived by SHA-256 under `operational_source_archive/`.

Use `/home/klkehl/thisenv/bin/python -u run_concurrent.py --workflow PHASE --parallel 40`, adding `--resume` only for a previously launched phase after verifying its driver is no longer running. The wrapper verifies the original frozen manifest, sets only the manager's runtime worker limit, and delegates to the original manager. Both the shared phase lock and prior-phase reporting gate remain active.

The existing hourly follow-up has been updated to use this entry point for recovery and future phases. Monitor server waiting requests, KV-cache use, and preemptions before further increases.

## Server restart and authorized resume

The user paused the grid at 2026-09-23 22:06 UTC to restart vLLM with more GPU memory and changed offloading settings. The driver and heartbeat were stopped. All 1,638 recorded responses were preserved and verified by SHA-256; 40 unfinished requests were recorded separately in `persistent/pauses/20260923T220636Z/interruption.json`. Their partial usage is unavailable, in addition to the 30 requests interrupted by earlier concurrency changes and the separately recorded API timeout.

The user subsequently authorized resuming at concurrency 40 and increasing it if there is headroom. The externally reachable endpoint remains **http://sn4622130540:8001/v1**: the user's initial reference to port 8000 was corrected as the internal Docker port. Original endpoint, source, configurations, prompts, sampling, and replay fingerprints remain unchanged. Restart verification, preflight, and metrics are recorded under `persistent/pauses/20260923T220636Z/resume/`.

Each phase has only 40 runs, and the frozen coordinator makes model calls in order within each run (including deliberative peers). Thus 40 concurrent workers already schedules every independent run; a larger worker limit cannot add requests within this fixed grid. Keep all available runs active, monitor headroom, and preserve the required phase boundaries and within-run dependencies.
