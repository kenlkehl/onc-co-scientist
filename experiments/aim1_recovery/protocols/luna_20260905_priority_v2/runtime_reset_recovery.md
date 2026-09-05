# Runtime reset and reconstruction, September 5, 2026

The last saved pre-reset checkpoint (version 10) contained 187 validated completed
runs and six dispatched jobs (0188–0193). Subsequent conversation updates reported
at least 197 completions and an unidentified integrity failure, but those later
records are unavailable. The reconstructed batch must not be described as an
exact recovery of all original attempts. Jobs 0188–0203 form a conservative range
potentially affected by the lost work; report sensitivity excluding this range.
The previously reported failed attempt cannot be reliably mapped to a replicate.

Current origin main is 1b9f904, including vLLM transport failover. All seven
recorded launch implementation hashes match that main commit. Recovery rules,
input data, prompts, model (Luna 5.6), medium reasoning and advertised priority
service are unchanged. Held-out recovery remains unopened during reconstruction.

All 250 restored public input hashes and all 187 completed transcripts passed
validation. The initial partial-record validation incorrectly checked only retained
records, overlooking surplus event receipts in the non-atomic checkpoint:

- job_0188: one record, four event receipts.
- job_0189: seven records, ten event receipts.

Replacement sessions were launched before that mismatch was detected. Job_0189
subsequently failed receipt sequence validation. Job_0188's agent reported that it
deduplicated its event log to finalize, despite instructions to preserve submitted
records. Both attempts and all subsequent outputs are retained, but both receive
zero recovery credit as terminal technical failures, with no replacement attempt.
The untouched checkpoint versions of both workspaces are retained under
recovery_evidence/untouched_checkpoint in the experiment archive. This failure
classification was made before recovery scoring, not selected using outcomes.

Future checkpoints are taken only between groups, after all active workers have
finished. The first such checkpoint has 198 validated completions and two failures
(200 terminal attempts). Both excluded setup rounds and all older experiment data
remain preserved. The remaining formal jobs continue with fresh, isolated-context
Luna medium sessions and full 25 clinical / 10 DepMap iteration budgets.

The reporting change reads an explicit terminal_failures.json ledger, validates
its job identities and zero-credit disposition, and retains these jobs in every
primary denominator without evaluating their claims. Valid-run scoring is unchanged.
The complete restored-batch report must disclose reconstruction uncertainty and
must not claim the unidentified pre-reset failure has been precisely recovered.
