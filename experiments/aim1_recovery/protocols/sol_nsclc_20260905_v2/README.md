# Sol 5.6 medium NSCLC comparison

User-authorized fresh batch: 20 named and 20 masked NSCLC runs, each with the
existing 25-iteration v2 research protocol. A separate named/masked setup pair
is excluded from formal scoring.

Data, task instructions, descriptions, and schemas are byte-identical to the
corresponding reconstructed Luna v2 NSCLC inputs. Only model labels and job
identifiers change in public task inputs. The same archived 40,000 discovery
and 10,000 held-out rows, four-condition recovery definition, and deterministic
scoring rules are retained.

Each research session is a fresh Sol 5.6 agent with medium reasoning and no
inherited conversation. Work advertises priority service; per-response tier
and token telemetry are unavailable. The coordinator previously inspected the
Luna results and answer key; these are excluded from research-agent context.
No scientific recovery scores are examined during formal dispatch.

All original attempts are retained. Checkpoints are packed only when workers
are idle. A technical interruption resumes the original work rather than
selecting a fresh attempt based on its scientific outcome.

The new prepare_model_subset.py helper copies only frozen inputs from a
restored source plan; it never copies prior research records into a fresh task.
