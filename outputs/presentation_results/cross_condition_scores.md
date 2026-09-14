# Summary and separate focal interaction

Snapshot started 2026-09-12T20:27:30.468069+00:00. Cohort: finished. Selected runs: 820; statuses: {'completed': 787, 'failed': 33}. Attempted = selected run execution directory exists or terminal result is recorded; completed = status completed (all stages successful). Counts and median output tokens span all four conditions. Performance includes failed terminal runs in the default finished cohort; queued and unfinished runs are excluded. Missing values are unavailable, never zero. F1* uses diagnostic scientific scores consistently across harnesses. Biomni is a local export generated 2026-09-11T03:16:36.523825+00:00. 95% CIs reflect repeated-run uncertainty on this fixed dataset pair: whole-run bootstrap for scores, Wilson for focal recovery. See README.md for methods and rerunning.

| Metric | Llm | Harness | Workflow | Reasoning effort | Runs attempted | Runs successfully completed | Runs ended with errors | Median output tokens per completed run | Value | Ci95 low | Ci95 high | Ci method | Ci status | Ci valid resamples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| summary_score | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 84.4 | 83.7 | 84.9 | joint whole-run bootstrap | available | 10000 |
| label_surprise_interaction | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 0.0 | -39.2 | 39.2 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-6-astra | Codex | sequential | medium | 40 | 39 | 1 | 80,126.0 | 84.5 | 83.5 | 85.5 | joint whole-run bootstrap | available | 10000 |
| label_surprise_interaction | gpt-6-astra | Codex | sequential | medium | 40 | 39 | 1 | 80,126.0 | 0.0 | -39.2 | 39.2 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-6-astra | Codex | deliberative | medium | 40 | 40 | 0 | 255,276.0 | 85.2 | 84.9 | 85.5 | joint whole-run bootstrap | available | 10000 |
| label_surprise_interaction | gpt-6-astra | Codex | deliberative | medium | 40 | 40 | 0 | 255,276.0 | 0.0 | -39.2 | 39.2 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 80.9 | 77.9 | 82.8 | joint whole-run bootstrap | available | 10000 |
| label_surprise_interaction | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 10.0 | -30.1 | 51.2 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-5.6-sol | Codex | sequential | medium | 40 | 40 | 0 | 176,867.5 | 78.2 | 73.2 | 81.9 | joint whole-run bootstrap | available | 10000 |
| label_surprise_interaction | gpt-5.6-sol | Codex | sequential | medium | 40 | 40 | 0 | 176,867.5 | 10.0 | -33.6 | 52.4 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-5.6-sol | Codex | deliberative | medium | 40 | 38 | 2 | 547,345.5 | 71.2 | 64.4 | 77.0 | joint whole-run bootstrap | available | 10000 |
| label_surprise_interaction | gpt-5.6-sol | Codex | deliberative | medium | 40 | 38 | 2 | 547,345.5 | 40.0 | -10.6 | 78.9 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 77.5 | 73.2 | 80.9 | joint whole-run bootstrap | available | 9897 |
| label_surprise_interaction | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 10.0 | -37.1 | 52.7 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-5.6-terra | Codex | sequential | medium | 40 | 39 | 1 | 58,512.0 | 78.9 | 75.5 | 81.7 | joint whole-run bootstrap | available | 9992 |
| label_surprise_interaction | gpt-5.6-terra | Codex | sequential | medium | 40 | 39 | 1 | 58,512.0 | 30.0 | -13.7 | 71.1 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-5.6-terra | Codex | deliberative | medium | 40 | 39 | 1 | 160,931.0 | 70.7 | 65.6 | 75.3 | joint whole-run bootstrap | available | 9704 |
| label_surprise_interaction | gpt-5.6-terra | Codex | deliberative | medium | 40 | 39 | 1 | 160,931.0 | 80.0 | 27.8 | 114.6 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 46.6 | 38.9 | 53.1 | joint whole-run bootstrap | available | 10000 |
| label_surprise_interaction | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 10.0 | -40.3 | 60.1 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-5.6-luna | Codex | sequential | medium | 40 | 39 | 1 | 79,146.0 | 38.9 | 33.9 | 43.8 | joint whole-run bootstrap | available | 9989 |
| label_surprise_interaction | gpt-5.6-luna | Codex | sequential | medium | 40 | 39 | 1 | 79,146.0 | -10.0 | -53.2 | 39.2 | Wilson-based MOVER | available | 0 |
| summary_score | gpt-5.6-luna | Codex | deliberative | medium | 40 | 38 | 2 | 237,172.5 | 39.8 | 34.4 | 45.2 | joint whole-run bootstrap | available | 9987 |
| label_surprise_interaction | gpt-5.6-luna | Codex | deliberative | medium | 40 | 38 | 2 | 237,172.5 | 0.0 | -44.3 | 47.5 | Wilson-based MOVER | available | 0 |
| summary_score | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 79.8 | 78.5 | 81.1 | joint whole-run bootstrap | available | 9981 |
| label_surprise_interaction | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 0.0 | -61.4 | 61.4 | Wilson-based MOVER | available | 0 |
| summary_score | claude-opus-5 | Custom runner | sequential | medium | 20 | 19 | 1 | 309,733.0 | 79.5 | 78.3 | 80.8 | joint whole-run bootstrap | available | 9996 |
| label_surprise_interaction | claude-opus-5 | Custom runner | sequential | medium | 20 | 19 | 1 | 309,733.0 | 0.0 | -61.4 | 61.4 | Wilson-based MOVER | available | 0 |
| summary_score | claude-opus-5 | Custom runner | deliberative | medium | 20 | 18 | 2 | 1,222,954.0 | 80.1 | 79.0 | 81.1 | joint whole-run bootstrap | available | 9887 |
| label_surprise_interaction | claude-opus-5 | Custom runner | deliberative | medium | 20 | 18 | 2 | 1,222,954.0 | 0.0 | -61.4 | 61.4 | Wilson-based MOVER | available | 0 |
| summary_score | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 76.1 | 71.7 | 78.7 | joint whole-run bootstrap | available | 9655 |
| label_surprise_interaction | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 10.0 | -30.1 | 51.2 | Wilson-based MOVER | available | 0 |
| summary_score | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 72.3 | — | — | joint whole-run bootstrap | insufficient eligible runs | 6388 |
| label_surprise_interaction | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 40.0 | -4.2 | 83.8 | Wilson-based MOVER | available | 0 |
| summary_score | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | sequential | medium | 40 | 40 | 0 | 265,093.5 | 71.4 | — | — | joint whole-run bootstrap | too many bootstrap draws lack eligible evidence | 8804 |
| label_surprise_interaction | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | sequential | medium | 40 | 40 | 0 | 265,093.5 | 30.0 | -12.6 | 73.4 | Wilson-based MOVER | available | 0 |
| summary_score | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | deliberative | medium | 40 | 40 | 0 | 822,696.5 | 68.3 | — | — | joint whole-run bootstrap | too many bootstrap draws lack eligible evidence | 7915 |
| label_surprise_interaction | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | deliberative | medium | 40 | 40 | 0 | 822,696.5 | 40.0 | -7.6 | 84.1 | Wilson-based MOVER | available | 0 |
| summary_score | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 70.6 | 66.9 | 74.0 | joint whole-run bootstrap | available | 9753 |
| label_surprise_interaction | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 50.0 | 2.7 | 88.3 | Wilson-based MOVER | available | 0 |
| summary_score | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | sequential | medium | 40 | 40 | 0 | 310,695.5 | 72.0 | 67.4 | 75.5 | joint whole-run bootstrap | available | 9991 |
| label_surprise_interaction | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | sequential | medium | 40 | 40 | 0 | 310,695.5 | 20.0 | -25.4 | 61.9 | Wilson-based MOVER | available | 0 |
| summary_score | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | deliberative | medium | 40 | 33 | 7 | 810,756.0 | 75.4 | 72.5 | 77.6 | joint whole-run bootstrap | available | 9937 |
| label_surprise_interaction | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | deliberative | medium | 40 | 33 | 7 | 810,756.0 | 20.0 | -21.8 | 61.6 | Wilson-based MOVER | available | 0 |
