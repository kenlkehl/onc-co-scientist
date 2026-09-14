# Clinical experiment results

Snapshot started 2026-09-12T20:27:30.468069+00:00. Cohort: finished. Selected runs: 820; statuses: {'completed': 787, 'failed': 33}. Attempted = selected run execution directory exists or terminal result is recorded; completed = status completed (all stages successful). Counts and median output tokens span all four conditions. Performance includes failed terminal runs in the default finished cohort; queued and unfinished runs are excluded. Missing values are unavailable, never zero. F1* uses diagnostic scientific scores consistently across harnesses. Biomni is a local export generated 2026-09-11T03:16:36.523825+00:00. 95% CIs reflect repeated-run uncertainty on this fixed dataset pair: whole-run bootstrap for scores, Wilson for focal recovery. See README.md for methods and rerunning.

| Metric | Llm | Harness | Workflow | Reasoning effort | Runs attempted | Runs successfully completed | Runs ended with errors | Median output tokens per completed run | Expected-unmasked | Expected-unmasked ci95 low | Expected-unmasked ci95 high | Expected-masked | Expected-masked ci95 low | Expected-masked ci95 high | Surprising-unmasked | Surprising-unmasked ci95 low | Surprising-unmasked ci95 high | Surprising-masked | Surprising-masked ci95 low | Surprising-masked ci95 high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Precision (%) | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 97.7 | 95.1 | 99.7 | 93.1 | 90.0 | 96.2 | 93.5 | 89.5 | 97.1 | 93.5 | 89.7 | 97.2 |
| Precision (%) | gpt-6-astra | Codex | sequential | medium | 40 | 39 | 1 | 80,126.0 | 97.5 | 95.6 | 99.0 | 94.8 | 92.2 | 96.9 | 95.9 | 93.5 | 98.2 | 88.7 | 83.1 | 94.1 |
| Precision (%) | gpt-6-astra | Codex | deliberative | medium | 40 | 40 | 0 | 255,276.0 | 99.0 | 98.1 | 99.8 | 94.8 | 92.0 | 97.4 | 99.3 | 98.6 | 100.0 | 93.8 | 88.7 | 98.3 |
| Precision (%) | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 95.7 | 94.6 | 96.8 | 96.0 | 94.4 | 97.7 | 91.4 | 87.1 | 95.3 | 93.2 | 87.9 | 97.0 |
| Precision (%) | gpt-5.6-sol | Codex | sequential | medium | 40 | 40 | 0 | 176,867.5 | 93.4 | 91.6 | 95.0 | 94.9 | 92.1 | 97.0 | 90.2 | 86.5 | 93.5 | 88.7 | 82.6 | 93.8 |
| Precision (%) | gpt-5.6-sol | Codex | deliberative | medium | 40 | 38 | 2 | 547,345.5 | 93.1 | 89.5 | 96.2 | 93.8 | 90.5 | 96.5 | 89.1 | 85.1 | 92.7 | 92.2 | 86.0 | 97.2 |
| Precision (%) | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 97.6 | 95.6 | 99.4 | 98.4 | 97.5 | 99.3 | 97.0 | 94.8 | 98.8 | 97.8 | 93.7 | 100.0 |
| Precision (%) | gpt-5.6-terra | Codex | sequential | medium | 40 | 39 | 1 | 58,512.0 | 93.5 | 90.8 | 96.4 | 99.1 | 97.2 | 100.0 | 94.4 | 91.2 | 97.1 | 97.7 | 94.5 | 100.0 |
| Precision (%) | gpt-5.6-terra | Codex | deliberative | medium | 40 | 39 | 1 | 160,931.0 | 95.7 | 92.8 | 98.3 | 99.8 | 99.3 | 100.0 | 91.1 | 78.8 | 98.6 | 91.6 | 80.0 | 100.0 |
| Precision (%) | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 59.5 | 36.9 | 78.5 | 91.9 | 81.9 | 98.2 | 72.2 | 64.1 | 80.2 | 85.2 | 73.7 | 95.8 |
| Precision (%) | gpt-5.6-luna | Codex | sequential | medium | 40 | 39 | 1 | 79,146.0 | 76.2 | 62.9 | 86.8 | 89.3 | 83.1 | 95.1 | 58.2 | 40.9 | 72.5 | 80.8 | 61.8 | 95.9 |
| Precision (%) | gpt-5.6-luna | Codex | deliberative | medium | 40 | 38 | 2 | 237,172.5 | 60.9 | 41.2 | 78.8 | 90.5 | 84.0 | 96.1 | 67.7 | 56.2 | 78.1 | 90.0 | 82.4 | 96.7 |
| Precision (%) | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 92.5 | 86.2 | 98.4 | 93.9 | 89.4 | 98.4 | 93.5 | 89.8 | 97.2 | 91.9 | 82.4 | 98.3 |
| Precision (%) | claude-opus-5 | Custom runner | sequential | medium | 20 | 19 | 1 | 309,733.0 | 98.0 | 95.8 | 99.8 | 93.8 | 91.1 | 96.5 | 94.4 | 90.6 | 97.8 | 92.9 | 86.4 | 96.8 |
| Precision (%) | claude-opus-5 | Custom runner | deliberative | medium | 20 | 18 | 2 | 1,222,954.0 | 97.2 | 95.4 | 98.9 | 96.5 | 94.6 | 98.5 | 93.3 | 86.5 | 97.7 | 94.6 | 90.0 | 98.4 |
| Precision (%) | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 97.6 | 92.9 | 100.0 | 97.4 | 93.5 | 100.0 | 95.5 | 89.0 | 100.0 | 99.5 | 98.5 | 100.0 |
| Precision (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 100.0 | 100.0 | 100.0 | 96.0 | 90.4 | 100.0 | 95.0 | 85.0 | 100.0 | 89.7 | 79.2 | 98.3 |
| Precision (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | sequential | medium | 40 | 40 | 0 | 265,093.5 | 100.0 | 100.0 | 100.0 | 91.8 | 76.8 | 100.0 | 86.0 | 73.0 | 98.0 | 99.3 | 97.9 | 100.0 |
| Precision (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | deliberative | medium | 40 | 40 | 0 | 822,696.5 | 100.0 | 100.0 | 100.0 | 94.9 | 86.9 | 100.0 | 100.0 | 100.0 | 100.0 | 99.5 | 98.5 | 100.0 |
| Precision (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 99.2 | 97.7 | 100.0 | 100.0 | 100.0 | 100.0 | 82.4 | 68.9 | 93.8 | 99.3 | 98.0 | 100.0 |
| Precision (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | sequential | medium | 40 | 40 | 0 | 310,695.5 | 99.1 | 97.3 | 100.0 | 100.0 | 100.0 | 100.0 | 97.9 | 94.3 | 100.0 | 97.4 | 92.1 | 100.0 |
| Precision (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | deliberative | medium | 40 | 33 | 7 | 810,756.0 | 99.4 | 98.2 | 100.0 | 97.9 | 93.7 | 100.0 | 96.3 | 90.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| Recall (%) | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 68.3 | 66.7 | 71.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 68.3 | 66.7 | 71.7 |
| Recall (%) | gpt-6-astra | Codex | sequential | medium | 40 | 39 | 1 | 80,126.0 | 63.3 | 56.7 | 66.7 | 73.3 | 68.3 | 78.3 | 70.0 | 66.7 | 76.7 | 68.3 | 66.7 | 71.7 |
| Recall (%) | gpt-6-astra | Codex | deliberative | medium | 40 | 40 | 0 | 255,276.0 | 68.3 | 66.7 | 71.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 63.3 | 56.7 | 66.7 | 66.7 | 66.7 | 66.7 | 61.7 | 51.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | gpt-5.6-sol | Codex | sequential | medium | 40 | 40 | 0 | 176,867.5 | 54.4 | 41.1 | 66.7 | 66.7 | 66.7 | 66.7 | 58.3 | 46.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | gpt-5.6-sol | Codex | deliberative | medium | 40 | 38 | 2 | 547,345.5 | 48.9 | 34.4 | 63.3 | 66.7 | 66.7 | 66.7 | 38.3 | 23.3 | 53.3 | 66.7 | 66.7 | 66.7 |
| Recall (%) | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 51.1 | 37.8 | 63.3 | 66.7 | 66.7 | 66.7 | 60.0 | 51.7 | 65.0 | 61.7 | 53.3 | 66.7 |
| Recall (%) | gpt-5.6-terra | Codex | sequential | medium | 40 | 39 | 1 | 58,512.0 | 52.2 | 41.1 | 63.3 | 66.7 | 66.7 | 66.7 | 55.0 | 45.0 | 65.0 | 65.0 | 61.7 | 66.7 |
| Recall (%) | gpt-5.6-terra | Codex | deliberative | medium | 40 | 39 | 1 | 160,931.0 | 54.4 | 42.2 | 66.7 | 64.4 | 60.0 | 66.7 | 35.0 | 23.3 | 46.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 13.3 | 4.4 | 22.2 | 37.8 | 23.3 | 52.2 | 21.7 | 11.7 | 31.7 | 41.7 | 30.0 | 53.3 |
| Recall (%) | gpt-5.6-luna | Codex | sequential | medium | 40 | 39 | 1 | 79,146.0 | 4.4 | 0.0 | 8.9 | 51.1 | 41.1 | 60.0 | 6.7 | 0.0 | 13.3 | 36.7 | 23.3 | 50.0 |
| Recall (%) | gpt-5.6-luna | Codex | deliberative | medium | 40 | 38 | 2 | 237,172.5 | 4.4 | 0.0 | 10.0 | 41.1 | 27.8 | 54.4 | 10.0 | 5.0 | 15.0 | 40.0 | 26.7 | 53.3 |
| Recall (%) | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | claude-opus-5 | Custom runner | sequential | medium | 20 | 19 | 1 | 309,733.0 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 63.3 | 56.7 | 66.7 |
| Recall (%) | claude-opus-5 | Custom runner | deliberative | medium | 20 | 18 | 2 | 1,222,954.0 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 63.3 | 56.7 | 66.7 | 66.7 | 66.7 | 66.7 | 56.7 | 41.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 60.0 | 50.0 | 66.7 | 58.9 | 47.8 | 66.7 | 60.0 | 50.0 | 66.7 | 58.3 | 50.0 | 66.7 |
| Recall (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | sequential | medium | 40 | 40 | 0 | 265,093.5 | 46.7 | 36.7 | 56.7 | 60.0 | 50.0 | 66.7 | 53.3 | 45.0 | 60.0 | 61.7 | 53.3 | 66.7 |
| Recall (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | deliberative | medium | 40 | 40 | 0 | 822,696.5 | 53.3 | 43.3 | 63.3 | 50.0 | 35.6 | 63.3 | 56.7 | 46.7 | 66.7 | 51.7 | 40.0 | 63.3 |
| Recall (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 38.9 | 31.1 | 48.9 | 65.6 | 63.3 | 66.7 | 50.0 | 40.0 | 60.0 | 66.7 | 66.7 | 66.7 |
| Recall (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | sequential | medium | 40 | 40 | 0 | 310,695.5 | 42.2 | 30.0 | 54.4 | 66.7 | 66.7 | 66.7 | 53.3 | 41.7 | 63.3 | 66.7 | 66.7 | 66.7 |
| Recall (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | deliberative | medium | 40 | 33 | 7 | 810,756.0 | 56.7 | 46.7 | 66.7 | 66.7 | 66.7 | 66.7 | 56.7 | 48.3 | 65.0 | 66.7 | 66.7 | 66.7 |
| Scientific F1* | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 80.3 | 78.5 | 83.0 | 77.6 | 76.5 | 78.7 | 77.8 | 76.4 | 79.0 | 78.8 | 76.5 | 81.8 |
| Scientific F1* | gpt-6-astra | Codex | sequential | medium | 40 | 39 | 1 | 80,126.0 | 76.2 | 70.2 | 79.5 | 82.4 | 79.1 | 85.9 | 80.5 | 78.1 | 84.4 | 77.0 | 74.0 | 80.3 |
| Scientific F1* | gpt-6-astra | Codex | deliberative | medium | 40 | 40 | 0 | 255,276.0 | 80.8 | 79.4 | 83.1 | 78.2 | 77.3 | 79.1 | 79.8 | 79.5 | 80.0 | 77.8 | 76.0 | 79.4 |
| Scientific F1* | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 75.6 | 69.7 | 78.8 | 78.7 | 78.1 | 79.2 | 72.4 | 62.0 | 78.4 | 77.6 | 75.6 | 79.0 |
| Scientific F1* | gpt-5.6-sol | Codex | sequential | medium | 40 | 40 | 0 | 176,867.5 | 66.3 | 54.3 | 77.4 | 78.3 | 77.3 | 79.0 | 69.0 | 57.6 | 77.1 | 75.9 | 73.5 | 77.9 |
| Scientific F1* | gpt-5.6-sol | Codex | deliberative | medium | 40 | 38 | 2 | 547,345.5 | 60.2 | 45.4 | 73.2 | 77.9 | 76.7 | 78.9 | 49.6 | 35.1 | 64.4 | 77.2 | 74.8 | 79.1 |
| Scientific F1* | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 64.3 | 49.9 | 76.0 | 79.5 | 79.2 | 79.8 | 73.4 | 66.8 | 78.4 | 75.1 | 68.4 | 80.0 |
| Scientific F1* | gpt-5.6-terra | Codex | sequential | medium | 40 | 39 | 1 | 58,512.0 | 65.5 | 54.9 | 75.8 | 79.7 | 79.0 | 80.0 | 67.6 | 56.5 | 76.3 | 77.9 | 74.9 | 80.0 |
| Scientific F1* | gpt-5.6-terra | Codex | deliberative | medium | 40 | 39 | 1 | 160,931.0 | 66.9 | 52.9 | 78.0 | 78.1 | 74.4 | 80.0 | 48.1 | 33.7 | 61.2 | 76.4 | 71.5 | 80.0 |
| Scientific F1* | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 18.8 | 6.7 | 31.1 | 48.9 | 32.0 | 64.7 | 30.0 | 16.2 | 43.0 | 53.3 | 41.3 | 65.0 |
| Scientific F1* | gpt-5.6-luna | Codex | sequential | medium | 40 | 39 | 1 | 79,146.0 | 6.9 | 0.0 | 13.8 | 64.0 | 54.1 | 73.3 | 9.1 | 0.0 | 18.3 | 48.3 | 32.2 | 63.7 |
| Scientific F1* | gpt-5.6-luna | Codex | deliberative | medium | 40 | 38 | 2 | 237,172.5 | 6.8 | 0.0 | 14.3 | 53.5 | 38.7 | 67.7 | 16.2 | 8.0 | 24.3 | 51.8 | 36.2 | 66.0 |
| Scientific F1* | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 77.4 | 75.1 | 79.5 | 77.9 | 76.3 | 79.5 | 77.8 | 76.5 | 79.1 | 77.1 | 73.4 | 79.4 |
| Scientific F1* | claude-opus-5 | Custom runner | sequential | medium | 20 | 19 | 1 | 309,733.0 | 79.3 | 78.6 | 79.9 | 77.9 | 77.0 | 78.8 | 78.1 | 76.8 | 79.3 | 74.9 | 69.9 | 78.7 |
| Scientific F1* | claude-opus-5 | Custom runner | deliberative | medium | 20 | 18 | 2 | 1,222,954.0 | 79.1 | 78.5 | 79.6 | 78.8 | 78.2 | 79.5 | 77.7 | 75.2 | 79.2 | 78.2 | 76.5 | 79.5 |
| Scientific F1* | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 76.6 | 69.9 | 80.0 | 79.1 | 77.6 | 80.0 | 67.9 | 51.3 | 78.7 | 79.8 | 79.5 | 80.0 |
| Scientific F1* | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 74.0 | 65.0 | 80.0 | 71.6 | 58.5 | 80.0 | 73.0 | 63.0 | 80.0 | 70.6 | 61.1 | 79.2 |
| Scientific F1* | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | sequential | medium | 40 | 40 | 0 | 265,093.5 | 62.0 | 53.0 | 71.0 | 71.6 | 59.2 | 80.0 | 65.4 | 56.2 | 73.3 | 75.5 | 69.5 | 80.0 |
| Scientific F1* | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | deliberative | medium | 40 | 40 | 0 | 822,696.5 | 68.0 | 59.0 | 77.0 | 62.4 | 44.9 | 76.2 | 71.0 | 62.0 | 80.0 | 65.7 | 54.5 | 76.7 |
| Scientific F1* | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 54.5 | 47.2 | 63.6 | 79.1 | 77.4 | 80.0 | 61.1 | 50.3 | 71.9 | 79.8 | 79.3 | 80.0 |
| Scientific F1* | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | sequential | medium | 40 | 40 | 0 | 310,695.5 | 56.0 | 40.9 | 69.2 | 80.0 | 80.0 | 80.0 | 66.8 | 55.5 | 76.5 | 79.0 | 77.0 | 80.0 |
| Scientific F1* | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | deliberative | medium | 40 | 33 | 7 | 810,756.0 | 70.8 | 61.8 | 79.4 | 79.2 | 77.7 | 80.0 | 70.7 | 62.4 | 78.4 | 80.0 | 80.0 | 80.0 |
| Exploration E | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 65.8 | 65.6 | 66.0 | 64.3 | 64.1 | 64.4 | 65.2 | 65.0 | 65.3 | 64.5 | 64.1 | 64.9 |
| Exploration E | gpt-6-astra | Codex | sequential | medium | 40 | 39 | 1 | 80,126.0 | 65.6 | 65.5 | 65.8 | 66.0 | 64.2 | 68.4 | 66.2 | 64.9 | 68.5 | 64.8 | 64.2 | 65.6 |
| Exploration E | gpt-6-astra | Codex | deliberative | medium | 40 | 40 | 0 | 255,276.0 | 66.3 | 65.2 | 68.2 | 64.1 | 63.9 | 64.3 | 64.8 | 64.7 | 65.0 | 64.3 | 63.9 | 64.6 |
| Exploration E | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 66.0 | 65.7 | 66.3 | 61.7 | 59.9 | 63.2 | 60.5 | 50.3 | 65.9 | 62.3 | 61.3 | 63.1 |
| Exploration E | gpt-5.6-sol | Codex | sequential | medium | 40 | 40 | 0 | 176,867.5 | 60.4 | 49.1 | 66.2 | 61.4 | 60.4 | 62.2 | 60.5 | 50.0 | 66.0 | 62.1 | 60.7 | 63.3 |
| Exploration E | gpt-5.6-sol | Codex | deliberative | medium | 40 | 38 | 2 | 547,345.5 | 54.0 | 37.9 | 65.8 | 61.4 | 60.6 | 62.3 | 40.5 | 25.6 | 55.5 | 61.4 | 60.5 | 62.3 |
| Exploration E | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 59.2 | 46.5 | 66.0 | 63.1 | 61.5 | 64.4 | 65.6 | 64.8 | 66.2 | 64.0 | 62.1 | 65.1 |
| Exploration E | gpt-5.6-terra | Codex | sequential | medium | 40 | 39 | 1 | 58,512.0 | 64.9 | 63.9 | 65.9 | 61.2 | 59.1 | 63.1 | 61.5 | 57.8 | 64.7 | 62.8 | 61.6 | 63.8 |
| Exploration E | gpt-5.6-terra | Codex | deliberative | medium | 40 | 39 | 1 | 160,931.0 | 59.7 | 48.5 | 65.8 | 57.4 | 51.1 | 62.1 | 59.6 | 49.4 | 66.5 | 60.9 | 57.1 | 64.1 |
| Exploration E | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 25.3 | 7.1 | 44.0 | 47.1 | 34.1 | 57.7 | 38.1 | 21.3 | 52.9 | 43.7 | 31.7 | 54.9 |
| Exploration E | gpt-5.6-luna | Codex | sequential | medium | 40 | 39 | 1 | 79,146.0 | 18.7 | 5.6 | 32.0 | 52.4 | 47.3 | 56.6 | 29.5 | 14.1 | 44.6 | 44.1 | 35.5 | 52.1 |
| Exploration E | gpt-5.6-luna | Codex | deliberative | medium | 40 | 38 | 2 | 237,172.5 | 17.8 | 4.4 | 32.2 | 36.0 | 24.2 | 47.6 | 18.8 | 7.1 | 31.7 | 48.7 | 41.3 | 55.5 |
| Exploration E | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 64.0 | 61.5 | 65.3 | 61.3 | 59.6 | 62.8 | 63.9 | 61.9 | 65.5 | 63.9 | 63.6 | 64.0 |
| Exploration E | claude-opus-5 | Custom runner | sequential | medium | 20 | 19 | 1 | 309,733.0 | 65.4 | 65.2 | 65.7 | 61.9 | 61.3 | 62.8 | 64.8 | 64.0 | 65.5 | 63.9 | 63.1 | 64.7 |
| Exploration E | claude-opus-5 | Custom runner | deliberative | medium | 20 | 18 | 2 | 1,222,954.0 | 64.2 | 63.2 | 65.2 | 58.0 | 54.8 | 61.3 | 64.7 | 64.7 | 64.7 | 62.0 | 61.1 | 63.2 |
| Exploration E | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 65.9 | 64.7 | 66.7 | 66.7 | 66.7 | 66.7 | 59.4 | 46.1 | 66.5 | 66.7 | 66.7 | 66.7 |
| Exploration E | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 64.8 | 64.4 | 65.2 | 59.4 | 56.0 | 62.1 | 64.7 | 64.1 | 65.1 | 61.1 | 60.3 | 62.0 |
| Exploration E | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | sequential | medium | 40 | 40 | 0 | 265,093.5 | 63.5 | 62.0 | 64.8 | 58.7 | 55.5 | 60.9 | 63.6 | 62.1 | 64.9 | 57.5 | 51.1 | 61.4 |
| Exploration E | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | deliberative | medium | 40 | 40 | 0 | 822,696.5 | 65.0 | 64.2 | 65.8 | 51.3 | 40.8 | 60.7 | 64.9 | 64.4 | 65.5 | 48.0 | 37.3 | 57.6 |
| Exploration E | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 62.5 | 60.1 | 64.4 | 62.3 | 59.1 | 64.2 | 63.5 | 62.5 | 64.3 | 63.9 | 62.7 | 64.9 |
| Exploration E | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | sequential | medium | 40 | 40 | 0 | 310,695.5 | 63.0 | 60.4 | 64.6 | 63.8 | 62.8 | 64.6 | 56.9 | 50.3 | 62.1 | 64.3 | 63.6 | 65.0 |
| Exploration E | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | deliberative | medium | 40 | 33 | 7 | 810,756.0 | 64.9 | 63.5 | 65.8 | 62.8 | 60.2 | 64.4 | 62.2 | 58.8 | 64.7 | 64.5 | 63.5 | 65.1 |
| Evidence responsiveness B | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 100.0 | 100.0 | 100.0 | 94.2 | 89.2 | 98.3 | 87.5 | 80.0 | 94.6 | 95.4 | 91.7 | 98.5 |
| Evidence responsiveness B | gpt-6-astra | Codex | sequential | medium | 40 | 39 | 1 | 80,126.0 | 87.5 | 76.2 | 100.0 | 96.9 | 92.5 | 100.0 | 96.2 | 92.3 | 100.0 | 93.7 | 87.8 | 98.7 |
| Evidence responsiveness B | gpt-6-astra | Codex | deliberative | medium | 40 | 40 | 0 | 255,276.0 | 97.6 | 92.6 | 100.0 | 97.2 | 93.3 | 100.0 | 97.9 | 94.5 | 100.0 | 97.8 | 94.4 | 100.0 |
| Evidence responsiveness B | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 87.0 | 77.8 | 95.2 | 93.1 | 86.4 | 100.0 | 79.8 | 72.6 | 87.5 | 89.2 | 81.7 | 96.7 |
| Evidence responsiveness B | gpt-5.6-sol | Codex | sequential | medium | 40 | 40 | 0 | 176,867.5 | 91.6 | 85.0 | 97.2 | 87.2 | 80.6 | 93.3 | 88.8 | 80.5 | 95.8 | 84.7 | 74.4 | 93.9 |
| Evidence responsiveness B | gpt-5.6-sol | Codex | deliberative | medium | 40 | 38 | 2 | 547,345.5 | 91.7 | 84.8 | 97.5 | 89.4 | 76.7 | 100.0 | 91.4 | 84.4 | 97.2 | 90.9 | 84.5 | 97.2 |
| Evidence responsiveness B | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 86.8 | 78.1 | 95.6 | 87.8 | 68.9 | 100.0 | 83.4 | 71.1 | 95.8 | 92.2 | 85.0 | 98.3 |
| Evidence responsiveness B | gpt-5.6-terra | Codex | sequential | medium | 40 | 39 | 1 | 58,512.0 | 90.6 | 79.8 | 100.0 | 91.1 | 76.7 | 100.0 | 84.0 | 73.6 | 94.4 | 94.4 | 87.9 | 100.0 |
| Evidence responsiveness B | gpt-5.6-terra | Codex | deliberative | medium | 40 | 39 | 1 | 160,931.0 | 80.0 | 66.1 | 93.3 | 92.8 | 80.0 | 100.0 | 89.6 | 79.4 | 97.6 | 91.1 | 85.4 | 96.7 |
| Evidence responsiveness B | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 71.0 | 57.9 | 82.7 | 79.0 | 66.5 | 90.6 | 93.0 | 85.5 | 98.3 | 80.1 | 67.1 | 90.4 |
| Evidence responsiveness B | gpt-5.6-luna | Codex | sequential | medium | 40 | 39 | 1 | 79,146.0 | 78.4 | 65.0 | 89.3 | 76.4 | 60.9 | 91.1 | 78.4 | 70.5 | 86.0 | 75.0 | 63.6 | 84.9 |
| Evidence responsiveness B | gpt-5.6-luna | Codex | deliberative | medium | 40 | 38 | 2 | 237,172.5 | 74.0 | 63.8 | 83.3 | 86.8 | 73.7 | 96.5 | 87.3 | 81.0 | 93.1 | 91.8 | 85.6 | 98.3 |
| Evidence responsiveness B | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 87.5 | 72.2 | 100.0 | 75.0 | 66.7 | 91.7 | 69.4 | 66.7 | 75.0 | 68.7 | 57.3 | 80.0 |
| Evidence responsiveness B | claude-opus-5 | Custom runner | sequential | medium | 20 | 19 | 1 | 309,733.0 | 70.0 | 66.7 | 76.7 | 68.6 | 66.7 | 73.3 | 75.0 | 66.7 | 91.7 | 76.7 | 66.7 | 90.0 |
| Evidence responsiveness B | claude-opus-5 | Custom runner | deliberative | medium | 20 | 18 | 2 | 1,222,954.0 | 74.1 | 66.7 | 88.9 | 73.9 | 66.7 | 82.2 | 73.6 | 68.0 | 79.1 | 81.8 | 69.6 | 93.3 |
| Evidence responsiveness B | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 61.7 | 54.9 | 66.2 | 63.5 | 60.6 | 66.0 | 57.5 | 49.3 | 63.0 | 64.3 | 60.0 | 66.7 |
| Evidence responsiveness B | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 63.3 | — | — | 59.1 | 52.8 | 64.4 | 60.4 | 54.2 | 65.6 | 57.4 | 49.4 | 64.2 |
| Evidence responsiveness B | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | sequential | medium | 40 | 40 | 0 | 265,093.5 | 63.6 | — | — | 60.2 | 53.0 | 65.8 | 55.7 | 45.0 | 64.1 | 64.3 | 61.9 | 66.7 |
| Evidence responsiveness B | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | deliberative | medium | 40 | 40 | 0 | 822,696.5 | 62.4 | — | — | 61.4 | 56.9 | 65.0 | 57.8 | — | — | 64.3 | 62.0 | 66.7 |
| Evidence responsiveness B | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 60.0 | 52.5 | 65.8 | 62.3 | 59.8 | 64.7 | 65.0 | 61.7 | 66.7 | 63.8 | 61.1 | 66.1 |
| Evidence responsiveness B | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | sequential | medium | 40 | 40 | 0 | 310,695.5 | 64.2 | 61.3 | 66.7 | 61.4 | 58.0 | 64.4 | 63.7 | 60.5 | 66.7 | 65.1 | 63.5 | 66.7 |
| Evidence responsiveness B | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | deliberative | medium | 40 | 33 | 7 | 810,756.0 | 64.9 | 62.4 | 66.7 | 60.9 | 57.9 | 63.4 | 63.2 | 59.3 | 66.7 | 65.6 | 63.9 | 66.7 |
| Focal recovery (%) | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-6-astra | Codex | sequential | medium | 40 | 39 | 1 | 80,126.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-6-astra | Codex | deliberative | medium | 40 | 40 | 0 | 255,276.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 90.0 | 59.6 | 98.2 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-5.6-sol | Codex | sequential | medium | 40 | 40 | 0 | 176,867.5 | 90.0 | 59.6 | 98.2 | 100.0 | 72.2 | 100.0 | 80.0 | 49.0 | 94.3 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-5.6-sol | Codex | deliberative | medium | 40 | 38 | 2 | 547,345.5 | 80.0 | 49.0 | 94.3 | 100.0 | 72.2 | 100.0 | 40.0 | 16.8 | 68.7 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 90.0 | 59.6 | 98.2 | 100.0 | 72.2 | 100.0 | 70.0 | 39.7 | 89.2 | 90.0 | 59.6 | 98.2 |
| Focal recovery (%) | gpt-5.6-terra | Codex | sequential | medium | 40 | 39 | 1 | 58,512.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 70.0 | 39.7 | 89.2 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-5.6-terra | Codex | deliberative | medium | 40 | 39 | 1 | 160,931.0 | 90.0 | 59.6 | 98.2 | 90.0 | 59.6 | 98.2 | 20.0 | 5.7 | 51.0 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 40.0 | 16.8 | 68.7 | 70.0 | 39.7 | 89.2 | 10.0 | 1.8 | 40.4 | 50.0 | 23.7 | 76.3 |
| Focal recovery (%) | gpt-5.6-luna | Codex | sequential | medium | 40 | 39 | 1 | 79,146.0 | 10.0 | 1.8 | 40.4 | 80.0 | 49.0 | 94.3 | 0.0 | 0.0 | 27.8 | 60.0 | 31.3 | 83.2 |
| Focal recovery (%) | gpt-5.6-luna | Codex | deliberative | medium | 40 | 38 | 2 | 237,172.5 | 10.0 | 1.8 | 40.4 | 80.0 | 49.0 | 94.3 | 0.0 | 0.0 | 27.8 | 70.0 | 39.7 | 89.2 |
| Focal recovery (%) | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 |
| Focal recovery (%) | claude-opus-5 | Custom runner | sequential | medium | 20 | 19 | 1 | 309,733.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 |
| Focal recovery (%) | claude-opus-5 | Custom runner | deliberative | medium | 20 | 18 | 2 | 1,222,954.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 |
| Focal recovery (%) | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 90.0 | 59.6 | 98.2 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 100.0 | 72.2 | 100.0 | 80.0 | 49.0 | 94.3 | 80.0 | 49.0 | 94.3 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | sequential | medium | 40 | 40 | 0 | 265,093.5 | 100.0 | 72.2 | 100.0 | 90.0 | 59.6 | 98.2 | 80.0 | 49.0 | 94.3 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | deliberative | medium | 40 | 40 | 0 | 822,696.5 | 100.0 | 72.2 | 100.0 | 80.0 | 49.0 | 94.3 | 70.0 | 39.7 | 89.2 | 90.0 | 59.6 | 98.2 |
| Focal recovery (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 50.0 | 23.7 | 76.3 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | sequential | medium | 40 | 40 | 0 | 310,695.5 | 90.0 | 59.6 | 98.2 | 100.0 | 72.2 | 100.0 | 70.0 | 39.7 | 89.2 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | deliberative | medium | 40 | 33 | 7 | 810,756.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 80.0 | 49.0 | 94.3 | 100.0 | 72.2 | 100.0 |
| Weighted condition score C | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 86.3 | 85.6 | 87.2 | 83.9 | 82.6 | 84.9 | 82.8 | 81.3 | 84.2 | 84.6 | 83.4 | 85.9 |
| Weighted condition score C | gpt-6-astra | Codex | sequential | medium | 40 | 39 | 1 | 80,126.0 | 82.3 | 79.8 | 84.8 | 86.4 | 84.4 | 88.4 | 85.7 | 84.5 | 87.1 | 83.6 | 81.6 | 85.6 |
| Weighted condition score C | gpt-6-astra | Codex | deliberative | medium | 40 | 40 | 0 | 255,276.0 | 86.1 | 85.8 | 86.5 | 84.6 | 83.8 | 85.4 | 85.5 | 84.8 | 85.9 | 84.6 | 83.9 | 85.4 |
| Weighted condition score C | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 82.1 | 79.3 | 84.5 | 83.5 | 82.1 | 84.7 | 75.9 | 66.0 | 81.8 | 82.4 | 80.4 | 84.1 |
| Weighted condition score C | gpt-5.6-sol | Codex | sequential | medium | 40 | 40 | 0 | 176,867.5 | 76.1 | 64.4 | 83.2 | 82.1 | 80.9 | 83.3 | 74.0 | 60.2 | 83.8 | 80.9 | 78.6 | 83.2 |
| Weighted condition score C | gpt-5.6-sol | Codex | deliberative | medium | 40 | 38 | 2 | 547,345.5 | 70.2 | 55.1 | 81.6 | 82.4 | 79.6 | 84.5 | 53.8 | 38.2 | 69.7 | 82.5 | 80.4 | 84.2 |
| Weighted condition score C | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 74.2 | 62.9 | 81.7 | 83.0 | 78.9 | 85.5 | 73.0 | 64.1 | 81.0 | 80.0 | 72.6 | 85.2 |
| Weighted condition score C | gpt-5.6-terra | Codex | sequential | medium | 40 | 39 | 1 | 58,512.0 | 79.0 | 74.3 | 83.4 | 83.4 | 80.2 | 85.4 | 70.3 | 59.7 | 79.1 | 83.7 | 82.2 | 85.0 |
| Weighted condition score C | gpt-5.6-terra | Codex | deliberative | medium | 40 | 39 | 1 | 160,931.0 | 73.8 | 62.2 | 80.8 | 79.9 | 72.4 | 84.7 | 51.7 | 40.7 | 63.4 | 82.1 | 79.6 | 84.4 |
| Weighted condition score C | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 35.8 | 21.3 | 51.1 | 59.8 | 45.1 | 73.1 | 39.2 | 29.5 | 49.9 | 55.9 | 41.0 | 69.9 |
| Weighted condition score C | gpt-5.6-luna | Codex | sequential | medium | 40 | 39 | 1 | 79,146.0 | 24.3 | 17.1 | 33.6 | 68.2 | 60.1 | 75.2 | 24.8 | 19.9 | 30.4 | 55.7 | 40.3 | 70.9 |
| Weighted condition score C | gpt-5.6-luna | Codex | deliberative | medium | 40 | 38 | 2 | 237,172.5 | 23.2 | 14.8 | 34.3 | 63.3 | 50.9 | 74.3 | 26.9 | 21.5 | 32.4 | 63.8 | 51.3 | 75.0 |
| Weighted condition score C | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 82.4 | 78.8 | 85.6 | 79.5 | 77.7 | 82.4 | 78.9 | 78.2 | 79.5 | 78.5 | 75.2 | 81.2 |
| Weighted condition score C | claude-opus-5 | Custom runner | sequential | medium | 20 | 19 | 1 | 309,733.0 | 79.9 | 79.0 | 81.3 | 78.4 | 77.6 | 79.7 | 80.3 | 78.2 | 83.9 | 79.3 | 76.0 | 83.0 |
| Weighted condition score C | claude-opus-5 | Custom runner | deliberative | medium | 20 | 18 | 2 | 1,222,954.0 | 80.3 | 78.7 | 83.4 | 79.0 | 77.2 | 80.3 | 79.8 | 78.8 | 81.3 | 81.1 | 77.9 | 83.8 |
| Weighted condition score C | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 77.3 | 73.6 | 79.4 | 78.7 | 77.7 | 79.4 | 69.7 | 55.6 | 78.3 | 79.1 | 78.2 | 79.7 |
| Weighted condition score C | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 76.5 | — | — | 68.8 | 56.3 | 78.1 | 70.6 | 58.7 | 79.0 | 73.4 | 68.5 | 77.7 |
| Weighted condition score C | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | sequential | medium | 40 | 40 | 0 | 265,093.5 | 72.1 | — | — | 71.3 | 61.0 | 77.4 | 66.8 | 54.6 | 76.1 | 75.8 | 72.4 | 78.2 |
| Weighted condition score C | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | deliberative | medium | 40 | 40 | 0 | 822,696.5 | 74.3 | — | — | 64.4 | 50.4 | 75.2 | 66.9 | — | — | 68.0 | 57.5 | 76.2 |
| Weighted condition score C | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 68.6 | 66.1 | 71.4 | 77.6 | 76.0 | 78.7 | 59.6 | 48.0 | 71.2 | 78.5 | 78.0 | 78.9 |
| Weighted condition score C | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | sequential | medium | 40 | 40 | 0 | 310,695.5 | 67.5 | 57.1 | 74.4 | 78.0 | 77.4 | 78.6 | 65.0 | 53.2 | 75.6 | 78.5 | 77.8 | 79.1 |
| Weighted condition score C | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | deliberative | medium | 40 | 33 | 7 | 810,756.0 | 75.7 | 72.4 | 78.6 | 77.5 | 76.3 | 78.4 | 69.8 | 59.7 | 77.6 | 79.0 | 78.6 | 79.3 |

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
