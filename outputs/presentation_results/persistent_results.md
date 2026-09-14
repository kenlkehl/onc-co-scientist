# Persistent workflows + Biomni: condition metrics

Persistent workflows plus Qwen 3.8 27B / Biomni as a separate comparator. Biomni uses xhigh reasoning; other included runs use medium. Estimates, 95% CIs, execution counts, and token medians retain their original definitions. No runs are pooled across harnesses or workflows.

| Metric | Llm | Harness | Workflow | Reasoning effort | Runs attempted | Runs successfully completed | Runs ended with errors | Median output tokens per completed run | Expected-unmasked | Expected-unmasked ci95 low | Expected-unmasked ci95 high | Expected-masked | Expected-masked ci95 low | Expected-masked ci95 high | Surprising-unmasked | Surprising-unmasked ci95 low | Surprising-unmasked ci95 high | Surprising-masked | Surprising-masked ci95 low | Surprising-masked ci95 high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Precision (%) | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 97.7 | 95.1 | 99.7 | 93.1 | 90.0 | 96.2 | 93.5 | 89.5 | 97.1 | 93.5 | 89.7 | 97.2 |
| Precision (%) | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 95.7 | 94.6 | 96.8 | 96.0 | 94.4 | 97.7 | 91.4 | 87.1 | 95.3 | 93.2 | 87.9 | 97.0 |
| Precision (%) | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 97.6 | 95.6 | 99.4 | 98.4 | 97.5 | 99.3 | 97.0 | 94.8 | 98.8 | 97.8 | 93.7 | 100.0 |
| Precision (%) | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 59.5 | 36.9 | 78.5 | 91.9 | 81.9 | 98.2 | 72.2 | 64.1 | 80.2 | 85.2 | 73.7 | 95.8 |
| Precision (%) | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 92.5 | 86.2 | 98.4 | 93.9 | 89.4 | 98.4 | 93.5 | 89.8 | 97.2 | 91.9 | 82.4 | 98.3 |
| Precision (%) | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 97.6 | 92.9 | 100.0 | 97.4 | 93.5 | 100.0 | 95.5 | 89.0 | 100.0 | 99.5 | 98.5 | 100.0 |
| Precision (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 100.0 | 100.0 | 100.0 | 96.0 | 90.4 | 100.0 | 95.0 | 85.0 | 100.0 | 89.7 | 79.2 | 98.3 |
| Precision (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 99.2 | 97.7 | 100.0 | 100.0 | 100.0 | 100.0 | 82.4 | 68.9 | 93.8 | 99.3 | 98.0 | 100.0 |
| Recall (%) | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 68.3 | 66.7 | 71.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 68.3 | 66.7 | 71.7 |
| Recall (%) | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 63.3 | 56.7 | 66.7 | 66.7 | 66.7 | 66.7 | 61.7 | 51.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 51.1 | 37.8 | 63.3 | 66.7 | 66.7 | 66.7 | 60.0 | 51.7 | 65.0 | 61.7 | 53.3 | 66.7 |
| Recall (%) | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 13.3 | 4.4 | 22.2 | 37.8 | 23.3 | 52.2 | 21.7 | 11.7 | 31.7 | 41.7 | 30.0 | 53.3 |
| Recall (%) | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 63.3 | 56.7 | 66.7 | 66.7 | 66.7 | 66.7 | 56.7 | 41.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| Recall (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 60.0 | 50.0 | 66.7 | 58.9 | 47.8 | 66.7 | 60.0 | 50.0 | 66.7 | 58.3 | 50.0 | 66.7 |
| Recall (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 38.9 | 31.1 | 48.9 | 65.6 | 63.3 | 66.7 | 50.0 | 40.0 | 60.0 | 66.7 | 66.7 | 66.7 |
| Scientific F1* | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 80.3 | 78.5 | 83.0 | 77.6 | 76.5 | 78.7 | 77.8 | 76.4 | 79.0 | 78.8 | 76.5 | 81.8 |
| Scientific F1* | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 75.6 | 69.7 | 78.8 | 78.7 | 78.1 | 79.2 | 72.4 | 62.0 | 78.4 | 77.6 | 75.6 | 79.0 |
| Scientific F1* | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 64.3 | 49.9 | 76.0 | 79.5 | 79.2 | 79.8 | 73.4 | 66.8 | 78.4 | 75.1 | 68.4 | 80.0 |
| Scientific F1* | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 18.8 | 6.7 | 31.1 | 48.9 | 32.0 | 64.7 | 30.0 | 16.2 | 43.0 | 53.3 | 41.3 | 65.0 |
| Scientific F1* | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 77.4 | 75.1 | 79.5 | 77.9 | 76.3 | 79.5 | 77.8 | 76.5 | 79.1 | 77.1 | 73.4 | 79.4 |
| Scientific F1* | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 76.6 | 69.9 | 80.0 | 79.1 | 77.6 | 80.0 | 67.9 | 51.3 | 78.7 | 79.8 | 79.5 | 80.0 |
| Scientific F1* | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 74.0 | 65.0 | 80.0 | 71.6 | 58.5 | 80.0 | 73.0 | 63.0 | 80.0 | 70.6 | 61.1 | 79.2 |
| Scientific F1* | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 54.5 | 47.2 | 63.6 | 79.1 | 77.4 | 80.0 | 61.1 | 50.3 | 71.9 | 79.8 | 79.3 | 80.0 |
| Exploration E | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 65.8 | 65.6 | 66.0 | 64.3 | 64.1 | 64.4 | 65.2 | 65.0 | 65.3 | 64.5 | 64.1 | 64.9 |
| Exploration E | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 66.0 | 65.7 | 66.3 | 61.7 | 59.9 | 63.2 | 60.5 | 50.3 | 65.9 | 62.3 | 61.3 | 63.1 |
| Exploration E | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 59.2 | 46.5 | 66.0 | 63.1 | 61.5 | 64.4 | 65.6 | 64.8 | 66.2 | 64.0 | 62.1 | 65.1 |
| Exploration E | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 25.3 | 7.1 | 44.0 | 47.1 | 34.1 | 57.7 | 38.1 | 21.3 | 52.9 | 43.7 | 31.7 | 54.9 |
| Exploration E | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 64.0 | 61.5 | 65.3 | 61.3 | 59.6 | 62.8 | 63.9 | 61.9 | 65.5 | 63.9 | 63.6 | 64.0 |
| Exploration E | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 65.9 | 64.7 | 66.7 | 66.7 | 66.7 | 66.7 | 59.4 | 46.1 | 66.5 | 66.7 | 66.7 | 66.7 |
| Exploration E | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 64.8 | 64.4 | 65.2 | 59.4 | 56.0 | 62.1 | 64.7 | 64.1 | 65.1 | 61.1 | 60.3 | 62.0 |
| Exploration E | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 62.5 | 60.1 | 64.4 | 62.3 | 59.1 | 64.2 | 63.5 | 62.5 | 64.3 | 63.9 | 62.7 | 64.9 |
| Evidence responsiveness B | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 100.0 | 100.0 | 100.0 | 94.2 | 89.2 | 98.3 | 87.5 | 80.0 | 94.6 | 95.4 | 91.7 | 98.5 |
| Evidence responsiveness B | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 87.0 | 77.8 | 95.2 | 93.1 | 86.4 | 100.0 | 79.8 | 72.6 | 87.5 | 89.2 | 81.7 | 96.7 |
| Evidence responsiveness B | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 86.8 | 78.1 | 95.6 | 87.8 | 68.9 | 100.0 | 83.4 | 71.1 | 95.8 | 92.2 | 85.0 | 98.3 |
| Evidence responsiveness B | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 71.0 | 57.9 | 82.7 | 79.0 | 66.5 | 90.6 | 93.0 | 85.5 | 98.3 | 80.1 | 67.1 | 90.4 |
| Evidence responsiveness B | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 87.5 | 72.2 | 100.0 | 75.0 | 66.7 | 91.7 | 69.4 | 66.7 | 75.0 | 68.7 | 57.3 | 80.0 |
| Evidence responsiveness B | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 61.7 | 54.9 | 66.2 | 63.5 | 60.6 | 66.0 | 57.5 | 49.3 | 63.0 | 64.3 | 60.0 | 66.7 |
| Evidence responsiveness B | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 63.3 | — | — | 59.1 | 52.8 | 64.4 | 60.4 | 54.2 | 65.6 | 57.4 | 49.4 | 64.2 |
| Evidence responsiveness B | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 60.0 | 52.5 | 65.8 | 62.3 | 59.8 | 64.7 | 65.0 | 61.7 | 66.7 | 63.8 | 61.1 | 66.1 |
| Focal recovery (%) | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 90.0 | 59.6 | 98.2 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 90.0 | 59.6 | 98.2 | 100.0 | 72.2 | 100.0 | 70.0 | 39.7 | 89.2 | 90.0 | 59.6 | 98.2 |
| Focal recovery (%) | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 40.0 | 16.8 | 68.7 | 70.0 | 39.7 | 89.2 | 10.0 | 1.8 | 40.4 | 50.0 | 23.7 | 76.3 |
| Focal recovery (%) | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 | 100.0 | 56.6 | 100.0 |
| Focal recovery (%) | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 90.0 | 59.6 | 98.2 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 100.0 | 72.2 | 100.0 | 80.0 | 49.0 | 94.3 | 80.0 | 49.0 | 94.3 | 100.0 | 72.2 | 100.0 |
| Focal recovery (%) | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 100.0 | 72.2 | 100.0 | 100.0 | 72.2 | 100.0 | 50.0 | 23.7 | 76.3 | 100.0 | 72.2 | 100.0 |
| Weighted condition score C | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 86.3 | 85.6 | 87.2 | 83.9 | 82.6 | 84.9 | 82.8 | 81.3 | 84.2 | 84.6 | 83.4 | 85.9 |
| Weighted condition score C | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 82.1 | 79.3 | 84.5 | 83.5 | 82.1 | 84.7 | 75.9 | 66.0 | 81.8 | 82.4 | 80.4 | 84.1 |
| Weighted condition score C | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 74.2 | 62.9 | 81.7 | 83.0 | 78.9 | 85.5 | 73.0 | 64.1 | 81.0 | 80.0 | 72.6 | 85.2 |
| Weighted condition score C | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 35.8 | 21.3 | 51.1 | 59.8 | 45.1 | 73.1 | 39.2 | 29.5 | 49.9 | 55.9 | 41.0 | 69.9 |
| Weighted condition score C | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 82.4 | 78.8 | 85.6 | 79.5 | 77.7 | 82.4 | 78.9 | 78.2 | 79.5 | 78.5 | 75.2 | 81.2 |
| Weighted condition score C | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 77.3 | 73.6 | 79.4 | 78.7 | 77.7 | 79.4 | 69.7 | 55.6 | 78.3 | 79.1 | 78.2 | 79.7 |
| Weighted condition score C | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 76.5 | — | — | 68.8 | 56.3 | 78.1 | 70.6 | 58.7 | 79.0 | 73.4 | 68.5 | 77.7 |
| Weighted condition score C | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 68.6 | 66.1 | 71.4 | 77.6 | 76.0 | 78.7 | 59.6 | 48.0 | 71.2 | 78.5 | 78.0 | 78.9 |
