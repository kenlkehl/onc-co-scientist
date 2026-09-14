# Persistent workflows + Biomni: summary and focal interaction

Persistent workflows plus Qwen 3.8 27B / Biomni as a separate comparator. Biomni uses xhigh reasoning; other included runs use medium. Estimates, 95% CIs, execution counts, and token medians retain their original definitions. No runs are pooled across harnesses or workflows.

| Metric | Llm | Harness | Workflow | Reasoning effort | Runs attempted | Runs successfully completed | Runs ended with errors | Median output tokens per completed run | Value | Ci95 low | Ci95 high | Ci method | Ci status | Ci valid resamples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| summary_score | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 84.4 | 83.7 | 84.9 | joint whole-run bootstrap | available | 10000 |
| summary_score | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 80.9 | 77.9 | 82.8 | joint whole-run bootstrap | available | 10000 |
| summary_score | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 77.5 | 73.2 | 80.9 | joint whole-run bootstrap | available | 9897 |
| summary_score | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 46.6 | 38.9 | 53.1 | joint whole-run bootstrap | available | 10000 |
| summary_score | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 79.8 | 78.5 | 81.1 | joint whole-run bootstrap | available | 9981 |
| summary_score | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 76.1 | 71.7 | 78.7 | joint whole-run bootstrap | available | 9655 |
| summary_score | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 72.3 | — | — | joint whole-run bootstrap | insufficient eligible runs | 6388 |
| summary_score | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 70.6 | 66.9 | 74.0 | joint whole-run bootstrap | available | 9753 |
| label_surprise_interaction | gpt-6-astra | Codex | persistent | medium | 40 | 40 | 0 | 85,653.0 | 0.0 | -39.2 | 39.2 | Wilson-based MOVER | available | 0 |
| label_surprise_interaction | gpt-5.6-sol | Codex | persistent | medium | 40 | 40 | 0 | 172,152.0 | 10.0 | -30.1 | 51.2 | Wilson-based MOVER | available | 0 |
| label_surprise_interaction | gpt-5.6-terra | Codex | persistent | medium | 40 | 40 | 0 | 56,127.5 | 10.0 | -37.1 | 52.7 | Wilson-based MOVER | available | 0 |
| label_surprise_interaction | gpt-5.6-luna | Codex | persistent | medium | 40 | 36 | 4 | 77,247.5 | 10.0 | -40.3 | 60.1 | Wilson-based MOVER | available | 0 |
| label_surprise_interaction | claude-opus-5 | Custom runner | persistent | medium | 20 | 20 | 0 | 321,510.5 | 0.0 | -61.4 | 61.4 | Wilson-based MOVER | available | 0 |
| label_surprise_interaction | Inferact/Qwen3.8-27B-NVFP4 | Biomni | Biomni | xhigh | 40 | 34 | 6 | 81,632.5 | 10.0 | -30.1 | 51.2 | Wilson-based MOVER | available | 0 |
| label_surprise_interaction | Inferact/Qwen3.8-27B-NVFP4 | Custom runner | persistent | medium | 40 | 38 | 2 | 183,076.0 | 40.0 | -4.2 | 83.8 | Wilson-based MOVER | available | 0 |
| label_surprise_interaction | RedHatAI/Gemma-4-31B-IT-FP8-Dynamic | Custom runner | persistent | medium | 40 | 37 | 3 | 199,780.0 | 50.0 | 2.7 | 88.3 | Wilson-based MOVER | available | 0 |
