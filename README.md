# LLM Agentic Architectures Learning

Notes, experiments, and small projects exploring large language models and agentic architectures. The focus is on the Hugging Face tools and ecosystem.

## Experiments

* [Intent Classification: Comparing Prompting Techniques](experiments/intent-classification-prompting/README.md): 7 prompting techniques compared on Banking77, with a Streamlit dashboard showing exact prompts, accuracy, and operational cost side by side. [Live dashboard](https://llm-agentic-architectures-learning-8nn36axsm2lvbifxk8w4tt.streamlit.app/).
* [PEFT, LoRA, and QLoRA: A Theoretical Comparison](experiments/peft-comparison/README.md): full fine-tuning against five parameter-efficient fine-tuning techniques from four design families, compared by real trainable-parameter counts and memory footprints computed against `llama3.1:8b`'s architecture. No training run involved. [Live dashboard](https://swati-peft-comparison.streamlit.app/).
* [SFT vs RLHF vs DPO](experiments/alignment-comparison/README.md): a small model taken through SFT, then RLHF (a real reward model plus a real PPO loop, not just cited) and DPO, all real local training on a laptop, evaluated by LLM-judge win rate. DPO won decisively (75% win rate); RLHF's reward model came in under chance accuracy on held-out pairs at this data scale, and the run's own numbers explain why. [Live dashboard](https://swati-alignment-comparison.streamlit.app/).

## Roadmap

Planned next, in order of difficulty:

1. **Knowledge distillation on a richer domain.** Teacher versus distilled student, moving past text classification into either image generation (a full diffusion model against a distilled fast variant like SDXL-Turbo or LCM) or recommendation and search (a large embedding model against a distilled student, evaluated by retrieval ranking metrics like NDCG or Recall@k). Domain choice still open; recommendation and search fits this repo's LLM/agentic focus more directly than image generation does.

Previously planned as "SFT vs DPO, with RLHF/PPO covered theoretically," on the expectation that a credible PPO pipeline would be too unstable to run on laptop compute. That turned out to be wrong: a real reward model plus a real PPO loop, both LoRA-tuned on `SmolLM2-135M-Instruct`, trained stably end to end (after fixing a real MPS memory leak along the way, see that experiment's README). The roadmap item is done, not deferred; see [SFT vs RLHF vs DPO](experiments/alignment-comparison/README.md) above.

## Contents

* Notes and write-ups on LLM and agent concepts
* Hands-on experiments and prototypes
* Reference links and resources

## Status

🚧 Work in progress. This repo grows as learning progresses.
