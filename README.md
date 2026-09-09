# LLM Agentic Architectures Learning

Notes, experiments, and small projects exploring large language models and agentic architectures. The focus is on the Hugging Face tools and ecosystem.

## Experiments

* [Intent Classification: Comparing Prompting Techniques](experiments/intent-classification-prompting/README.md): 7 prompting techniques compared on Banking77, with a Streamlit dashboard showing exact prompts, accuracy, and operational cost side by side. [Live dashboard](https://llm-agentic-architectures-learning-8nn36axsm2lvbifxk8w4tt.streamlit.app/).
* [PEFT, LoRA, and QLoRA: A Theoretical Comparison](experiments/peft-comparison/README.md): full fine-tuning against five parameter-efficient fine-tuning techniques from four design families, compared by real trainable-parameter counts and memory footprints computed against `llama3.1:8b`'s architecture. No training run involved; run `streamlit run app.py` locally to view.

## Roadmap

Planned next, in order of difficulty:

1. **SFT vs DPO, with RLHF/PPO covered theoretically.** A small model taken through Base, then SFT, then DPO, with real training (`peft`/`trl`), evaluated by LLM-judge win-rate. RLHF/PPO is explained and cited rather than run, since a credible PPO pipeline needs a trained reward model plus an RL loop that stays unstable even with real compute, and a shaky toy run would tell us more about our implementation than about the technique.
2. **Knowledge distillation on a richer domain.** Teacher versus distilled student, moving past text classification into either image generation (a full diffusion model against a distilled fast variant like SDXL-Turbo or LCM) or recommendation and search (a large embedding model against a distilled student, evaluated by retrieval ranking metrics like NDCG or Recall@k). Domain choice still open; recommendation and search fits this repo's LLM/agentic focus more directly than image generation does.

## Contents

* Notes and write-ups on LLM and agent concepts
* Hands-on experiments and prototypes
* Reference links and resources

## Status

🚧 Work in progress. This repo grows as learning progresses.
