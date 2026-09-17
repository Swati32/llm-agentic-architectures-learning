# LLM Agentic Architectures Learning

Notes, experiments, and small projects exploring large language models and agentic architectures. The focus is on the Hugging Face tools and ecosystem.

## Experiments

* [Intent Classification: Comparing Prompting Techniques](experiments/intent-classification-prompting/README.md): 7 prompting techniques compared on Banking77, with a Streamlit dashboard showing exact prompts, accuracy, and operational cost side by side. [Live dashboard](https://llm-agentic-architectures-learning-8nn36axsm2lvbifxk8w4tt.streamlit.app/).
* [RAG Architectures Compared](experiments/rag-architectures/README.md): 6 RAG architectures compared on MultiHop-RAG. Query Decomposition RAG won on answer quality (0.675 F1) despite mediocre-to-worst retrieval metrics, and Corrective RAG, built specifically to avoid hallucinating, had the worst incorrect-abstention rate of all six. A chunking sub-experiment is included. [Live dashboard](https://swati-rag-architectures.streamlit.app/).
* [Agentic Architectures: Sequential vs. Orchestrator Styles](experiments/agentic-architectures/README.md): 5 agent control-flow structures compared on HotpotQA. Supervisor + Verification Loop won on accuracy (60% exact match), while agentic metrics (not accuracy) revealed that an adaptive planner never once decided on its own that it had enough information, in all 40 runs. [Live dashboard](https://swati-agentic-architectures.streamlit.app/).
* [Evaluation Criteria in Practice: Plain LLM vs. RAG vs. Single Agent vs. Multi-Agent](experiments/evaluation-criteria-experiment/README.md): the same HotpotQA task run through 4 increasingly capable system shapes. Single-shot RAG tied the winning multi-agent shape (60.0% exact match each) and beat the looping single-agent shape (47.5%), because a plain retrieval of the raw question outperformed the agent's own self-generated search query. A new redundant-step-rate metric cleanly separated the multi-agent shape's successful verification rounds (0% redundant) from its failed ones (100% redundant). Not yet deployed to a public dashboard.
* [PEFT, LoRA, and QLoRA: A Theoretical Comparison](experiments/peft-comparison/README.md): full fine-tuning against five parameter-efficient fine-tuning techniques from four design families, compared by real trainable-parameter counts and memory footprints computed against `llama3.1:8b`'s architecture. No training run involved. [Live dashboard](https://swati-peft-comparison.streamlit.app/).
* [SFT vs RLHF vs DPO](experiments/alignment-comparison/README.md): a small model taken through SFT, then RLHF (a real reward model plus a real PPO loop, not just cited) and DPO, all real local training on a laptop, evaluated by LLM-judge win rate. DPO won decisively (75% win rate); RLHF's reward model came in under chance accuracy on held-out pairs at this data scale, and the run's own numbers explain why. [Live dashboard](https://swati-alignment-techinques.streamlit.app/).

## Cross-cutting notes

* [Evaluation Criteria Across LLM, RAG, Agent, and Multi-Agent Systems](EVALUATION_CRITERIA.md): not an experiment, a decision framework synthesized from the experiments above plus current research, on which evaluation criteria matter for which kind of system and how to use that to decide something. Has its own dashboard under [experiments/evaluation-criteria](experiments/evaluation-criteria/), not yet deployed publicly.

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
