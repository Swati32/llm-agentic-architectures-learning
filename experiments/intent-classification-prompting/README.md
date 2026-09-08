# Intent Classification: Comparing Prompting Techniques

Seven ways to ask a small open model to classify a customer support message into one of 77 fine-grained intents — and what each one actually costs to run.

**TL;DR:** dynamically retrieving relevant examples per query beat every other technique by a wide margin (90.5% vs. 44–67%), fixed static examples barely beat no examples at all, decomposing the decision into "group, then intent" made accuracy *worse*, and self-consistency's extra voting didn't pay for its 3x token cost.

## Setup

**Dataset:** [Banking77](https://huggingface.co/datasets/PolyAI/banking77) — 13,083 customer support queries across 77 banking intents, all in one domain. It was picked specifically because its classes are semantically close (`card_not_working` vs. `virtual_card_not_working`, `top_up_failed` vs. `top_up_reverted`), so getting it right requires more than recognizing a topic — it requires distinguishing near-synonyms. Every technique ran on the same stratified sample of 231 test queries (proportional to each intent's share of the test set), drawn once and reused across all seven techniques, so no technique saw an easier slice.

**Model:** `llama3.1:8b` (Q4_K_M quantization), served locally via [Ollama](https://ollama.com). This wasn't the original plan — the experiment started on HF Inference Providers, which ran out of free monthly credits mid-run; Claude was considered next, but Claude Pro doesn't include API credits, and paying per-token for a bulk classification benchmark didn't make sense. A free, local 8B model turned out to be the right call anyway: it removes cost as a variable entirely and makes the whole pipeline reproducible on any machine with Ollama installed.

**Output schema:** Banking77 ships no official coarse categories, so this project defines its own — a 10-group taxonomy over the 77 intents (`taxonomy.py`), authored by hand against the dataset's real label strings. Every technique is asked to return **both** a coarse group and a fine intent, as strict JSON:

```json
{"coarse_group": "Transfers", "intent": "failed_transfer"}
```

That two-field schema is what makes several of the metrics below possible — Coarse Accuracy, Fine Accuracy, and Hierarchy Consistency (does the model's own stated intent actually belong to its own stated group?) are all readable off the same response.

## Techniques compared

| Technique | Idea |
|---|---|
| Zero-shot | Label catalog only, no examples |
| Zero-shot, schema-guided | Same, but every label carries a one-line definition |
| Few-shot, fixed | A static set of 10 examples (one per coarse group) in every prompt |
| Few-shot, semantic retrieval | 5 nearest training examples per query, by embedding similarity |
| Chain-of-thought | Reason step by step, then answer |
| Self-consistency | Sample chain-of-thought 3 times at temperature 0.7, majority vote |
| Hierarchical (coarse-to-fine) | Two calls: pick the group, then pick the intent from only that group's candidates |

## Evaluation methodology

**Quality:**
- **Fine Accuracy** — exact match on the specific intent
- **Macro F1** — unweighted mean F1 across all 77 intents, so the handful of rare classes aren't drowned out by common ones
- **Coarse Accuracy** — exact match on the 10-group label
- **Label Validity Rate** — did the response parse to a real label at all, independent of whether it was correct
- **Hierarchy Consistency Rate** — does the model's stated intent actually belong to its stated group

**Operational:** tokens (prompt + completion), latency, time to first token, context payload size, error rate, plus two that only apply to the retrieval technique — Semantic Cache Hit Rate (queries answered from a cached near-duplicate prediction without a fresh model call) and Empty Retrieval Rate (queries where nothing in the training pool was similar enough to use as a demonstration).

## Results

| Technique | Fine Acc. | Macro F1 | Coarse Acc. | Hierarchy Consistency | Mean Prompt Tokens | Mean Latency |
|---|---|---|---|---|---|---|
| Zero-shot | 59.7% | 57.7% | 81.8% | 95.7% | 697 | 1.37s |
| Zero-shot, schema-guided | 67.1% | 65.9% | 81.8% | 94.8% | 1,515 | 1.59s |
| Few-shot, fixed | 64.5% | 61.5% | 80.5% | 94.4% | 1,080 | 1.65s |
| **Few-shot, retrieval** | **90.5%** | **90.2%** | **97.8%** | **100.0%** | 875 | 3.22s |
| Chain-of-thought | 63.2% | 60.1% | 81.8% | 95.7% | 728 | 3.18s |
| Self-consistency (n=3) | 62.3% | 60.5% | 83.1% | 93.9% | 2,183 | 9.69s |
| Hierarchical | 44.2% | 40.8% | 56.7% | 99.1% | 263 | 2.04s |

Every technique had a **0% error rate** across all 1,617 calls, and label validity stayed at or above 98.7% throughout — Llama 3.1 8B followed the strict-JSON instruction reliably regardless of technique.

Full per-query results, exact prompts, and confusion matrices are in the [Streamlit dashboard](app.py) (`streamlit run app.py`).

## What we learned

**1. Retrieval beat everything else, by a lot.** Few-shot with per-query semantic retrieval hit 90.5% fine accuracy — 23 points above the next best technique, and roughly 1.4-2x every other technique's number. It's also the only technique to reach 100% hierarchy consistency and a near-perfect 97.8% coarse accuracy. On a 77-class problem, this lines up with what the many-label in-context-learning literature (Milios et al., 2023) argues: no fixed prompt can demonstrate 77 classes, so *which* examples you show the model matters more than almost anything else you can do to the prompt.

**2. Fixed few-shot barely helps — and costs more than zero-shot.** 10 static examples (one per coarse group) moved fine accuracy from 59.7% to only 64.5%, while spending more tokens than zero-shot-schema. With 77 classes and only 10 demonstrated, the model has no example to lean on for the other 67 intents most of the time — the fixed set mostly teaches output format, not classification. This is the clearest evidence in this run that examples only help when they're *relevant* to the query, not just present.

**3. Decomposing into "group, then intent" made things worse, not better.** This was the most counterintuitive result. Hierarchical scored lowest on every quality metric, including Coarse Accuracy (56.7% — the actual worst of all seven techniques, well below the ~82% every other technique gets for the *same* coarse-group judgment made jointly with the intent). Forcing the model to commit to a group before it has considered the specific intent removes information rather than adding structure: the fine-grained cues in the query that would help distinguish "Card Payments" from "ATM & Cash Withdrawals" are exactly the cues the intent-level reasoning would have used. Once the wrong group is locked in, the second call is solving the wrong sub-problem. Decomposition is not automatically a simplification.

**4. Chain-of-thought didn't clearly help, and self-consistency didn't fix that.** Plain CoT (63.2%) landed in the same range as zero-shot-schema and fixed few-shot — reasoning text before the answer didn't reliably improve a task that's fundamentally a single lookup, not a multi-step derivation. Self-consistency, which samples CoT three times and majority-votes, scored *lower* than plain CoT (62.3%) while spending 3x the tokens and 3x the latency. Voting over noisy reasoning only helps when the noise is unbiased; if the model's CoT has a systematic blind spot for a given query, sampling it three times just gets three similar wrong answers.

**5. The cheapest real win was schema-guided zero-shot.** Adding a one-line definition to each of the 77 labels — no examples, one call — took zero-shot from 59.7% to 67.1% for roughly 2x the prompt tokens. Of all the techniques that don't need a training pool to draw from, this had the best accuracy-per-token.

**6. Latency and token cost track technique complexity almost exactly, except retrieval is worth its premium.** Self-consistency's 9.69s and 2,183 prompt tokens make it the most expensive technique here for one of the weaker accuracy scores — a clear case where the added cost isn't earning its keep. Retrieval, by contrast, costs more than zero-shot (875 vs. 697 prompt tokens, 3.22s vs. 1.37s) but converts that cost into by far the largest accuracy gain in the experiment.

## Caveats

- **Time to First Token here is a prefill-time proxy specific to local, single-user serving.** Ollama's KV-cache reuse across calls that share a long system prompt makes TTFT look small relative to total latency (e.g. chain-of-thought's 0.40s TTFT against a 3.18s total) because most of the shared prompt is already cached from the previous call. A stateless hosted API serving multiple users concurrently would not show this pattern — treat the *relative* TTFT differences between techniques here with more confidence than the absolute numbers.
- **One model, one run.** These numbers are specific to `llama3.1:8b` and a single pass over the sample (temperature 0 except where a technique samples on purpose); a larger or differently-trained model could easily change which techniques help. Self-consistency and chain-of-thought in particular are known to matter more on tasks with real multi-step reasoning — this one, deliberately, doesn't have much.
- **The 10-group taxonomy is this project's own construction**, not an official Banking77 label. Coarse Accuracy and Hierarchy Consistency are only as meaningful as that taxonomy is reasonable.

## Grounding research

- Brown et al., 2020 — *Language Models are Few-Shot Learners*
- Kojima et al., 2022 — *Large Language Models are Zero-Shot Reasoners*
- Wei et al., 2022 — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
- Wang et al., 2022 — *Self-Consistency Improves Chain of Thought Reasoning in Language Models*
- Zhao et al., 2021 — *Calibrate Before Use: Improving Few-Shot Performance of Language Models*
- Milios, Reddy & Bahdanau, 2023 — *In-Context Learning for Text Classification with Many Labels*
- Sclar et al., 2023 — *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*

## Reproducing this

```bash
ollama pull llama3.1:8b        # one-time
pip install -r requirements.txt
python run_experiment.py       # ~75 minutes on an M4 MacBook Air
streamlit run app.py
```
