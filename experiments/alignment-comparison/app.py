"""Streamlit dashboard: methodology and dataset, an architecture deep dive
(LoRA, then SFT / RLHF / DPO, each with a diagram and a plain-language
write-up), the comparison and win-rate results, and a query explorer with
a handful of curated example generations. Reads only results/records.json,
same as every other experiment in this repo; no model calls here.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import diagrams

RESULTS_PATH = Path(__file__).parent / "results" / "records.json"

TERMINOLOGY = {
    "Fine-tuning": (
        "Taking a model that already knows how to talk, and training it a bit more on new "
        "examples so it picks up a specific style or behavior. Think of it like a new hire who "
        "already knows how to do the job in general, going through a short training course on "
        "how *this specific company* wants things done."
    ),
    "LoRA (Low-Rank Adaptation)": (
        "A cheap way to fine-tune a model without touching almost any of its original numbers. "
        "Instead of adjusting the whole model, LoRA bolts on a small, separate set of trainable "
        "numbers next to a few chosen spots, and only trains those. See the LoRA section below "
        "for the full picture; [peft-comparison](../peft-comparison/README.md) is the experiment "
        "that goes deep on exactly how much cheaper this makes fine-tuning."
    ),
    "Preference pair": (
        "Two different responses to the same question, where a human said one was better than "
        "the other. Not a right-or-wrong answer key, just \"this one over that one.\" This is the "
        "raw material both RLHF and DPO learn from."
    ),
    "Reward model": (
        "A model whose only job is to read a response and give it a single number saying how "
        "good it is. It's trained on preference pairs first, then RLHF uses its scores to guide "
        "training, the same way a teacher's grade tells a student how they're doing without the "
        "student re-reading the whole textbook every time."
    ),
    "Policy": (
        "The model that's actually being trained to respond better. In RLHF, it's the "
        "one being steered by the reward model's scores. \"Policy\" is just the standard name "
        "for \"the thing making decisions\" in this kind of training."
    ),
    "PPO (Proximal Policy Optimization)": (
        "The specific method RLHF uses to update the policy based on the reward model's scores. "
        "\"Proximal\" means it deliberately takes small, careful steps instead of big ones, so "
        "the model doesn't suddenly swing into weird behavior chasing a high score."
    ),
    "KL divergence / KL penalty": (
        "A number measuring how far the model has drifted from where it started. RLHF adds a "
        "penalty for drifting too far, so the model doesn't get so obsessed with pleasing the "
        "reward model that it starts writing nonsense the reward model just happens to score highly."
    ),
    "DPO (Direct Preference Optimization)": (
        "A way to learn from the same preference pairs as RLHF, but without training a separate "
        "reward model or running the step-by-step PPO process. It goes straight from \"here's a "
        "pair humans compared\" to \"nudge the model to prefer the better one,\" in one training "
        "step. See the DPO section below for exactly how."
    ),
    "Win rate": (
        "Out of every head-to-head comparison between two techniques' responses, the share one "
        "technique was picked as better. 60% win rate means that technique's response was judged "
        "better 6 times out of 10."
    ),
    "Held-out / held-out prompts": (
        "Questions none of the three techniques ever saw during training. Testing on these "
        "instead of training questions is what makes the comparison fair. It's the same idea as "
        "a final exam covering material the class studied, but using new questions the students "
        "never saw the answers to."
    ),
}


# Hand-picked from the actual results after reviewing every held-out prompt's three
# responses and judgments, not a representative sample: each one exists to make a
# specific, visible difference between the techniques concrete, rather than asking a
# reader to click through all of them to rediscover it themselves. Filled in from the
# real run; see the Comparison tab's analysis for how these were selected.
SPOTLIGHT_PROMPTS = [
    {
        "title": "SFT imitates the shape of a helpful answer without the content",
        "why": (
            "SFT was trained to copy 'good' responses, but with only 600 examples, it "
            "sometimes learns the conversational texture of a good answer (friendly, "
            "affirmative) without learning to actually produce the substance. DPO's response "
            "here isn't perfect, but it's a real attempt to answer the question."
        ),
        "prompt": "How do I find my measurements for a formal men's suit?",
        "responses": {
            "SFT": "Yes, I do have a great list of measurements for men's suits. I’m glad you asked.",
            "RLHF (PPO)": "It’s a bit like asking someone to do the measurement for you. You can measure the feet, and then you can also measure the shirt.",
            "DPO": "To find the measurements for a formal men's suit, you can use a combination of measurements from the front and back. A good rule of thumb is to start with the length of the shirt, and then add the length of the pants. \n\nStart with your front, and add the length of the shirt. Then, add the length of the pants. \n\nThen, add the length of the shoes. And finally, add the length of your shirt. \n\nFor",
        },
    },
    {
        "title": "RLHF drifts into rambling that a weak reward model didn't catch",
        "why": (
            "RLHF's reward model scored only 48.1% accuracy on held-out pairs, worse than "
            "chance (see Which metrics actually mattered, below). This is what that looks like "
            "downstream: a fluent-sounding but repetitive, off-topic response that a "
            "near-random reward model had no reliable way to penalize."
        ),
        "prompt": "What are some things a residential electrician should be able to do?",
        "responses": {
            "SFT": "Most residential electricians work in commercial buildings, like commercial kitchens, gyms, and shopping centers. They might also work in residential homes.",
            "RLHF (PPO)": "You know, I’ve been doing this for 20 years, and I’ve seen it all. But I have to say, I’ve been doing this for a long time. There are so many things a residential electrician should be able to do, and I think it’s really important to have that kind of skill.",
            "DPO": "Typically, a residential electrician will work on a variety of issues, including wiring and circuit breaker trips, making sure the electricity is safe to use, and troubleshooting electrical issues. They may also be able to work on fixing problems related to appliances, such as how to adjust the thermostat, how to change the light bulb, and how to adjust the thermostat to make sure the lights are on correctly.",
        },
    },
    {
        "title": "DPO loses too: an honest counter-example",
        "why": (
            "DPO won this experiment's overall win rate decisively, but not on every single "
            "prompt. Here it collapses into a repetition loop, and SFT's plainer, more generic "
            "answer wins instead. Included so the win-rate numbers don't read as 'DPO always "
            "wins' when the real result is 'DPO wins most of the time.'"
        ),
        "prompt": "What are pros and cons of leasing a car?",
        "responses": {
            "SFT": "Pros of leasing a car:\n\n1. You can keep the car for as long as you want, it is a long-term investment.\n2. The car can be driven for many years without needing maintenance.\n3. The cost of the car is not significantly higher than the price of a similar vehicle.\n\nCons of leasing a car:\n\n1. You will pay the full price of the car, but you won’t get the same level of driving",
            "RLHF (PPO)": "Selling your car is a relatively new business, but leasing is still an option.",
            "DPO": "Lessee: Pros:\n\nIt is a way to pay for the car with cash.\n\nIt is a way to pay for the car with cash.\n\nIt is a way to pay for the car with cash.",
        },
    },
]


@st.cache_data
def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _which_metrics_mattered(data):
    """Written by hand after reviewing this run's actual numbers (see the
    README's "Which metrics actually mattered" section for the full
    version); kept here as its own function so it's easy to find and
    update the next time this experiment is re-run.
    """
    return data.get("which_metrics_mattered_analysis", "_Run `python run_experiment.py`, then fill this in from the real numbers._")


def technique_badge(name):
    colors = {"SFT": "#6B7280", "RLHF (PPO)": "#DB2777", "DPO": "#5B21B6"}
    color = colors.get(name, "#6B7280")
    return f'<span style="background:{color};color:white;padding:0.15rem 0.6rem;border-radius:1rem;font-size:0.85rem;">{name}</span>'


st.set_page_config(page_title="SFT vs RLHF vs DPO", layout="wide")

st.markdown(
    """
    <div style="background:linear-gradient(135deg, #5B21B6, #DB2777);padding:1.5rem 1.5rem;border-radius:0.5rem;margin-bottom:1.5rem;">
        <h1 style="color:white;margin:0;font-size:2rem;">SFT vs RLHF vs DPO</h1>
        <p style="color:white;opacity:0.9;margin:0.3rem 0 0 0;">Three ways to fine-tune a language model, trained on the same data, compared head to head.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not RESULTS_PATH.exists():
    st.warning(
        "No results yet. Run `python run_experiment.py` in this experiment's "
        "folder first (needs `pip install -r requirements-experiment.txt` and a "
        "local Ollama server for judging). This dashboard reads from `results/records.json`."
    )
    st.stop()

data = load_results()

tab_methodology, tab_architecture, tab_comparison, tab_explorer = st.tabs(
    ["Methodology & Dataset", "Architecture Deep Dive", "Comparison", "Query Explorer"]
)

with tab_methodology:
    st.header("Goal")
    st.markdown(
        "Does skipping the reward model and the step-by-step RL training loop (DPO) lose "
        "quality compared to the full RLHF recipe (a reward model plus PPO), or does it get "
        "there cheaper and more reliably? And how much do either of them actually buy you "
        "over just training on demonstrations directly (SFT)? All three techniques here start "
        "from the exact same SFT checkpoint, train through the exact same size of LoRA update "
        "(see the LoRA section below), and are judged on the exact same held-out prompts, so "
        "the only thing that differs between them is *what they're being trained to optimize*."
    )

    st.header(f"Dataset: {data['dataset']}")
    st.markdown(
        "[Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) is a dataset "
        "of AI assistant conversations where crowdworkers were shown two possible responses to "
        "the same message and picked which one they preferred, along two axes: helpfulness and "
        "harmlessness (the \"hh\" in the name). It comes from Anthropic's own paper on building "
        "RLHF-trained assistants, [Bai et al., 2022](https://arxiv.org/abs/2204.05862), and was "
        "built specifically to train reward models, which is exactly the role it plays here too.\n\n"
        "Each row has a `chosen` and a `rejected` transcript: identical conversation up to the "
        "final turn, then two different possible assistant replies, one marked as preferred. "
        "The full dataset has four parts: `helpful-base`, `helpful-online`, "
        "`helpful-rejection-sampled`, and `harmless-base`. This experiment uses only "
        "`helpful-base`, for two reasons. First, `harmless-base` is built from adversarial "
        "red-teaming prompts meant to provoke unsafe responses, which isn't appropriate content "
        "for a public dashboard's example outputs. Second, this experiment is about comparing "
        "*how* SFT, RLHF, and DPO learn from preference data, not about safety tuning "
        "specifically, so one clean, on-topic subset is enough.\n\n"
        "Within `helpful-base`, this experiment further restricts to single-turn conversations "
        "(one human message, one assistant reply), to keep prompts short and training tractable "
        "on a laptop. That still leaves tens of thousands of eligible pairs to sample from. "
        f"This run trained on {data['n_train_pairs']} of them, and evaluated on "
        f"{data['n_eval_prompts']} held-out prompts from hh-rlhf's separate test split, so "
        "nothing used to measure the techniques was ever seen during training."
    )

    st.header("How each technique uses this data, differently")
    st.markdown(
        "- **SFT** only ever looks at the `chosen` half of each pair, treated as \"here's the "
        "correct response,\" and never sees `rejected` at all.\n"
        "- **RLHF** uses the full pairs, but indirectly: they train a reward model first, and "
        "from that point on, RLHF's PPO stage never looks at the original pairs again, only at "
        "the reward model's opinion of whatever the policy generates on its own.\n"
        "- **DPO** uses the full pairs directly, every step, with no reward model in between."
    )

    st.header("Terminology")
    st.caption("Plain-language definitions for the terms used throughout this page.")
    for term, definition in TERMINOLOGY.items():
        with st.expander(term):
            st.markdown(definition)

with tab_architecture:
    st.header("First: LoRA, the training method all three techniques share")
    st.markdown(
        "Before getting into what makes SFT, RLHF, and DPO different, it helps to cover what "
        "they all have in common: none of them change the model's original numbers directly. "
        "A model like this one has 135 million individual numbers (its parameters) that "
        "control everything it does. Training all 135 million of them, every time, for every "
        "technique, would be slow and would need a lot of memory.\n\n"
        "LoRA is a shortcut. Picture the model's original numbers as a locked book: instead of "
        "rewriting pages in that book, LoRA writes a small set of sticky notes that sit next to "
        "a few specific pages, and only those sticky notes get written on during training. When "
        "the model runs, it reads the original page *and* the sticky note next to it, and "
        "combines the two. The book itself never changes.\n\n"
        "Every technique in this experiment, SFT, RLHF's reward model and policy, and DPO, "
        "trains its own fresh set of sticky notes on top of the same starting book. That's "
        "deliberate: it means any difference in the results comes from *what each technique is "
        "trying to teach the sticky notes to do*, not from one technique quietly getting more "
        "room to write than another. How much cheaper this makes training, and exactly how many "
        "numbers LoRA touches versus training the whole book, is its own question, answered in "
        "detail in [peft-comparison](../peft-comparison/README.md)."
    )
    st.markdown(diagrams.render("lora"), unsafe_allow_html=True)

    st.divider()
    st.header("SFT: copy the good example")
    st.markdown(
        "SFT (supervised fine-tuning) is the simplest of the three. For every training example, "
        "the model is shown a question and a good answer to it, and trained to get better at "
        "predicting that answer, one word at a time. It never sees a bad answer, and it's never "
        "asked to judge anything, only to imitate.\n\n"
        "This is the same basic idea as a student copying out a worked example from a textbook "
        "until they can reproduce the method on their own. It works well when the good answer is "
        "genuinely representative of what you want, and it's the fastest and simplest of the "
        "three to train, but it has a ceiling: the model can only get as good as the examples "
        "it was shown, and it never learns anything about what makes an answer *better*, only "
        "what one acceptable answer looks like."
    )
    st.markdown(diagrams.render("sft"), unsafe_allow_html=True)

    st.divider()
    st.header("RLHF (PPO): train a scorer, then train against it")
    st.markdown(
        "RLHF starts from the SFT model, then adds two more stages. First, it trains a "
        "**reward model**: a separate model shown pairs of responses and taught to give the "
        "better one a higher score, the way a teacher grades essays. Once that's trained, the "
        "second stage, **PPO**, has the policy (the model being trained) write its own new "
        "responses, has the reward model score them, and nudges the policy's numbers toward "
        "whatever tended to score higher.\n\n"
        "This is closer to learning by practicing and getting feedback than to copying an "
        "example: the model isn't shown a correct answer, it writes something itself and finds "
        "out how good it was. One risk with this setup is that the policy could learn to write "
        "things that score well with the reward model without actually being good responses, "
        "a failure mode usually called *reward hacking*. RLHF guards against this with a "
        "penalty (see \"KL divergence\" in Methodology) that punishes the policy for drifting "
        "too far from where it started, so it can't stray far chasing a high score."
    )
    st.markdown(diagrams.render("rlhf_ppo"), unsafe_allow_html=True)

    st.divider()
    st.header("DPO: skip the scorer, learn from the pairs directly")
    st.markdown(
        "DPO also starts from the SFT model, and uses the exact same chosen/rejected pairs "
        "RLHF's reward model is trained on. But instead of training a separate scorer and then "
        "running a whole practice-and-feedback loop, DPO does something more direct: for each "
        "pair, it just nudges the model to make the chosen response more likely and the "
        "rejected one less likely, compared to how a frozen, untouched copy of the SFT model "
        "would have scored that same pair. One training pass over the pairs, no scorer, no "
        "generation step, no separate practice loop.\n\n"
        "The appeal is obvious: it's simpler to implement, needs less infrastructure (no "
        "second model to train and keep in memory), and tends to be more stable to train than "
        "PPO's practice-and-feedback loop. The trade-off is that DPO never generalizes beyond "
        "the exact pairs it's shown the way a reward model can, since it has no separate scoring "
        "function of its own, only ever a direct comparison against pairs it's seen."
    )
    st.markdown(diagrams.render("dpo"), unsafe_allow_html=True)

with tab_comparison:
    st.header("Win rates")
    st.markdown(
        "Each technique's held-out prompts were generated fresh (not copied from the "
        "dataset), then judged pairwise by a local Ollama model, blind to which technique "
        "produced which response and with response order randomized per call to avoid "
        "position bias. This is the number that actually answers this experiment's goal "
        "question; everything below it is *why*, not *what*."
    )
    win_rates = data["overall_win_rates"]
    cols = st.columns(len(win_rates))
    for col, (technique, rate) in zip(cols, win_rates.items()):
        col.metric(technique, f"{rate:.1%}" if rate is not None else "n/a")

    pairwise_df = pd.DataFrame(data["pairwise_summary"])
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.rename(
            columns={
                "technique_a": "Technique A",
                "technique_b": "Technique B",
                "a_win_rate": "A win rate",
                "b_win_rate": "B win rate",
                "tie_rate": "Tie rate",
                "n": "N",
            }
        )
        for col in ["A win rate", "B win rate", "Tie rate"]:
            pairwise_df[col] = pairwise_df[col].map(lambda x: f"{x:.1%}")
        st.dataframe(pairwise_df, hide_index=True, use_container_width=True)

    st.header("Training metrics")
    metrics_rows = []
    for key, label in [("sft", "SFT"), ("reward_model", "Reward model"), ("rlhf_ppo", "RLHF (PPO)"), ("dpo", "DPO")]:
        m = data["training_metrics"].get(key, {})
        metrics_rows.append(
            {
                "Technique": label,
                "Wall-clock (s)": round(m.get("wall_clock_seconds", 0), 1),
                "Peak memory (MB)": round(m.get("peak_memory_mb", 0), 0),
                "Train loss": m.get("train_loss"),
                "Held-out accuracy": m.get("held_out_accuracy") or m.get("held_out_reward_accuracy"),
                "Examples": m.get("num_examples"),
            }
        )
    st.dataframe(pd.DataFrame(metrics_rows), hide_index=True, use_container_width=True)
    st.caption(
        "RLHF's row above is its PPO stage only; the reward model it depends on is trained "
        "separately, one row up, and its cost should be counted alongside PPO's when comparing "
        "RLHF's total training cost to SFT's or DPO's."
    )

    st.header("Which metrics actually mattered")
    st.markdown(_which_metrics_mattered(data))

    st.header("When to use which technique")
    st.markdown(
        "**Narrow task, demonstration data already captures it → SFT.** Imagine you only want "
        "a model that writes polite replies to customer complaints. If you already have "
        "thousands of examples of a human writing exactly that kind of reply, showing the model "
        "those examples directly teaches it to imitate the pattern. There's no judgment call "
        "being asked of the model, every example it sees is already a correct answer, so "
        "there's nothing to weigh between \"better\" and \"worse.\" In practice, teams building "
        "a tool scoped to one specific, well-bounded job have found plain SFT sufficient for "
        "exactly this reason, because the full range of desired behavior can realistically be "
        "demonstrated ([Taori et al., 2023](https://crfm.stanford.edu/2023/03/13/alpaca.html)).\n\n"
        "**Huge, unpredictable variety of queries → RLHF.** Now imagine a general assistant "
        "fielding millions of completely different requests: coding help, medical questions, "
        "creative writing, disagreements about politics. No fixed set of \"here's the one right "
        "answer\" examples can cover territory that wide. What helps instead is teaching the "
        "model a general sense of what a good answer looks like, one it can apply to situations "
        "it's never specifically seen, which is exactly what a reward model provides: not a "
        "memorized set of good answers, but a general scoring sense that can judge a brand-new "
        "response. Teams building open-ended assistants at large scale have leaned on RLHF for "
        "this reason ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155), "
        "[Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).\n\n"
        "**Real compute constraints, want stability and simplicity → DPO.** Teams working under "
        "tighter compute budgets, or that want a simpler and more stable training pipeline, "
        "have found DPO captures most of the benefit of preference tuning without the "
        "instability and infrastructure cost of a full reward-model-plus-RL loop "
        "([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290), "
        "[Tunstall et al., 2023](https://arxiv.org/abs/2310.16944)).\n\n"
        "**This isn't a strict ranking, though.** At least one team that started with DPO later "
        "found a well-tuned RL approach outperformed it on certain tasks, a real-world echo of "
        "the finding that DPO's simplicity can come at some cost to the ceiling RLHF reaches "
        "when tuned well ([Xu et al., 2024](https://arxiv.org/abs/2404.10719))."
    )

with tab_explorer:
    st.header("A few interesting examples")
    st.markdown(
        "This isn't every held-out prompt, on purpose: the point of this tab is to make a few "
        "real, specific differences between the techniques visible, not to hand over all "
        f"{data['n_eval_prompts']} prompts for a reader to re-derive that themselves. Each "
        "example below was picked because it shows something the aggregate win-rate numbers on "
        "the Comparison tab can't: *what* the difference between two techniques actually looks "
        "like, in their own words."
    )

    if not SPOTLIGHT_PROMPTS:
        st.info("Examples not yet curated for this run. Run the experiment, review results/records.json, then fill in SPOTLIGHT_PROMPTS in app.py.")

    for example in SPOTLIGHT_PROMPTS:
        with st.container(border=True):
            st.markdown(f"**{example['title']}**")
            st.caption(example["why"])
            st.markdown(f"> {example['prompt']}")
            cols = st.columns(3)
            for col, technique in zip(cols, ["SFT", "RLHF (PPO)", "DPO"]):
                with col:
                    st.markdown(technique_badge(technique), unsafe_allow_html=True)
                    st.write(example["responses"][technique])
