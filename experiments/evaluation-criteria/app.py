"""Streamlit dashboard for the EVALUATION_CRITERIA.md framework doc at the
repo root. This is not an experiment with its own results/records.json:
it renders that one markdown file directly, so the dashboard and the doc
can never drift out of sync with each other. Edit EVALUATION_CRITERIA.md;
this app picks the change up on its next reload.

Single page, no tabs, on purpose: the dashboard only shows the three
sections a reader actually needs to use this framework (the four shapes,
the metrics that apply to them, and the decision framework for choosing
between them). Open gaps, future work, grounding research, and
reproducing this stay in the source document, readable on GitHub, but
aren't duplicated here.
"""

import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

DOC_PATH = Path(__file__).parent.parent.parent / "EVALUATION_CRITERIA.md"
MERMAID_RE = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)

SECTIONS = ["The four system shapes", "Critical metrics", "The decision framework"]


@st.cache_data
def load_sections() -> tuple[str, dict[str, str]]:
    text = DOC_PATH.read_text()
    parts = re.split(r"\n(?=## [^#])", text)
    preamble = parts[0]
    sections = {}
    for part in parts[1:]:
        header_line, _, body = part.partition("\n")
        title = header_line[3:].strip()
        sections[title] = body.strip()
    return preamble, sections


def render_mermaid(diagram: str, height: int = 130) -> None:
    # Same fix as the other dashboards in this repo: mermaid's startOnLoad
    # can race Streamlit's own layout pass inside a freshly-mounted tab,
    # rendering the diagram at zero width. Polling for a real width first
    # avoids that.
    components.html(
        f"""
        <div class="mermaid" id="diagram" style="font-family: sans-serif;">{diagram}</div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: false, flowchart: {{ useMaxWidth: true }} }});
            function renderWhenReady(attemptsLeft) {{
                var el = document.getElementById("diagram");
                if (el.offsetWidth > 0 || attemptsLeft <= 0) {{
                    mermaid.run({{ nodes: [el] }});
                }} else {{
                    requestAnimationFrame(function () {{ renderWhenReady(attemptsLeft - 1); }});
                }}
            }}
            renderWhenReady(60);
        </script>
        """,
        height=height,
    )


def render_body(body: str, diagram_height: int = 130) -> None:
    last_end = 0
    for match in MERMAID_RE.finditer(body):
        before = body[last_end:match.start()].strip()
        if before:
            st.markdown(before)
        render_mermaid(match.group(1), height=diagram_height)
        last_end = match.end()
    remainder = body[last_end:].strip()
    if remainder:
        st.markdown(remainder)


st.set_page_config(page_title="Evaluation Criteria Framework", layout="wide")

st.markdown(
    """
    <div style="background:linear-gradient(135deg, #5B21B6, #DB2777);padding:1.5rem 1.5rem;border-radius:0.5rem;margin-bottom:1.5rem;">
        <h1 style="color:white;margin:0;font-size:2rem;">Evaluation Criteria Across LLM, RAG, Agent, and Multi-Agent Systems</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if not DOC_PATH.exists():
    st.warning("EVALUATION_CRITERIA.md not found at the repo root. This dashboard has nothing else to read.")
    st.stop()

preamble, sections = load_sections()

st.markdown(preamble.split("\n\n", 1)[1] if "\n\n" in preamble else preamble)

for title in SECTIONS:
    if title not in sections:
        continue
    st.header(title)
    render_body(sections[title])
