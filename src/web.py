"""Web page: ask a question and watch the retry loop.

    streamlit run src/web.py
"""

import sys
from collections import Counter
from pathlib import Path

# Streamlit runs this file as a script, so the project root isn't on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.ingest import document_files, ingest  # noqa: E402
from src.llm import LLM_ERRORS  # noqa: E402
from src.logs import step_totals  # noqa: E402
from src.nodes import is_fallback  # noqa: E402
from src.service import add_document, answer, problem, remove_uploads  # noqa: E402

# (button label, icon, question)
EXAMPLES = [
    ("Answered directly", ":material/bolt:",
     "What is the maximum flight time of the Nimbus N7?"),
    ("Needs a retry", ":material/autorenew:",
     "When does the drone fly home by itself?"),
    ("Not in the documents", ":material/block:",
     "What battery chemistry does the Nimbus N7 use?"),
]

# Names for the graph's steps, as shown on the page.
STEP_NAMES = {
    "retrieve": "Search",
    "filter_chunks": "Filter",
    "generate": "Answer",
    "critic": "Check",
    "transform_query": "Rewrite query",
    "give_up": "Give up",
}


def documents(count: int) -> str:
    return f"{count} document" + ("" if count == 1 else "s")


# --- page sections -----------------------------------------------------------

def header() -> None:
    st.title("Self-healing RAG", anchor=False)
    st.markdown(
        "Answers questions about the documents of Nimbus Robotics, a made-up drone "
        "company. Every answer is **checked against the documents**. If the check "
        "fails, the search is rewritten and tried again, and if nothing supports an "
        "answer, it says so instead of guessing."
    )
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge("Search", icon=":material/search:", color="blue")
        st.markdown(":material/arrow_forward:")
        st.badge("Answer", icon=":material/edit_note:", color="violet")
        st.markdown(":material/arrow_forward:")
        st.badge("Check", icon=":material/fact_check:", color="green")
        st.markdown(":material/arrow_forward:")
        st.badge("Accept or retry", icon=":material/autorenew:", color="orange")


def ask_box() -> str | None:
    """The examples and the question form. Returns a question to answer, if any."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown("**Try an example:**")
            for label, icon, question in EXAMPLES:
                st.button(label, icon=icon, on_click=use_example, args=(question,))
        with st.form("question_form", border=False):
            with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
                question = st.text_input(
                    "Question", key="question", label_visibility="collapsed",
                    placeholder="For example: How long is the warranty on the N7?",
                )
                submitted = st.form_submit_button("Ask", type="primary",
                                                  icon=":material/send:", width="content")
    if (submitted or st.session_state.pop("example_clicked", False)) and question.strip():
        return question.strip()
    return None


def use_example(question: str) -> None:
    st.session_state.question = question
    st.session_state.example_clicked = True


def answer_card(result: dict) -> None:
    st.subheader("Answer", anchor=False)
    with st.container(border=True):
        if result["grounded"]:
            st.badge("Grounded in the documents", icon=":material/verified:", color="green")
        else:
            st.badge("Declined: not supported by the documents",
                     icon=":material/do_not_disturb_on:", color="orange")
        st.markdown(result["answer"])

    # Equal columns keep the cards in one row (Streamlit stacks them on phones).
    # No "attempts" card: it's always retries + 1, and the timeline numbers them.
    retries, sources, seconds = st.columns(3, gap="small")
    retries.metric("Retries", result["retries"], border=True)
    sources.metric("Sources", len(result["sources"]), border=True)
    seconds.metric("Time", f"{result['seconds']:.0f} s", border=True)

    if result["sources"]:
        st.markdown("**Sources**" if result["sources_cited"] else "**Chunks used**")
        for source in result["sources"]:
            with st.expander(f"[{source['number']}] {source['file']}",
                             icon=":material/description:"):
                st.text(source["text"])

    totals = step_totals(result["timings"])
    if totals:
        st.markdown("**Time per step**")
        data = pd.DataFrame({
            "step": [STEP_NAMES.get(step, step) for step in totals],
            "seconds": [round(seconds, 1) for seconds in totals.values()],
        })
        chart = alt.Chart(data).mark_bar(color="#6366F1", cornerRadiusEnd=4).encode(
            x=alt.X("seconds:Q", title="seconds", scale=alt.Scale(domainMin=0)),
            y=alt.Y("step:N", title=None, sort=None),  # the order the steps ran in
            tooltip=["step", "seconds"],
        ).properties(height=40 * len(data) + 30)
        st.altair_chart(chart, width="stretch")


def attempt_timeline(result: dict) -> None:
    st.subheader("How it got there", anchor=False)
    attempts = result["attempts"]
    for number, attempt in enumerate(attempts, 1):
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**Attempt {number}**")
                if is_fallback(attempt["answer"]):
                    # The critic is skipped when the generator found nothing to answer with.
                    st.badge("No answer in these chunks", icon=":material/search_off:",
                             color="gray")
                elif attempt["grounded"]:
                    st.badge("Critic: grounded", icon=":material/check_circle:", color="green")
                else:
                    st.badge("Critic: rejected", icon=":material/cancel:", color="red")
            st.markdown(f":material/search: `{attempt['query']}`")
            with st.container(horizontal=True, gap="small"):
                for name, count in Counter(attempt["sources"]).items():
                    st.badge(name if count == 1 else f"{name} x{count}", color="gray")
            st.markdown(attempt["answer"])
            if not is_fallback(attempt["answer"]):
                st.caption(attempt["critique"])
        if number < len(attempts):
            st.caption(":material/autorenew: Rewrote the search query and tried again")
    if not result["grounded"]:
        st.info("Out of retries, so it says it doesn't know instead of guessing.",
                icon=":material/info:")


def how_it_works() -> None:
    """Shown before the first question: what each example demonstrates."""
    st.markdown("Each example takes a different path through the loop:")
    cards = [
        (":material/bolt:", "Answered directly",
         f"The first search finds the right chunk among the top {config.RETRIEVER_K}. "
         "The answer cites it and the critic accepts it on the first attempt."),
        (":material/autorenew:", "Needs a retry",
         "The question is worded unlike the documents, so the first search misses. "
         "The model rewrites the search query and finds the answer on the next attempt."),
        (":material/block:", "Not in the documents",
         "A trap: the documents give the battery type of two other drone models, but "
         f"not of the N7. After {config.MAX_RETRIES} retries it says it doesn't know "
         "instead of borrowing the wrong product's answer."),
    ]
    for column, (icon, title, text) in zip(st.columns(len(cards)), cards):
        with column.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.caption(text)


def sidebar() -> None:
    with st.sidebar:
        st.header("Documents", anchor=False)
        if notice := st.session_state.pop("notice", None):
            st.success(notice, icon=":material/check:")
        files = document_files()
        with st.expander(f"{documents(len(files))} indexed", icon=":material/folder:"):
            for path in files:
                st.markdown(f":material/description: {path.name}")

        # A new key empties the upload box after its files have been added.
        round_ = st.session_state.setdefault("upload_round", 0)
        uploads = st.file_uploader(
            "Add documents", type=["md", "txt", "pdf"], accept_multiple_files=True,
            key=f"uploads_{round_}",
        )
        if uploads and st.button("Add and rebuild the index", icon=":material/upload:",
                                 type="primary", width="stretch"):
            try:
                for upload in uploads:
                    add_document(upload.name, upload.getvalue())
            except ValueError as err:
                st.error(str(err))
            else:
                st.session_state.upload_round = round_ + 1
                rebuild_and_rerun(f"Added {documents(len(uploads))}.")

        if any(config.UPLOAD_DIR.glob("*")) and st.button(
            "Remove uploaded documents", icon=":material/delete:", width="stretch"
        ):
            rebuild_and_rerun(f"Removed {documents(remove_uploads())}.")

        st.header("Settings", anchor=False)
        st.markdown(
            f"**Model** {config.LLM_MODEL}  \n"
            f"**Search** {config.SEARCH}, top {config.RETRIEVER_K} chunks  \n"
            f"**Chunks** {config.CHUNK_SIZE} characters  \n"
            f"**Retries** up to {config.MAX_RETRIES}"
        )
        st.caption("Runs locally with Ollama, Chroma and LangGraph. "
                   "The settings are explained in .env.example.")


def rebuild_and_rerun(notice: str) -> None:
    # The document list above was drawn before the change, so rebuild the
    # index, then run the page again and show the notice there.
    with st.status("Rebuilding the index ..."):
        ingest()
    st.session_state.notice = notice
    st.rerun()


# --- the page ----------------------------------------------------------------

st.set_page_config(page_title="Self-healing RAG", page_icon=":material/fact_check:",
                   layout="wide")
header()

issue = problem()
if issue:
    st.error(issue, icon=":material/error:")
    st.stop()

sidebar()

question = ask_box()
if question:
    with st.spinner("Searching, answering and checking ..."):
        try:
            st.session_state.result = answer(question, source="web")
        except LLM_ERRORS as err:
            st.error(f"The model call failed: {err}", icon=":material/error:")

# Kept in the session so the answer stays when something else reruns the page,
# e.g. choosing a file in the sidebar.
result = st.session_state.get("result")
if result:
    left, right = st.columns(2, gap="large")
    with left:
        answer_card(result)
    with right:
        attempt_timeline(result)
else:
    how_it_works()
