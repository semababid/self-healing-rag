import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import service
from src.nodes import FALLBACK_MESSAGE

PAGE = str(Path(__file__).resolve().parents[1] / "src" / "web.py")
ACCEPT = {"grounded": True, "reasoning": "The context says 38 minutes."}


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setattr(service, "problem", lambda: None)


def page() -> AppTest:
    return AppTest.from_file(PAGE, default_timeout=30).run()


def click(app: AppTest, label: str) -> AppTest:
    next(b for b in app.button if b.label == label).click()
    return app.run()


def markdown(app: AppTest) -> list[str]:
    return [m.value for m in app.main.markdown]


def badges(app: AppTest) -> list[str]:
    """Badge texts in the main area. st.badge renders as ":green-badge[:material/x: text]"."""
    found = []
    for value in markdown(app):
        found += re.findall(r":\w+-badge\[(?::material/\w+: )?(.*?)\]", value)
    return found


def test_shows_the_problem_when_not_ready(monkeypatch):
    monkeypatch.setattr(service, "problem", lambda: "Can't reach Ollama.")
    app = page()
    assert [e.value for e in app.error] == ["Can't reach Ollama."]
    assert not app.button  # nothing else is drawn


def test_explains_the_examples_before_the_first_question(ready):
    app = page()
    assert "Each example takes a different path through the loop:" in markdown(app)
    assert not app.metric  # no result yet


def test_example_shows_answer_sources_and_every_attempt(ready, llm, retriever):
    llm.replies = [FALLBACK_MESSAGE, "Nimbus N7 flight time", "38 minutes [1].", ACCEPT]

    app = click(page(), "Needs a retry")

    assert "38 minutes [1]." in markdown(app)
    assert {m.label: m.value for m in app.metric if m.label != "Time"} == {
        "Attempts": "2", "Retries": "1", "Sources": "1",
    }
    assert [e.label for e in app.main.expander] == ["[1] nimbus_handbook.md"]
    assert "**Attempt 1**" in markdown(app) and "**Attempt 2**" in markdown(app)
    # The first attempt was the generator finding nothing, not a critic verdict.
    found = badges(app)
    assert "Grounded in the documents" in found
    assert found.index("No answer in these chunks") < found.index("Critic: grounded")
    assert "nimbus_handbook.md" in found  # the files each attempt searched
    assert "The context says 38 minutes." in [c.value for c in app.main.caption]


def test_refusal_shows_declined_and_no_sources(ready, llm, retriever):
    llm.replies = [FALLBACK_MESSAGE, "query 2", FALLBACK_MESSAGE, "query 3", FALLBACK_MESSAGE]

    app = click(page(), "Not in the documents")

    assert "Declined: not supported by the documents" in badges(app)
    assert FALLBACK_MESSAGE in markdown(app)
    assert not app.main.expander  # no sources for a refusal
    assert app.info[0].value.startswith("Out of retries")


def test_answer_stays_when_the_page_reruns(ready, llm, retriever):
    llm.replies = ["38 minutes [1].", ACCEPT]
    app = click(page(), "Answered directly")

    app.run()  # e.g. after picking a file in the sidebar

    assert "38 minutes [1]." in markdown(app)


def test_removing_uploads_updates_the_document_list(ready, temp_store):
    service.add_document("menu.md", b"# Menu\nFish soup on Fridays.")
    app = page()
    assert app.sidebar.expander[0].label == "3 documents indexed"

    app = click(app, "Remove uploaded documents")

    assert app.sidebar.success[0].value == "Removed 1 document."
    assert app.sidebar.expander[0].label == "2 documents indexed"
    assert not any(b.label == "Remove uploaded documents" for b in app.button)
