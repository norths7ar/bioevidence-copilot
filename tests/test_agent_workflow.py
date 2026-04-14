from bioevidence.agent.workflow import run_workflow
from bioevidence.schemas.query import Query


def test_run_workflow_returns_answer_bundle():
    answer = run_workflow(Query(text="delta"))

    assert answer.answer_text
    assert answer.citations == ()
