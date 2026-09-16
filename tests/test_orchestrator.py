"""
Tests the orchestrator's wiring logic with all three agents mocked —
verifies the pipeline calls agents in the right order and passes
data correctly between them, without touching the real API.
"""
from app.orchestrator import run_pipeline
from app.schemas import JobRequirements, SkillMatch, ApplicationDraft


def test_pipeline_wires_agents_correctly(mocker):
    fake_job_req = JobRequirements(job_title="Backend Developer")
    fake_match = SkillMatch(match_score=70)
    fake_draft = ApplicationDraft(cover_letter="Dear Hiring Manager...")

    mocker.patch("app.orchestrator.parse_job_description", return_value=fake_job_req)
    mocker.patch("app.orchestrator.match_resume_to_job", return_value=fake_match)
    mocker.patch("app.orchestrator.draft_application", return_value=fake_draft)

    result = run_pipeline("some JD", "some resume", "Miyaz")

    assert result.job_requirements.job_title == "Backend Developer"
    assert result.skill_match.match_score == 70
    assert result.application_draft.cover_letter.startswith("Dear")