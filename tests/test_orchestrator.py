"""
Tests the orchestrator's wiring logic with all agents mocked —
verifies the pipeline calls agents in the right order (including the
parallel JD/resume fan-out), passes data correctly between them, and
runs the critic's revision loop correctly, without touching the real API.
"""
import pytest
from app.orchestrator import run_pipeline
from app.schemas import (
    CritiqueResult,
    JobRequirements,
    ResumeProfile,
    SkillMatch,
    ApplicationDraft,
)


def test_pipeline_wires_agents_correctly(mocker):
    fake_job_req = JobRequirements(job_title="Backend Developer")
    fake_profile = ResumeProfile(name="Miyaz", skills=["Python"])
    fake_match = SkillMatch(match_score=70)
    fake_draft = ApplicationDraft(cover_letter="Dear Hiring Manager...")
    fake_critique = CritiqueResult(verdict="pass")

    mocker.patch("app.orchestrator.parse_job_description", return_value=fake_job_req)
    mocker.patch("app.orchestrator.parse_resume", return_value=fake_profile)
    match_mock = mocker.patch("app.orchestrator.match_resume_to_job", return_value=fake_match)
    mocker.patch("app.orchestrator.draft_application", return_value=fake_draft)
    mocker.patch("app.orchestrator.critique_application", return_value=fake_critique)

    result = run_pipeline("some JD", "some resume", "Miyaz")

    assert result.job_requirements.job_title == "Backend Developer"
    assert result.resume_profile.name == "Miyaz"
    assert result.skill_match.match_score == 70
    assert result.application_draft.cover_letter.startswith("Dear")
    assert result.critique.verdict == "pass"
    assert result.revision_count == 0
    # The matcher should get the structured profile from the parallel
    # fan-out, not just the raw resume text.
    match_mock.assert_called_once_with(fake_job_req, "some resume", resume_profile=fake_profile)


def test_pipeline_runs_jd_and_resume_parsing_in_parallel(mocker):
    """
    Both parse calls should happen even though nothing sequences one
    after the other in the orchestrator code - proves they're
    independent fan-out branches, not a hidden sequential dependency.
    """
    fake_job_req = JobRequirements(job_title="Backend Developer")
    fake_profile = ResumeProfile(name="Miyaz")
    jd_mock = mocker.patch("app.orchestrator.parse_job_description", return_value=fake_job_req)
    resume_mock = mocker.patch("app.orchestrator.parse_resume", return_value=fake_profile)
    mocker.patch("app.orchestrator.match_resume_to_job", return_value=SkillMatch())
    mocker.patch("app.orchestrator.draft_application", return_value=ApplicationDraft())
    mocker.patch("app.orchestrator.critique_application", return_value=CritiqueResult(verdict="pass"))

    run_pipeline("some JD", "some resume")

    jd_mock.assert_called_once_with("some JD")
    resume_mock.assert_called_once_with("some resume")


def test_pipeline_propagates_error_from_either_side_of_fan_out(mocker):
    fake_profile = ResumeProfile(name="Miyaz")
    mocker.patch("app.orchestrator.parse_job_description", side_effect=ValueError("JD parsing failed"))
    mocker.patch("app.orchestrator.parse_resume", return_value=fake_profile)

    with pytest.raises(ValueError, match="JD parsing failed"):
        run_pipeline("some JD", "some resume")


def test_pipeline_revises_when_critic_flags_issues(mocker):
    fake_job_req = JobRequirements(job_title="Backend Developer")
    fake_profile = ResumeProfile(name="Miyaz")
    fake_match = SkillMatch(match_score=70)
    bad_draft = ApplicationDraft(cover_letter="I am an expert in Kubernetes...")
    fixed_draft = ApplicationDraft(cover_letter="Dear Hiring Manager, honest letter...")

    mocker.patch("app.orchestrator.parse_job_description", return_value=fake_job_req)
    mocker.patch("app.orchestrator.parse_resume", return_value=fake_profile)
    mocker.patch("app.orchestrator.match_resume_to_job", return_value=fake_match)
    # First draft call returns the bad draft, the revision call returns the fixed one.
    mocker.patch(
        "app.orchestrator.draft_application",
        side_effect=[bad_draft, fixed_draft],
    )
    # First critique fails (asks for revision), second passes.
    mocker.patch(
        "app.orchestrator.critique_application",
        side_effect=[
            CritiqueResult(verdict="revise", issues=["fabricated Kubernetes claim"], feedback="Remove it."),
            CritiqueResult(verdict="pass"),
        ],
    )

    result = run_pipeline("some JD", "some resume", "Miyaz")

    assert result.revision_count == 1
    assert result.critique.verdict == "pass"
    assert result.application_draft.cover_letter.startswith("Dear Hiring Manager, honest")


def test_pipeline_stops_after_max_revisions_and_warns(mocker):
    fake_job_req = JobRequirements(job_title="Backend Developer")
    fake_profile = ResumeProfile(name="Miyaz")
    fake_match = SkillMatch(match_score=70)
    stubborn_draft = ApplicationDraft(cover_letter="Still not great...")

    mocker.patch("app.orchestrator.parse_job_description", return_value=fake_job_req)
    mocker.patch("app.orchestrator.parse_resume", return_value=fake_profile)
    mocker.patch("app.orchestrator.match_resume_to_job", return_value=fake_match)
    mocker.patch("app.orchestrator.draft_application", return_value=stubborn_draft)
    mocker.patch(
        "app.orchestrator.critique_application",
        return_value=CritiqueResult(verdict="revise", issues=["still generic"], feedback="Be specific."),
    )

    result = run_pipeline("some JD", "some resume", "Miyaz", max_revisions=2)

    assert result.revision_count == 2
    assert result.critique.verdict == "revise"
    assert any("unresolved issues" in w for w in result.guardrail_warnings)