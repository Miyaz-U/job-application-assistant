"""
Tests for the Critic Agent. The Gemini client is mocked so these run
without any real API calls.
"""
import json
import pytest
from app.agents.critic_agent import critique_application
from app.schemas import ApplicationDraft, JobRequirements, SkillMatch


def _fake_response(payload: dict):
    class _Resp:
        text = json.dumps(payload)
    return _Resp()


def test_critique_pass(mocker):
    mocker.patch(
        "app.agents.critic_agent._client.models.generate_content",
        return_value=_fake_response({"verdict": "pass", "issues": [], "feedback": ""}),
    )

    result = critique_application(
        JobRequirements(job_title="Backend Developer"),
        SkillMatch(matched_skills=["Python"], match_score=80),
        ApplicationDraft(cover_letter="Dear Hiring Manager, I bring strong Python experience..."),
    )

    assert result.verdict == "pass"
    assert result.issues == []


def test_critique_revise(mocker):
    mocker.patch(
        "app.agents.critic_agent._client.models.generate_content",
        return_value=_fake_response(
            {
                "verdict": "revise",
                "issues": ["Claims 5 years of Kubernetes experience not present in skill_match"],
                "feedback": "Remove the Kubernetes claim; emphasize matched Python/AWS experience instead.",
            }
        ),
    )

    result = critique_application(
        JobRequirements(job_title="Backend Developer"),
        SkillMatch(matched_skills=["Python", "AWS"], match_score=60),
        ApplicationDraft(cover_letter="With 5 years of Kubernetes experience, I..."),
    )

    assert result.verdict == "revise"
    assert result.feedback != ""
    assert len(result.issues) == 1


def test_critique_raises_on_invalid_json(mocker):
    mocker.patch(
        "app.agents.critic_agent._client.models.generate_content",
        return_value=type("R", (), {"text": "not json"})(),
    )

    with pytest.raises(ValueError):
        critique_application(
            JobRequirements(job_title="Backend Developer"),
            SkillMatch(match_score=50),
            ApplicationDraft(cover_letter="..."),
        )
