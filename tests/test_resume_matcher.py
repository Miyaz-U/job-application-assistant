"""
Tests for Agent B (Resume Matcher) using a mocked Gemini response.
"""
import json
import pytest
from app.agents import resume_matcher
from app.schemas import JobRequirements
from tests.test_jd_parser import FakeResponse


def _sample_job_requirements():
    return JobRequirements(
        job_title="Backend Developer",
        required_skills=["Node.js", "MongoDB"],
        preferred_skills=["Docker"],
    )


def test_match_resume_success(mocker):
    fake_json = json.dumps({
        "matched_skills": ["Node.js"],
        "missing_skills": ["MongoDB", "Docker"],
        "matched_qualifications": [],
        "missing_qualifications": [],
        "strengths": ["Strong JS fundamentals"],
        "gap_summary": "Good foundation but missing database experience.",
        "match_score": 55,
    })

    mocker.patch.object(
        resume_matcher._client.models,
        "generate_content",
        return_value=FakeResponse(fake_json),
    )

    result = resume_matcher.match_resume_to_job(
        _sample_job_requirements(), "Resume text mentioning Node.js"
    )

    assert result.match_score == 55
    assert "Node.js" in result.matched_skills


def test_match_resume_empty_resume_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        resume_matcher.match_resume_to_job(_sample_job_requirements(), "")


def test_match_resume_includes_structured_profile_when_given(mocker):
    from app.schemas import ResumeProfile

    fake_json = json.dumps({
        "matched_skills": ["Node.js"],
        "missing_skills": ["MongoDB"],
        "matched_qualifications": [],
        "missing_qualifications": [],
        "strengths": [],
        "gap_summary": "",
        "match_score": 60,
    })

    mock_generate = mocker.patch.object(
        resume_matcher._client.models,
        "generate_content",
        return_value=FakeResponse(fake_json),
    )

    profile = ResumeProfile(name="Jordan", skills=["Node.js", "React"])
    resume_matcher.match_resume_to_job(
        _sample_job_requirements(), "Resume text mentioning Node.js", resume_profile=profile
    )

    sent_contents = mock_generate.call_args.kwargs["contents"]
    assert any("Structured Resume Profile" in c for c in sent_contents)