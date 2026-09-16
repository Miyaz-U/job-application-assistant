"""
Tests for Agent A (JD Parser) using a mocked Gemini response —
no real API calls, no quota used, fully deterministic.
"""
import json
import pytest
from app.agents import jd_parser


class FakeResponse:
    """Mimics the shape of the real Gemini response object."""
    def __init__(self, text):
        self.text = text


def test_parse_job_description_success(mocker):
    fake_json = json.dumps({
        "job_title": "Backend Developer",
        "company": "Acme Corp",
        "required_skills": ["Node.js", "Express", "MongoDB"],
        "preferred_skills": ["Docker"],
        "responsibilities": ["Build APIs"],
        "qualifications": ["Bachelor's degree"],
        "experience_required": "2+ years",
        "location": "Bangalore",
        "raw_summary": "Backend role focused on Node.js APIs.",
    })

    mocker.patch.object(
        jd_parser._client.models,
        "generate_content",
        return_value=FakeResponse(fake_json),
    )

    result = jd_parser.parse_job_description("Some raw JD text")

    assert result.job_title == "Backend Developer"
    assert "Node.js" in result.required_skills


def test_parse_job_description_empty_input_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        jd_parser.parse_job_description("")


def test_parse_job_description_invalid_json_raises(mocker):
    mocker.patch.object(
        jd_parser._client.models,
        "generate_content",
        return_value=FakeResponse("this is not json"),
    )

    with pytest.raises(ValueError, match="did not return valid JSON"):
        jd_parser.parse_job_description("Some JD text")