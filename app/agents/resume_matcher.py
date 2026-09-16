"""
Agent B: Resume Matcher.

Takes the structured JobRequirements (from Agent A) plus the
user's raw resume text, and produces a SkillMatch report:
what matches, what's missing, and an honest fit assessment.
"""
import json
import re
from google import genai
from app.config import settings
from app.schemas import JobRequirements, SkillMatch

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

_SYSTEM_PROMPT = """You are a resume-to-job matching engine.
You will be given structured job requirements (as JSON) and a candidate's
resume text. Compare them carefully and honestly.

Respond with ONLY a valid JSON object, no markdown fences, no preamble,
matching exactly this shape:

{
  "matched_skills": ["string", ...],
  "missing_skills": ["string", ...],
  "matched_qualifications": ["string", ...],
  "missing_qualifications": ["string", ...],
  "strengths": ["string", ...],
  "gap_summary": "string (1-3 sentences, honest and specific)",
  "match_score": integer 0-100
}

Rules:
- Only count a skill as "matched" if it's reasonably evident in the resume text
  (explicitly mentioned, or clearly demonstrated via project/experience descriptions).
- Do not inflate the match_score to be encouraging — be accurate. A genuinely
  weak match should score low.
- missing_skills should only include skills that appear in required_skills or
  preferred_skills from the job requirements but are absent from the resume.
"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def match_resume_to_job(job_requirements: JobRequirements, resume_text: str) -> SkillMatch:
    """
    Compares a resume against structured job requirements.

    Raises:
        ValueError: if resume_text is empty or model output is invalid.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    job_req_json = job_requirements.model_dump_json(indent=2)

    response = _client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            _SYSTEM_PROMPT,
            f"Job Requirements:\n{job_req_json}",
            f"Candidate Resume:\n{resume_text}",
        ],
    )

    raw_output = _strip_json_fences(response.text)

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output: {raw_output[:200]}"
        ) from e

    try:
        return SkillMatch(**data)
    except Exception as e:
        raise ValueError(f"Model output didn't match expected schema: {e}") from e