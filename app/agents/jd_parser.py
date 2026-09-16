"""
Agent A: JD Parser.

Takes a raw job description string and returns a structured
JobRequirements object. Uses Gemini with a strict prompt asking
for JSON-only output, then validates that output against our
Pydantic schema so downstream agents can trust its shape.
"""
import json
import re
from google import genai
from app.config import settings
from app.schemas import JobRequirements

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

_SYSTEM_PROMPT = """You are a job description analysis engine.
Extract structured information from the job description provided.

Respond with ONLY a valid JSON object, no markdown fences, no preamble,
matching exactly this shape:

{
  "job_title": "string",
  "company": "string",
  "required_skills": ["string", ...],
  "preferred_skills": ["string", ...],
  "responsibilities": ["string", ...],
  "qualifications": ["string", ...],
  "experience_required": "string",
  "location": "string",
  "raw_summary": "string (1-2 sentences)"
}

If a field isn't mentioned in the JD, use an empty string or empty list.
Do not invent information that isn't in the text.
"""


def _strip_json_fences(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` even when told not to."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_job_description(jd_text: str) -> JobRequirements:
    """
    Parses a raw job description into structured requirements.

    Raises:
        ValueError: if jd_text is empty or the model output can't
                    be parsed/validated.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("Job description text cannot be empty.")

    response = _client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[_SYSTEM_PROMPT, f"Job Description:\n{jd_text}"],
    )

    raw_output = _strip_json_fences(response.text)

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output: {raw_output[:200]}"
        ) from e

    try:
        return JobRequirements(**data)
    except Exception as e:
        raise ValueError(f"Model output didn't match expected schema: {e}") from e