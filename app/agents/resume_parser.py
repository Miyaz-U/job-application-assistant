"""
Agent: Resume Parser.

Takes raw resume text (already extracted from a PDF, or pasted
directly) and returns a structured ResumeProfile: name, contact
details, education, skills, experience, and projects. Uses Gemini
with a strict prompt asking for JSON-only output, then validates
that output against our Pydantic schema so downstream code can
trust its shape.
"""
import json
import re
from google import genai
from app.config import settings
from app.schemas import ResumeProfile

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

_SYSTEM_PROMPT = """You are a resume parsing engine.
Extract structured information from the resume text provided.

Respond with ONLY a valid JSON object, no markdown fences, no preamble,
matching exactly this shape:

{
  "name": "string",
  "email": "string",
  "phone": "string",
  "location": "string",
  "linkedin": "string",
  "portfolio": "string",
  "summary": "string",
  "skills": ["string", ...],
  "education": [
    {
      "institution": "string",
      "degree": "string",
      "field_of_study": "string",
      "start_date": "string",
      "end_date": "string"
    }
  ],
  "experience": [
    {
      "company": "string",
      "title": "string",
      "start_date": "string",
      "end_date": "string",
      "location": "string",
      "description": ["string", ...]
    }
  ],
  "projects": [
    {
      "name": "string",
      "description": "string",
      "technologies": ["string", ...]
    }
  ]
}

Rules:
- If a field isn't present in the resume, use an empty string or empty list.
- Do not invent information that isn't in the text.
- Preserve dates as they appear in the resume (don't reformat them).
- "skills" should be a flat deduplicated list of individual skills/tools/technologies.
"""


def _strip_json_fences(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` even when told not to."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_resume(resume_text: str) -> ResumeProfile:
    """
    Parses raw resume text into a structured ResumeProfile.

    Raises:
        ValueError: if resume_text is empty or the model output can't
                    be parsed/validated.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    response = _client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[_SYSTEM_PROMPT, f"Resume Text:\n{resume_text}"],
    )

    raw_output = _strip_json_fences(response.text)

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output: {raw_output[:200]}"
        ) from e

    try:
        return ResumeProfile(**data)
    except Exception as e:
        raise ValueError(f"Model output didn't match expected schema: {e}") from e