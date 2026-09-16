"""
Agent C: Drafting Agent.

Takes the structured JobRequirements (Agent A) and SkillMatch
(Agent B) and produces a tailored cover letter plus concrete
resume improvement suggestions.
"""
import json
import re
from google import genai
from app.config import settings
from app.schemas import JobRequirements, SkillMatch, ApplicationDraft
from app.utils import call_with_retry

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

_SYSTEM_PROMPT = """You are a career writing assistant helping a job
candidate prepare their application materials.

You will be given:
1. Structured job requirements (JSON)
2. A skill match/gap analysis (JSON) comparing the candidate's resume to those requirements

Using ONLY the matched skills, strengths, and genuine qualifications provided
(never invent experience the candidate doesn't have), produce:
- A tailored, professional cover letter (3-4 paragraphs) that honestly
  emphasizes the candidate's real matching strengths and addresses the role
- Specific, actionable resume improvement suggestions (not generic advice)
- 3-5 key talking points the candidate could use in an interview

Respond with ONLY a valid JSON object, no markdown fences, no preamble,
matching exactly this shape:

{
  "cover_letter": "string (full text, use \\n\\n between paragraphs)",
  "resume_recommendations": ["string", ...],
  "key_talking_points": ["string", ...]
}

Important: Be honest. Do not fabricate experience or skills the candidate
doesn't have. If there are gaps, the cover letter should focus on genuine
strengths and transferable experience rather than pretending the gap doesn't exist.
"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def draft_application(
    job_requirements: JobRequirements,
    skill_match: SkillMatch,
    candidate_name: str = "",
) -> ApplicationDraft:
    """
    Generates a tailored cover letter and resume recommendations.

    Raises:
        ValueError: if model output is invalid or can't be parsed.
    """
    context = {
        "job_requirements": job_requirements.model_dump(),
        "skill_match": skill_match.model_dump(),
        "candidate_name": candidate_name or "the candidate",
    }

    response = call_with_retry(
        lambda: _client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[
                _SYSTEM_PROMPT,
                f"Context:\n{json.dumps(context, indent=2)}",
            ],
        )
    )

    raw_output = _strip_json_fences(response.text)

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output: {raw_output[:200]}"
        ) from e

    try:
        return ApplicationDraft(**data)
    except Exception as e:
        raise ValueError(f"Model output didn't match expected schema: {e}") from e