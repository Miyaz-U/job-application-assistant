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
from app.schemas import JobRequirements, ResumeProfile, SkillMatch
from app.guardrails import sanitize_resume, wrap_untrusted, verify_skill_match_grounding

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

_SYSTEM_PROMPT = """You are a resume-to-job matching engine.
You will be given structured job requirements (as JSON), a candidate's raw
resume text, and — when available — that same resume already parsed into
structured fields (skills, experience, education). Compare them carefully
and honestly.

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


def match_resume_to_job(
    job_requirements: JobRequirements,
    resume_text: str,
    resume_profile: ResumeProfile | None = None,
) -> SkillMatch:
    """
    Compares a resume against structured job requirements.

    `resume_profile` is optional: when the orchestrator has already run
    the Resume Parser (Agent) in parallel with the JD Parser, pass its
    output here so the model reasons over structured skills/experience
    instead of only the raw text blob. Grounding still checks claims
    against the raw resume_text either way, since that's the actual
    source of truth.

    Raises:
        GuardrailError (a ValueError subclass): if resume_text is empty,
                    too long, or fails another input guardrail.
        ValueError: if model output is invalid or fails schema validation.
    """
    sanitized = sanitize_resume(resume_text)
    if sanitized.flagged:
        print(f"[guardrails] resume_matcher: suspicious pattern(s) flagged: {sanitized.flagged_reason}")

    job_req_json = job_requirements.model_dump_json(indent=2)

    contents = [
        _SYSTEM_PROMPT,
        f"Job Requirements:\n{job_req_json}",
    ]
    if resume_profile is not None:
        contents.append(f"Structured Resume Profile:\n{resume_profile.model_dump_json(indent=2)}")
    contents.append(wrap_untrusted(sanitized.text, "CANDIDATE RESUME"))

    response = _client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=contents,
    )

    raw_output = _strip_json_fences(response.text)

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output: {raw_output[:200]}"
        ) from e

    try:
        skill_match = SkillMatch(**data)
    except Exception as e:
        raise ValueError(f"Model output didn't match expected schema: {e}") from e

    # Guardrail: don't trust the model's own "matched_skills"/match_score
    # at face value - re-check each claimed match against the actual
    # resume text and demote/rescale anything it can't back up.
    grounded_match, _grounding_report = verify_skill_match_grounding(skill_match, sanitized.text)
    return grounded_match