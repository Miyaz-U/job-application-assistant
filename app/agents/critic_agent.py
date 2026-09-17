"""
Critic Agent.

Reviews an ApplicationDraft against the JobRequirements and SkillMatch
it was supposed to be grounded in, and decides whether it's ready to
send or needs another pass.

This sits between Agent C (Drafting) and the guardrails layer:
- guardrails.py catches specific, mechanical failure modes with plain
  Python (substring/token matching for fabricated skills).
- The Critic Agent uses an LLM to catch the subtler failures that
  substring matching misses: a claim that's technically not a listed
  "skill" but still overstates the candidate's experience, a cover
  letter that ignores real gaps instead of addressing them honestly,
  weak/generic content that doesn't actually reflect the JD.

The orchestrator calls this agent, and if verdict == "revise", feeds
its `feedback` back into draft_application() for another attempt
(bounded by max_revisions so a stubborn model can't loop forever).
"""
import json
import re
from google import genai
from app.config import settings
from app.schemas import ApplicationDraft, CritiqueResult, JobRequirements, SkillMatch
from app.utils import call_with_retry

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

_SYSTEM_PROMPT = """You are a strict, detail-oriented editor reviewing a job
application draft before it gets sent to the candidate.

You will be given:
1. Structured job requirements (JSON)
2. A skill match/gap analysis (JSON) — the ONLY source of truth for what the
   candidate has actually demonstrated (matched_skills, strengths, matched_qualifications)
3. The drafted cover letter, resume recommendations, and talking points

Check the draft against these criteria:
- HONESTY: does the cover letter claim any skill, tool, years of experience, or
  qualification that is not present in matched_skills, strengths, or
  matched_qualifications? Paraphrased or implied claims count too, not just
  exact skill names.
- GROUNDING: are the talking points and resume recommendations actually
  supported by the skill match data, not generic filler that could apply to
  any candidate?
- ADDRESSES GAPS HONESTLY: if missing_skills/missing_qualifications is
  non-empty, does the letter avoid pretending those gaps don't exist (e.g. by
  overclaiming elsewhere to compensate)?
- QUALITY: is it professional, specific to this role and company (not
  generic), and free of placeholder text like "[Company Name]" when the
  company name was actually provided?

Respond with ONLY a valid JSON object, no markdown fences, no preamble,
matching exactly this shape:

{
  "verdict": "pass" or "revise",
  "issues": ["string", ...],
  "feedback": "string - if revise, concrete instructions for what to fix; empty string if pass"
}

Be strict but fair: minor stylistic preferences are not grounds for "revise".
Only flag genuine fabrication, ungrounded claims, or clear quality problems.
"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def critique_application(
    job_requirements: JobRequirements,
    skill_match: SkillMatch,
    draft: ApplicationDraft,
) -> CritiqueResult:
    """
    Reviews a drafted application for fabrication, weak grounding, and
    quality issues.

    Raises:
        ValueError: if model output is invalid or can't be parsed.
    """
    context = {
        "job_requirements": job_requirements.model_dump(),
        "skill_match": skill_match.model_dump(),
        "draft": draft.model_dump(),
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
        return CritiqueResult(**data)
    except Exception as e:
        raise ValueError(f"Model output didn't match expected schema: {e}") from e
