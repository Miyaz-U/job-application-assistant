"""
Orchestrator: wires Agents A, B, and C into a single sequential
pipeline. This is the one place that knows the run order — each
agent itself has no awareness of the others.

Flow:
  raw JD text ─────────────────► Agent A ─────► JobRequirements
  JobRequirements + resume text ─► Agent B ─────► SkillMatch
  JobRequirements + SkillMatch ──► Agent C ─────► ApplicationDraft
"""
from dataclasses import dataclass, field
from app.agents.jd_parser import parse_job_description
from app.agents.resume_matcher import match_resume_to_job
from app.agents.drafting_agent import draft_application
from app.guardrails import verify_no_fabricated_skills
from app.schemas import JobRequirements, SkillMatch, ApplicationDraft


@dataclass
class PipelineResult:
    job_requirements: JobRequirements
    skill_match: SkillMatch
    application_draft: ApplicationDraft
    # Non-fatal guardrail findings surfaced to the caller instead of
    # silently swallowed. Empty list = nothing flagged. main.py decides
    # whether/how to show this to the end user; the pipeline itself
    # never blocks on it, since a false positive here shouldn't stop
    # someone from getting their cover letter.
    guardrail_warnings: list[str] = field(default_factory=list)


def run_pipeline(
    job_description: str,
    resume_text: str,
    candidate_name: str = "",
) -> PipelineResult:
    """
    Runs the full Agent A -> B -> C pipeline.

    Raises:
        GuardrailError (a ValueError subclass): input rejected by a
                    guardrail (empty/too long) before any model call.
        ValueError: propagated from whichever agent fails
                    (bad model output, schema mismatch, etc.)
    """
    job_requirements = parse_job_description(job_description)
    # Grounding against the resume is already applied inside
    # match_resume_to_job() - by the time we get skill_match back here,
    # its matched_skills/match_score have already been fact-checked.
    skill_match = match_resume_to_job(job_requirements, resume_text)
    application_draft = draft_application(job_requirements, skill_match, candidate_name)

    warnings: list[str] = []
    fabricated = verify_no_fabricated_skills(application_draft, skill_match)
    if fabricated:
        warnings.append(
            "Cover letter mentions skill(s) not established in the resume: "
            + ", ".join(fabricated)
        )

    return PipelineResult(
        job_requirements=job_requirements,
        skill_match=skill_match,
        application_draft=application_draft,
        guardrail_warnings=warnings,
    )