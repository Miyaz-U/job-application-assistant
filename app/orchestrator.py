"""
Orchestrator: wires Agents A, B, and C into a single sequential
pipeline. This is the one place that knows the run order — each
agent itself has no awareness of the others.

Flow:
  raw JD text ─────────────────► Agent A ─────► JobRequirements
  JobRequirements + resume text ─► Agent B ─────► SkillMatch
  JobRequirements + SkillMatch ──► Agent C ─────► ApplicationDraft
"""
from dataclasses import dataclass
from app.agents.jd_parser import parse_job_description
from app.agents.resume_matcher import match_resume_to_job
from app.agents.drafting_agent import draft_application
from app.schemas import JobRequirements, SkillMatch, ApplicationDraft


@dataclass
class PipelineResult:
    job_requirements: JobRequirements
    skill_match: SkillMatch
    application_draft: ApplicationDraft


def run_pipeline(
    job_description: str,
    resume_text: str,
    candidate_name: str = "",
) -> PipelineResult:
    """
    Runs the full Agent A -> B -> C pipeline.

    Raises:
        ValueError: propagated from whichever agent fails
                    (invalid input, bad model output, etc.)
    """
    job_requirements = parse_job_description(job_description)
    skill_match = match_resume_to_job(job_requirements, resume_text)
    application_draft = draft_application(job_requirements, skill_match, candidate_name)

    return PipelineResult(
        job_requirements=job_requirements,
        skill_match=skill_match,
        application_draft=application_draft,
    )