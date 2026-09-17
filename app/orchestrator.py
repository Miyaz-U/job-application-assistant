"""
Orchestrator: wires Agents A, B, C, and the Critic into a single
pipeline. This is the one place that knows the run order — each
agent itself has no awareness of the others.

Flow:
  raw JD text ────────► Agent A (JD Parser)  ──┐
                                                  ├──► Agent B (Matcher) ──► Agent C (Drafting) ──► Critic
  resume text ─────────► Resume Parser         ──┘         ▲                        │
                                                              └──── revise (feedback) ┘  (bounded by max_revisions)

JD Parser and Resume Parser take different inputs and don't depend on
each other, so they run in parallel threads instead of one after the
other - both are blocking network calls to Gemini, so this halves the
latency of that part of the pipeline.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from app.agents.jd_parser import parse_job_description
from app.agents.resume_parser import parse_resume
from app.agents.resume_matcher import match_resume_to_job
from app.agents.drafting_agent import draft_application
from app.agents.critic_agent import critique_application
from app.guardrails import verify_no_fabricated_skills
from app.schemas import (
    JobRequirements,
    ResumeProfile,
    SkillMatch,
    ApplicationDraft,
    CritiqueResult,
)

DEFAULT_MAX_REVISIONS = 2


@dataclass
class PipelineResult:
    job_requirements: JobRequirements
    resume_profile: ResumeProfile
    skill_match: SkillMatch
    application_draft: ApplicationDraft
    # Final critique of application_draft. verdict == "revise" here means
    # the pipeline ran out of revision attempts without the critic passing
    # it - the draft is still returned (never block the user from getting
    # *something*), but guardrail_warnings will flag it.
    critique: CritiqueResult
    # How many times the Drafting Agent re-ran based on critic feedback.
    revision_count: int = 0
    # Non-fatal guardrail findings surfaced to the caller instead of
    # silently swallowed. Empty list = nothing flagged. main.py decides
    # whether/how to show this to the end user; the pipeline itself
    # never blocks on it, since a false positive here shouldn't stop
    # someone from getting their cover letter.
    guardrail_warnings: list[str] = field(default_factory=list)


def _parse_jd_and_resume_in_parallel(
    job_description: str, resume_text: str
) -> tuple[JobRequirements, ResumeProfile]:
    """
    Runs Agent A and the Resume Parser concurrently in two threads.

    Both futures are always awaited (via `.result()`) even if the first
    one raises, so a slow/failing call on one side never leaves an
    orphaned thread running past this function - the `with` block joins
    both before returning or raising.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        jd_future = executor.submit(parse_job_description, job_description)
        resume_future = executor.submit(parse_resume, resume_text)

        jd_error = None
        resume_error = None
        job_requirements = None
        resume_profile = None

        try:
            job_requirements = jd_future.result()
        except Exception as e:  # noqa: BLE001 - re-raised below, just need to wait for the other future first
            jd_error = e
        try:
            resume_profile = resume_future.result()
        except Exception as e:  # noqa: BLE001
            resume_error = e

    if jd_error is not None:
        raise jd_error
    if resume_error is not None:
        raise resume_error
    return job_requirements, resume_profile


def run_pipeline(
    job_description: str,
    resume_text: str,
    candidate_name: str = "",
    max_revisions: int = DEFAULT_MAX_REVISIONS,
) -> PipelineResult:
    """
    Runs the full pipeline: JD Parser + Resume Parser (in parallel) ->
    Matcher -> Drafting -> Critic -> revise loop (bounded by max_revisions).

    Raises:
        GuardrailError (a ValueError subclass): input rejected by a
                    guardrail (empty/too long) before any model call.
        ValueError: propagated from whichever agent fails
                    (bad model output, schema mismatch, etc.)
    """
    job_requirements, resume_profile = _parse_jd_and_resume_in_parallel(
        job_description, resume_text
    )

    # Grounding against the resume is already applied inside
    # match_resume_to_job() - by the time we get skill_match back here,
    # its matched_skills/match_score have already been fact-checked.
    # resume_profile gives the matcher structured fields to reason over,
    # on top of the raw resume_text it still grounds claims against.
    skill_match = match_resume_to_job(job_requirements, resume_text, resume_profile=resume_profile)
    application_draft = draft_application(job_requirements, skill_match, candidate_name)

    critique = critique_application(job_requirements, skill_match, application_draft)
    revision_count = 0
    while critique.verdict == "revise" and revision_count < max_revisions:
        application_draft = draft_application(
            job_requirements,
            skill_match,
            candidate_name,
            revision_feedback=critique.feedback,
            previous_draft=application_draft,
        )
        revision_count += 1
        critique = critique_application(job_requirements, skill_match, application_draft)

    warnings: list[str] = []
    if critique.verdict == "revise":
        # Ran out of revision attempts without a clean pass. Don't hide
        # this - surface it so the caller/UI can flag the draft as
        # "review before sending" rather than presenting it as final.
        warnings.append(
            f"Draft still has unresolved issues after {revision_count} "
            "revision attempt(s): " + "; ".join(critique.issues)
        )

    # Belt-and-suspenders: the critic is an LLM call and can miss things
    # a cheap deterministic check catches (or vice versa). Keep both.
    fabricated = verify_no_fabricated_skills(application_draft, skill_match)
    if fabricated:
        warnings.append(
            "Cover letter mentions skill(s) not established in the resume: "
            + ", ".join(fabricated)
        )

    return PipelineResult(
        job_requirements=job_requirements,
        resume_profile=resume_profile,
        skill_match=skill_match,
        application_draft=application_draft,
        critique=critique,
        revision_count=revision_count,
        guardrail_warnings=warnings,
    )