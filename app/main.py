"""
FastAPI entrypoint for the Job Application Assistant.
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.config import settings
from app.agents.jd_parser import parse_job_description
from app.agents.resume_matcher import match_resume_to_job
from app.agents.resume_parser import parse_resume
from app.orchestrator import run_pipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form
from app.utils import extract_text_from_pdf
from app.guardrails import GuardrailError

app = FastAPI(title="Job Application Assistant", version="0.1.0")

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_check():
    settings.validate()


@app.get("/health")
def health_check():
    return {"status": "ok", "env": settings.APP_ENV}


class JDRequest(BaseModel):
    job_description: str


@app.post("/parse-jd")
def parse_jd(request: JDRequest):
    """Debug endpoint: test Agent A in isolation."""
    try:
        result = parse_job_description(request.job_description)
        return result.model_dump()
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ResumeTextRequest(BaseModel):
    resume_text: str


@app.post("/parse-resume")
def parse_resume_endpoint(request: ResumeTextRequest):
    """Debug endpoint: test the resume parser on raw text."""
    try:
        result = parse_resume(request.resume_text)
        return result.model_dump()
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/parse-resume-upload")
async def parse_resume_upload(resume_file: UploadFile = File(...)):
    """Debug endpoint: test the resume parser on an uploaded PDF."""
    try:
        pdf_bytes = await resume_file.read()
        resume_text = extract_text_from_pdf(pdf_bytes)
        result = parse_resume(resume_text)
        return result.model_dump()
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class MatchRequest(BaseModel):
    job_description: str
    resume_text: str


@app.post("/match-resume")
def match_resume(request: MatchRequest):
    """Debug endpoint: test Agents A + B in isolation."""
    try:
        job_requirements = parse_job_description(request.job_description)
        match_result = match_resume_to_job(job_requirements, request.resume_text)
        return {
            "job_requirements": job_requirements.model_dump(),
            "match_result": match_result.model_dump(),
        }
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class DraftRequest(BaseModel):
    job_description: str
    resume_text: str
    candidate_name: str = ""


@app.post("/generate-application")
def generate_application(request: DraftRequest):
    """Main endpoint: runs the full A -> B -> C pipeline."""
    try:
        result = run_pipeline(
            job_description=request.job_description,
            resume_text=request.resume_text,
            candidate_name=request.candidate_name,
        )
        return {
            "job_requirements": result.job_requirements.model_dump(),
            "match_result": result.skill_match.model_dump(),
            "application_draft": result.application_draft.model_dump(),
            "guardrail_warnings": result.guardrail_warnings,
        }
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-application-upload")
async def generate_application_upload(
    job_description: str = Form(...),
    candidate_name: str = Form(""),
    resume_file: UploadFile = File(...),
):
    """Same as /generate-application, but accepts a PDF resume upload instead of raw text."""
    try:
        pdf_bytes = await resume_file.read()
        resume_text = extract_text_from_pdf(pdf_bytes)
        result = run_pipeline(job_description, resume_text, candidate_name)
        return {
            "job_requirements": result.job_requirements.model_dump(),
            "match_result": result.skill_match.model_dump(),
            "application_draft": result.application_draft.model_dump(),
            "guardrail_warnings": result.guardrail_warnings,
        }
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))