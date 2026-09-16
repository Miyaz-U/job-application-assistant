"""
Shared data schemas used across agents.
Keeping these in one place means every agent speaks the
same "language" when passing data to the next one.
"""
from pydantic import BaseModel, Field
from typing import List


class EducationEntry(BaseModel):
    institution: str = Field(default="", description="School/university name")
    degree: str = Field(default="", description="e.g. 'B.Tech', 'M.Sc.'")
    field_of_study: str = Field(default="", description="e.g. 'Computer Science'")
    start_date: str = Field(default="")
    end_date: str = Field(default="", description="e.g. 'May 2024' or 'Present'")


class ExperienceEntry(BaseModel):
    company: str = Field(default="")
    title: str = Field(default="", description="Job title/role")
    start_date: str = Field(default="")
    end_date: str = Field(default="", description="e.g. 'Jan 2023' or 'Present'")
    location: str = Field(default="")
    description: List[str] = Field(default_factory=list, description="Bullet points describing the role")


class ProjectEntry(BaseModel):
    name: str = Field(default="")
    description: str = Field(default="")
    technologies: List[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    name: str = Field(default="", description="Candidate's full name")
    email: str = Field(default="")
    phone: str = Field(default="")
    location: str = Field(default="")
    linkedin: str = Field(default="")
    portfolio: str = Field(default="", description="Personal site, GitHub, or portfolio URL")
    summary: str = Field(default="", description="Resume summary/objective if present")
    skills: List[str] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    experience: List[ExperienceEntry] = Field(default_factory=list)
    projects: List[ProjectEntry] = Field(default_factory=list)


class JobRequirements(BaseModel):
    job_title: str = Field(description="The job title as stated in the JD")
    company: str = Field(default="", description="Company name if mentioned")
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)
    experience_required: str = Field(default="", description="e.g. '2-4 years'")
    location: str = Field(default="")
    raw_summary: str = Field(default="", description="1-2 sentence summary of the role")

class SkillMatch(BaseModel):
    matched_skills: List[str] = Field(default_factory=list, description="Skills present in both JD and resume")
    missing_skills: List[str] = Field(default_factory=list, description="Required/preferred JD skills not found in resume")
    matched_qualifications: List[str] = Field(default_factory=list)
    missing_qualifications: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list, description="Notable resume strengths relevant to this JD")
    gap_summary: str = Field(default="", description="1-3 sentence honest summary of overall fit and gaps")
    match_score: int = Field(default=0, ge=0, le=100, description="Rough overall fit score out of 100")

class ApplicationDraft(BaseModel):
    cover_letter: str = Field(default="", description="Full tailored cover letter text")
    resume_recommendations: List[str] = Field(
        default_factory=list,
        description="Specific, actionable suggestions to improve the resume for this JD"
    )
    key_talking_points: List[str] = Field(
        default_factory=list,
        description="Points to emphasize in an interview for this role"
    )