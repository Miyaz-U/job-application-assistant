# Job Application Assistant

An AI-powered assistant that helps job seekers tailor their applications. Given a job description and a resume, it parses both, evaluates how well they match, and drafts a tailored cover letter with actionable resume recommendations.

## How it works

The backend runs a small pipeline of independent agents, each with a single responsibility:

```
Resume (PDF/text) ──► Resume Parser ──► ResumeProfile (name, contact, education, skills, experience, projects)

Job description ──► JD Parser ──► JobRequirements
JobRequirements + Resume text ──► Resume Matcher ──► SkillMatch (matched/missing skills, fit score)
JobRequirements + SkillMatch ──► Drafting Agent ──► ApplicationDraft (cover letter, recommendations)
```

- **JD Parser** (`app/agents/jd_parser.py`) — extracts structured requirements from a raw job description.
- **Resume Parser** (`app/agents/resume_parser.py`) — extracts structured candidate info (name, contact details, education, skills, experience, projects) from resume text.
- **Resume Matcher** (`app/agents/resume_matcher.py`) — compares a resume against job requirements and produces an honest fit assessment.
- **Drafting Agent** (`app/agents/drafting_agent.py`) — writes a tailored cover letter and resume recommendations.
- **Orchestrator** (`app/orchestrator.py`) — chains JD Parser → Resume Matcher → Drafting Agent for the main end-to-end flow.

All agents use Google's Gemini API (`google-genai`) and validate model output against Pydantic schemas (`app/schemas.py`) before returning it.

## Tech stack

- **Backend:** Python, FastAPI, Pydantic, `google-genai` (Gemini API), `pypdf`
- **Frontend:** React (Vite)
- **Tests:** pytest

## Project structure

```
app/
  agents/
    jd_parser.py        # Agent A: JD -> JobRequirements
    resume_parser.py     # Agent: resume text/PDF -> ResumeProfile
    resume_matcher.py    # Agent B: JobRequirements + resume -> SkillMatch
    drafting_agent.py    # Agent C: -> ApplicationDraft
  config.py               # Loads and validates environment variables
  schemas.py              # Shared Pydantic models
  orchestrator.py         # Wires Agents A -> B -> C
  main.py                 # FastAPI app and endpoints
frontend/                 # React + Vite frontend
tests/                     # pytest test suite
```

## Prerequisites

- Python 3.10+
- Node.js 18+ (for the frontend)
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

## Backend setup

```bash
# from the project root
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create your `.env` file from the template and add your Gemini API key:

```bash
cp .env.example .env
```

```
GOOGLE_API_KEY=your_gemini_api_key_here
APP_ENV=development
```

Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` by default (already allowed in the backend's CORS config).

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/parse-jd` | Debug: parse a job description in isolation |
| POST | `/parse-resume` | Debug: parse raw resume text into a structured profile |
| POST | `/parse-resume-upload` | Debug: parse an uploaded resume PDF into a structured profile |
| POST | `/match-resume` | Debug: parse JD + match against resume text |
| POST | `/generate-application` | Full pipeline: JD + resume text -> match + cover letter |
| POST | `/generate-application-upload` | Full pipeline, accepting a resume PDF upload |

## Running tests

```bash
pytest
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key. The app fails fast on startup if this is missing. |
| `APP_ENV` | No | `development` (default) or `production`. |

`.env` is already listed in `.gitignore` and should never be committed. See `.env.example` for the expected format.

## Notes

- Free-tier Gemini API usage is rate-limited per model per day. If you hit a quota error, consider switching the `model=` value used in the agent files (e.g. `gemini-2.5-flash` -> `gemini-2.5-flash-lite`) to use a separate quota bucket.
- File uploads (`/parse-resume-upload`, `/generate-application-upload`) require the `python-multipart` package, which is included in `requirements.txt`.