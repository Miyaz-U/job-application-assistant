import { useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function ScoreGauge({ score }) {
  const radius = 80;
  const circumference = Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, score));
  const filled = (clamped / 100) * circumference;
  const angle = 180 - (clamped / 100) * 180;
  const rad = (angle * Math.PI) / 180;
  const needleX = 100 + 62 * Math.cos(rad);
  const needleY = 100 - 62 * Math.sin(rad);

  return (
    <svg
      className="score-gauge"
      width="200"
      height="118"
      viewBox="0 0 200 118"
      role="img"
      aria-label={`Match score ${clamped} out of 100`}
    >
      <path
        d="M 20 100 A 80 80 0 0 1 180 100"
        fill="none"
        stroke="var(--paper-shadow)"
        strokeOpacity="0.35"
        strokeWidth="12"
        strokeLinecap="round"
      />
      <path
        d="M 20 100 A 80 80 0 0 1 180 100"
        fill="none"
        stroke="var(--moss)"
        strokeWidth="12"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={circumference - filled}
      />
      <line
        x1="100"
        y1="100"
        x2={needleX}
        y2={needleY}
        stroke="var(--ochre)"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
      <circle cx="100" cy="100" r="4" fill="var(--ochre)" />
      <text x="100" y="88" textAnchor="middle" className="gauge-number" fill="var(--text-cream)">
        {clamped}
      </text>
      <text x="100" y="106" textAnchor="middle" className="gauge-of">
        out of 100
      </text>
    </svg>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <svg width="120" height="70" viewBox="0 0 120 70" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M10 60 A50 50 0 0 1 110 60"
          stroke="var(--text-cream-soft)"
          strokeWidth="2"
          strokeDasharray="4 5"
        />
        <rect x="40" y="8" width="40" height="30" rx="2" stroke="var(--text-cream-soft)" strokeWidth="2" strokeDasharray="3 4" />
      </svg>
      <h2>Nothing here yet</h2>
      <p>Add a job description and your resume, then generate an application. Your match score, cover letter, and talking points will show up here.</p>
    </div>
  );
}

export default function App() {
  const [jobDescription, setJobDescription] = useState("");
  const [candidateName, setCandidateName] = useState("");
  const [resumeFile, setResumeFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!jobDescription.trim() || !resumeFile) {
      setError("Add a job description and a resume file before generating an application.");
      return;
    }

    const formData = new FormData();
    formData.append("job_description", jobDescription);
    formData.append("candidate_name", candidateName);
    formData.append("resume_file", resumeFile);

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/generate-application-upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "The server couldn't process that request.");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="masthead">
        <h1>Job Application Assistant</h1>
        <p>Paste the posting, add your resume, and see where you stand before you apply.</p>
      </header>

      <div className="desk">
        <form className="form-card" onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="name">Your name (optional)</label>
            <input
              id="name"
              type="text"
              value={candidateName}
              onChange={(e) => setCandidateName(e.target.value)}
              placeholder="Jordan Lee"
            />
          </div>

          <div className="field">
            <label htmlFor="jd">Job description</label>
            <textarea
              id="jd"
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the full posting here..."
            />
          </div>

          <div className="field">
            <label htmlFor="resume">Resume</label>
            <label className="dropzone" htmlFor="resume">
              <span className="pick-btn">Choose PDF</span>
              <span className={`file-name ${resumeFile ? "chosen" : ""}`}>
                {resumeFile ? resumeFile.name : "No file selected"}
              </span>
              <input
                id="resume"
                type="file"
                accept="application/pdf"
                onChange={(e) => setResumeFile(e.target.files[0])}
              />
            </label>
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? "Reading your materials..." : "Generate application"}
          </button>

          {error && <p className="error-banner">{error}</p>}
        </form>

        <div className="results">
          {!result && <EmptyState />}

          {result && (
            <>
              <div className="score-row">
                <ScoreGauge score={result.match_result.match_score} />
                <div className="score-copy">
                  <h2>Where you stand</h2>
                  <p>{result.match_result.gap_summary}</p>
                </div>
              </div>

              <div className="skill-block">
                <h3>Skills you bring</h3>
                <div className="chip-row">
                  {result.match_result.matched_skills.map((s, i) => (
                    <span className="chip have" key={i}>{s}</span>
                  ))}
                </div>
              </div>

              <div className="skill-block">
                <h3>Skills to address</h3>
                <div className="chip-row">
                  {result.match_result.missing_skills.map((s, i) => (
                    <span className="chip need" key={i}>{s}</span>
                  ))}
                </div>
              </div>

              <p className="letter-heading">
                Draft cover letter
                {result.revision_count > 0 && (
                  <span className="revision-note">
                    {" "}
                    · refined {result.revision_count} {result.revision_count === 1 ? "time" : "times"} by an automatic reviewer
                  </span>
                )}
              </p>
              <div className="letter-card">
                <p className="letter-body">{result.application_draft.cover_letter}</p>
              </div>

              {result.guardrail_warnings.length > 0 && (
                <div className="review-banner">
                  <strong>Give this a look before sending:</strong>
                  <ul>
                    {result.guardrail_warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="list-block">
                <h3>Sharpen your resume</h3>
                <ul>
                  {result.application_draft.resume_recommendations.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>

              <div className="list-block">
                <h3>Talking points for the interview</h3>
                <ul>
                  {result.application_draft.key_talking_points.map((t, i) => (
                    <li key={i}>{t}</li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}