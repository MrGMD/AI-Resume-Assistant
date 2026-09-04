import io
import os
import re
from typing import List
from html import escape

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    from docx import Document
except ImportError:
    Document = None


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.6-flash"

MAX_FILE_MB = 10
MAX_TEXT_CHARS = 120_000


st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL THEME
   ========================================================= */

:root {
    --primary: #4f46e5;
    --primary-dark: #3730a3;
    --primary-light: #eef2ff;

    --text: #111827;
    --text-secondary: #64748b;
    --muted: #94a3b8;

    --background: #f8fafc;
    --surface: #ffffff;
    --border: #e2e8f0;

    --success: #059669;
    --success-bg: #ecfdf5;

    --warning: #d97706;
    --warning-bg: #fffbeb;

    --danger: #dc2626;
    --danger-bg: #fef2f2;

    --shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes ringGrow {
    from { stroke-dashoffset: 339.292; }
}

@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--background) !important;
}

.block-container {
    max-width: 1240px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown label,
[data-testid="stText"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stFileUploader"] label {
    color: var(--text) !important;
}

h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }

/* =========================================================
   HERO
   ========================================================= */

.hero-container {
    background:
        radial-gradient(circle at top right, rgba(99, 102, 241, 0.28), transparent 38%),
        radial-gradient(circle at bottom left, rgba(56, 189, 248, 0.14), transparent 45%),
        linear-gradient(135deg, #0f172a 0%, #172554 50%, #312e81 100%);
    border-radius: 24px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
    color: white !important;
    position: relative;
    overflow: hidden;
    animation: fadeSlideIn 0.5s ease-out;
}

.hero-container h1 {
    color: white !important;
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin: 0;
}

.hero-container p {
    color: #cbd5e1 !important;
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 780px;
    margin-top: 0.8rem;
    margin-bottom: 0;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: #e0e7ff !important;
    border-radius: 999px;
    padding: 0.45rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 1rem;
    backdrop-filter: blur(6px);
}

/* =========================================================
   INPUT CARDS
   ========================================================= */

.input-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.25rem;
    box-shadow: var(--shadow);
    height: 100%;
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}

.input-card:hover {
    box-shadow: 0 12px 34px rgba(15, 23, 42, 0.10);
    transform: translateY(-2px);
}

.input-card-title {
    font-size: 1.05rem;
    font-weight: 750;
    color: var(--text) !important;
    margin-bottom: 0.35rem;
}

.input-card-subtitle {
    font-size: 0.86rem;
    color: var(--text-secondary) !important;
    margin-bottom: 1rem;
}

/* =========================================================
   FILE UPLOADER / TEXT AREA
   ========================================================= */

[data-testid="stFileUploader"] {
    background: #f8fafc !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
    background: var(--primary-light) !important;
}

[data-testid="stFileUploader"] section { background: transparent !important; }
[data-testid="stFileUploaderDropzone"] { background: #f8fafc !important; }

.stTextArea textarea {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 14px !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
}

.stTextArea textarea::placeholder { color: #94a3b8 !important; }

/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    border-radius: 12px !important;
    min-height: 46px !important;
    font-weight: 750 !important;
    border: 1px solid #cbd5e1 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.10);
    filter: brightness(1.03);
}

.stButton > button:active { transform: translateY(0px) scale(0.99); }

/* =========================================================
   SCORE CARDS
   ========================================================= */

.score-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.15rem;
    min-height: 132px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeSlideIn 0.45s ease-out both;
}

.score-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.10);
}

.score-label {
    color: #64748b !important;
    font-size: 0.73rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.score-number {
    margin-top: 0.4rem;
    font-size: 2rem;
    font-weight: 850;
    letter-spacing: -0.04em;
}

.score-track {
    height: 6px;
    width: 100%;
    background: #e2e8f0;
    border-radius: 999px;
    margin-top: 0.8rem;
    overflow: hidden;
}

.score-fill {
    height: 100%;
    border-radius: 999px;
    width: 0%;
    transition: width 1s cubic-bezier(.22,1,.36,1);
}

/* =========================================================
   OVERALL SCORE (ring)
   ========================================================= */

.overall-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.75rem 1.75rem;
    box-shadow: var(--shadow);
    margin-top: 1.25rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
    animation: fadeSlideIn 0.5s ease-out both;
}

.ring-wrap { position: relative; width: 132px; height: 132px; flex-shrink: 0; }

.ring-wrap svg { transform: rotate(-90deg); }

.ring-bg { fill: none; stroke: #eef1f6; stroke-width: 12; }

.ring-fill {
    fill: none;
    stroke-width: 12;
    stroke-linecap: round;
    stroke-dasharray: 339.292;
    animation: ringGrow 1.2s cubic-bezier(.22,1,.36,1) both;
}

.ring-center {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}

.ring-score { font-size: 1.75rem; font-weight: 900; letter-spacing: -0.03em; }
.ring-max { font-size: 0.68rem; color: var(--muted) !important; font-weight: 700; }

.overall-title {
    color: #111827 !important;
    font-size: 1.1rem;
    font-weight: 800;
}

.overall-description {
    color: #64748b !important;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-top: 0.5rem;
    max-width: 560px;
}

.overall-pill {
    display: inline-block;
    margin-top: 0.6rem;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.03em;
}

/* =========================================================
   RESULT CARDS
   ========================================================= */

.result-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    animation: fadeSlideIn 0.4s ease-out both;
}

.result-card:hover {
    transform: translateX(3px);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

.result-card.card-danger  { border-left-color: var(--danger); }
.result-card.card-success { border-left-color: var(--success); }
.result-card.card-warning { border-left-color: var(--warning); }
.result-card.card-neutral { border-left-color: var(--primary); }

.result-card-title {
    color: #111827 !important;
    font-weight: 800;
    font-size: 0.98rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.result-card-text {
    color: #475569 !important;
    font-size: 0.92rem;
    line-height: 1.65;
}

/* =========================================================
   KEYWORD PILLS
   ========================================================= */

.keyword-pill {
    display: inline-block;
    background: #eef2ff;
    color: #3730a3 !important;
    border: 1px solid #c7d2fe;
    border-radius: 999px;
    padding: 0.4rem 0.75rem;
    margin: 0.22rem;
    font-size: 0.82rem;
    font-weight: 700;
    transition: transform 0.15s ease, background 0.15s ease;
}

.keyword-pill:hover {
    transform: translateY(-2px) scale(1.03);
    background: #e0e7ff;
}

/* =========================================================
   EMPTY STATE
   ========================================================= */

.empty-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 2.4rem;
    text-align: center;
    box-shadow: var(--shadow);
    margin-top: 1.5rem;
    animation: fadeSlideIn 0.5s ease-out both;
}

.empty-icon {
    font-size: 2.8rem;
    margin-bottom: 0.5rem;
}

.empty-title {
    color: #111827 !important;
    font-size: 1.3rem;
    font-weight: 800;
}

.empty-text {
    color: #64748b !important;
    max-width: 650px;
    margin: 0.5rem auto;
    line-height: 1.6;
}

.step-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeSlideIn 0.45s ease-out both;
}

.step-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08);
}

.step-number {
    color: #4f46e5;
    font-weight: 900;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
}

.step-title { font-size: 1.1rem; margin-top: 8px; font-weight: 800; color: var(--text) !important; }

/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] * { color: #111827 !important; }

.sidebar-model-card {
    background: linear-gradient(135deg, #eef2ff, #e0e7ff);
    border: 1px solid #c7d2fe;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 16px;
}

/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] { color: #64748b !important; font-weight: 700 !important; transition: color 0.15s ease; }
button[data-baseweb="tab"][aria-selected="true"] { color: #4f46e5 !important; }
[data-baseweb="tab-highlight"] { background-color: #4f46e5 !important; }

/* =========================================================
   EXPANDERS / ALERTS
   ========================================================= */

[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    margin-bottom: 0.7rem;
    transition: box-shadow 0.2s ease;
}

[data-testid="stExpander"]:hover { box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
[data-testid="stExpander"] summary p { color: #111827 !important; font-weight: 700 !important; }

[data-testid="stAlert"] { border-radius: 14px !important; }

/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .hero-container { padding: 1.5rem; border-radius: 18px; }
    .hero-container h1 { font-size: 2rem; }
    .hero-container p { font-size: 0.93rem; }
    .score-card { min-height: 110px; }
    .overall-card { flex-direction: column; text-align: center; }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA MODELS
# ============================================================

class CategoryScore(BaseModel):
    score: int = Field(ge=0, le=100)
    feedback: str


class ResumeAnalysis(BaseModel):
    summary: str
    ats_readiness: int = Field(ge=0, le=100)
    keyword_alignment: CategoryScore
    formatting: CategoryScore
    experience_impact: CategoryScore
    skills: CategoryScore
    completeness: CategoryScore
    strengths: List[str]
    critical_issues: List[str]
    improvements: List[str]
    missing_keywords: List[str]
    suggested_bullet_rewrites: List[str]
    recommended_sections: List[str]


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str | None:
    try:
        secret = st.secrets.get("GEMINI_API_KEY")
        if secret:
            return str(secret).strip()
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_pdf(file_bytes: bytes) -> str:
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is not installed.")

    parts = []
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            for page in pdf:
                text = page.get_text("text")
                if text:
                    parts.append(text)
    except Exception as exc:
        raise RuntimeError(f"Could not read PDF: {exc}") from exc

    return "\n".join(parts)


def extract_docx(file_bytes: bytes) -> str:
    if Document is None:
        raise RuntimeError("python-docx is not installed.")

    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise RuntimeError(f"Could not read DOCX: {exc}") from exc

    parts = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def extract_resume(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    size_mb = len(data) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:
        raise ValueError(
            f"File is {size_mb:.1f} MB. Please upload a file under {MAX_FILE_MB} MB."
        )

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        text = extract_pdf(data)
    elif filename.endswith(".docx"):
        text = extract_docx(data)
    elif filename.endswith(".txt"):
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError("Unsupported format. Please upload PDF, DOCX, or TXT.")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise ValueError(
            "No readable text was found. If this is a scanned/image-only PDF, "
            "please use a text-based PDF or DOCX."
        )

    return text[:MAX_TEXT_CHARS]


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_prompt(resume_text: str, job_description: str) -> str:
    if job_description.strip():
        jd_block = job_description.strip()
    else:
        jd_block = (
            "No job description was supplied. Evaluate keyword alignment against "
            "common ATS expectations and the resume's stated target role and skills."
        )

    return f"""
You are an expert ATS resume reviewer,
technical recruiter, and career advisor.

Analyze the resume below for ATS readiness.

This is an advisory heuristic score,
not a score from a specific ATS vendor.

Be evidence-based.

Never invent experience, skills, employers,
degrees, metrics, certifications, achievements,
or keywords as if the candidate already has them.

If suggesting a keyword, clearly treat it as
a keyword to consider adding only if truthful.

Evaluate:

1. Keyword alignment
2. Formatting and ATS parseability
3. Experience impact
4. Technical skills
5. Resume completeness
6. Overall ATS readiness

For experience, prioritize:

- action verbs
- measurable outcomes
- specificity
- technical depth
- relevant accomplishments

For formatting, consider:

- standard section headings
- dates
- tables
- text boxes
- headers
- footers
- unusual symbols
- excessive styling

Give realistic scores.

Do not give 90+ unless the resume is genuinely excellent.

Prioritize the highest-impact improvements.

Return concise, actionable feedback.

TARGET JOB DESCRIPTION:

{jd_block}

RESUME TEXT:

-------------------------
{resume_text}
-------------------------
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_resume(resume_text: str, job_description: str, api_key: str) -> ResumeAnalysis:
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=build_prompt(resume_text, job_description),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResumeAnalysis,
            ),
        )
    except Exception as exc:
        error_text = str(exc)

        if "404" in error_text:
            raise RuntimeError(
                f"Gemini model '{MODEL_NAME}' is not available for this API key/project."
            ) from exc
        if "401" in error_text:
            raise RuntimeError("Gemini authentication failed. Please check GEMINI_API_KEY.") from exc
        if "403" in error_text:
            raise RuntimeError("Gemini access was denied. Check your API project permissions.") from exc
        if "429" in error_text:
            raise RuntimeError("Gemini API quota or rate limit reached. Please try again later.") from exc

        raise RuntimeError(f"Gemini request failed: {error_text}") from exc

    if not response.text:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, ResumeAnalysis):
                return parsed
            return ResumeAnalysis.model_validate(parsed)
        raise RuntimeError("Gemini returned an empty response.")

    try:
        return ResumeAnalysis.model_validate_json(response.text)
    except Exception as exc:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, ResumeAnalysis):
                return parsed
            return ResumeAnalysis.model_validate(parsed)
        raise RuntimeError("Could not parse Gemini's response.") from exc


# ============================================================
# UI HELPERS
# ============================================================

def score_color(score: int) -> str:
    if score >= 80:
        return "#059669"
    if score >= 60:
        return "#d97706"
    return "#dc2626"


def score_band_label(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Needs work"
    return "At risk"


def score_card_html(label: str, score: int, delay: float = 0.0) -> str:
    color = score_color(score)
    safe_label = escape(label)

    return f"""
    <div class="score-card" style="animation-delay:{delay}s;">
        <div class="score-label">{safe_label}</div>
        <div class="score-number" style="color:{color};">{score}<span style="font-size:1rem;color:#94a3b8;">/100</span></div>
        <div class="score-track">
            <div class="score-fill" style="background:{color};" data-target="{score}"></div>
        </div>
    </div>
    """


def overall_score_html(score: int, summary_hint: str) -> str:
    color = score_color(score)
    band = score_band_label(score)
    circumference = 339.292
    offset = circumference - (circumference * score / 100)

    return f"""
    <div class="overall-card">
        <div class="ring-wrap">
            <svg width="132" height="132" viewBox="0 0 132 132">
                <circle class="ring-bg" cx="66" cy="66" r="54"></circle>
                <circle class="ring-fill" cx="66" cy="66" r="54"
                    stroke="{color}"
                    style="stroke-dashoffset:{offset:.3f};"></circle>
            </svg>
            <div class="ring-center">
                <div class="ring-score" style="color:{color};">{score}</div>
                <div class="ring-max">/ 100</div>
            </div>
        </div>
        <div>
            <div class="overall-title">Overall ATS Readiness</div>
            <div class="overall-pill" style="background:{color}1a; color:{color};">{escape(band)}</div>
            <div class="overall-description">{escape(summary_hint)}</div>
        </div>
    </div>
    """


def keyword_pills(keywords: List[str]) -> str:
    if not keywords:
        return ""
    return "".join(
        f'<span class="keyword-pill">{escape(item)}</span>' for item in keywords
    )


def result_card(title: str, text: str, icon: str = "•", variant: str = "neutral", delay: float = 0.0) -> str:
    return f"""
    <div class="result-card card-{variant}" style="animation-delay:{delay}s;">
        <div class="result-card-title">{icon} {escape(title)}</div>
        <div class="result-card-text">{escape(text)}</div>
    </div>
    """


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero-container">
        <div class="hero-badge">✦ AI-POWERED RESUME REVIEW</div>
        <h1>📄 Resume ATS Analyzer</h1>
        <p>
            Analyze your resume against ATS best practices,
            identify missing keywords, discover weaknesses,
            and get practical recommendations to improve
            your chances of reaching the recruiter.
        </p>
    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Analyzer Settings")

    st.html(
        """
        <div class="sidebar-model-card">
            <div style="color:#3730a3;font-weight:800;font-size:14px;">
                ✦ Gemini 3.6 Flash
            </div>
            <div style="color:#6366f1;font-size:12px;margin-top:4px;">
                AI-powered resume analysis
            </div>
        </div>
        """
    )

    st.markdown("### Supported files")
    st.markdown("PDF • DOCX • TXT")
    st.markdown(f"Maximum file size: **{MAX_FILE_MB} MB**")

    st.markdown("---")

    st.markdown("### 💡 ATS scoring")
    st.caption(
        "The score is an AI-assisted heuristic. "
        "Different ATS platforms and employers use different ranking systems."
    )

    if st.session_state.get("analysis"):
        st.markdown("---")
        if st.button("🗑️ Start New Analysis", use_container_width=True):
            st.session_state.pop("analysis", None)
            st.session_state.pop("resume_name", None)
            st.rerun()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("## Analyze your resume")

st.html(
    """
    <div style="color:#64748b;margin-top:-10px;margin-bottom:18px;font-size:14px;">
        Upload your resume and optionally provide the job description you are targeting.
    </div>
    """
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.html(
        """
        <div class="input-card">
            <div class="input-card-title">📄 Resume</div>
            <div class="input-card-subtitle">Upload a text-based resume for analysis.</div>
        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Upload resume",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
        help="PDF, DOCX or TXT. Scanned PDFs may not extract correctly.",
    )

    if uploaded_file:
        st.success(f"✓ {uploaded_file.name} is ready")

with right:
    st.html(
        """
        <div class="input-card">
            <div class="input-card-title">💼 Target job</div>
            <div class="input-card-subtitle">Add the job description for more accurate keyword matching.</div>
        </div>
        """
    )

    job_description = st.text_area(
        "Target job description",
        height=180,
        label_visibility="collapsed",
        placeholder="Paste the target job description here...",
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("")

analyze_clicked = st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None,
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze_clicked:
    api_key = get_api_key()

    if not api_key:
        st.error("Gemini API key not found.")
        st.info("Add GEMINI_API_KEY to Streamlit Secrets.")
        st.stop()

    try:
        with st.spinner("Analyzing your resume with Gemini..."):
            resume_text = extract_resume(uploaded_file)

            if len(resume_text) < 100:
                st.warning(
                    "Very little text was extracted. The document may be image-based."
                )

            analysis = analyze_resume(resume_text, job_description, api_key)

            st.session_state["analysis"] = analysis
            st.session_state["resume_name"] = uploaded_file.name

        st.success("Analysis completed successfully.")

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


# ============================================================
# RESULTS
# ============================================================

analysis = st.session_state.get("analysis")

if analysis:
    st.markdown("---")
    st.markdown("## 📊 Resume Analysis")
    st.caption(f"Results for {st.session_state.get('resume_name', 'resume')}")

    # --------------------------------------------------------
    # OVERALL SCORE (animated ring)
    # --------------------------------------------------------

    overall = analysis.ats_readiness
    hint = (
        "This resume is in strong shape for ATS parsing and keyword relevance."
        if overall >= 80
        else "Solid foundation, but a few targeted fixes will meaningfully raise your callback odds."
        if overall >= 60
        else "Several structural or keyword gaps are likely limiting how ATS and recruiters read this resume."
    )
    st.html(overall_score_html(overall, hint))

    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    st.markdown("### Score breakdown")

    cols = st.columns(5, gap="medium")

    scores = [
        ("ATS readiness", analysis.ats_readiness),
        ("Keywords", analysis.keyword_alignment.score),
        ("Formatting", analysis.formatting.score),
        ("Experience", analysis.experience_impact.score),
        ("Completeness", analysis.completeness.score),
    ]

    for i, (col, (label, score)) in enumerate(zip(cols, scores)):
        with col:
            st.html(score_card_html(label, score, delay=i * 0.06))

    # small script to animate the fill bars after render (staggered)
    st.html(
        """
        <script>
        setTimeout(function() {
            const bars = window.parent.document.querySelectorAll('.score-fill');
            bars.forEach(function(bar, i) {
                setTimeout(function() {
                    bar.style.width = bar.dataset.target + '%';
                }, i * 100);
            });
        }, 80);
        </script>
        """
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.markdown("### 🧠 Overall assessment")
    st.html(result_card("Summary", analysis.summary, icon="🧠", variant="neutral"))

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🚨 Priority fixes", "💪 Strengths", "🔑 Keywords & skills", "✍️ Rewrite ideas"]
    )

    # ---------------- PRIORITY FIXES ----------------

    with tab1:
        st.markdown("### Highest-impact improvements")

        if analysis.critical_issues:
            for i, item in enumerate(analysis.critical_issues, 1):
                st.html(
                    result_card(
                        f"Critical issue {i}", item, icon="🚨", variant="danger", delay=i * 0.05
                    )
                )
        else:
            st.success("No critical issues were identified.")

        st.markdown("### Recommended improvements")

        if analysis.improvements:
            for i, item in enumerate(analysis.improvements, 1):
                st.html(
                    result_card(
                        f"{i:02d}  Improvement", item, icon="🛠️", variant="warning", delay=i * 0.05
                    )
                )

    # ---------------- STRENGTHS ----------------

    with tab2:
        st.markdown("### What your resume does well")

        if analysis.strengths:
            for i, item in enumerate(analysis.strengths, 1):
                st.html(
                    result_card("Strength", item, icon="✓", variant="success", delay=i * 0.05)
                )
        else:
            st.info("No specific strengths were identified.")

        st.markdown("### Detailed category analysis")

        detail_cols = st.columns(2, gap="large")

        details = [
            ("🔑 Keyword alignment", analysis.keyword_alignment),
            ("🎨 Formatting", analysis.formatting),
            ("📈 Experience impact", analysis.experience_impact),
            ("🛠️ Skills", analysis.skills),
        ]

        for index, (title, item) in enumerate(details):
            with detail_cols[index % 2]:
                with st.expander(f"{title} — {item.score}/100", expanded=True):
                    st.write(item.feedback)

    # ---------------- KEYWORDS ----------------

    with tab3:
        st.markdown("### 🔑 Missing keywords")

        if analysis.missing_keywords:
            st.caption(
                "Consider adding these only when they truthfully represent your experience."
            )
            st.html(f'<div>{keyword_pills(analysis.missing_keywords)}</div>')
        else:
            st.success("No major missing keywords were identified.")

        st.markdown("### 📑 Recommended sections")

        if analysis.recommended_sections:
            for item in analysis.recommended_sections:
                st.markdown(f"✓ {item}")
        else:
            st.info("No additional sections were recommended.")

    # ---------------- REWRITE IDEAS ----------------

    with tab4:
        st.markdown("### ✍️ Suggested bullet improvements")

        if analysis.suggested_bullet_rewrites:
            for i, item in enumerate(analysis.suggested_bullet_rewrites, 1):
                st.html(
                    result_card(
                        f"Bullet improvement {i}", item, icon="✍️", variant="neutral", delay=i * 0.05
                    )
                )
        else:
            st.info("No bullet rewrites were suggested.")

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown("---")
    st.caption(
        "This analyzer provides an AI-assisted ATS heuristic based on extracted resume "
        "text and, when supplied, the target job description. It cannot reproduce "
        "every employer's ATS ranking algorithm and should not be treated as a hiring "
        "prediction."
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:
    st.html(
        """
        <div class="empty-card">
            <div class="empty-icon">📋</div>
            <div class="empty-title">Ready to improve your resume?</div>
            <div class="empty-text">
                Upload your resume above and optionally add a target job description.
                The AI will analyze your ATS readiness and show exactly where your
                resume can be improved.
            </div>
        </div>
        """
    )

    st.markdown("### How the analyzer works")

    step_cols = st.columns(3, gap="large")

    steps = [
        ("01", "Upload", "Upload your PDF, DOCX, or TXT resume."),
        ("02", "Match", "Add a target job description for keyword matching."),
        ("03", "Improve", "Get an ATS score and actionable recommendations."),
    ]

    for i, (col, (number, title, description)) in enumerate(zip(step_cols, steps)):
        with col:
            st.html(
                f"""
                <div class="step-card" style="animation-delay:{i * 0.1}s;">
                    <div class="step-number">STEP {number}</div>
                    <div class="step-title">{title}</div>
                    <div class="result-card-text" style="margin-top:6px;">{description}</div>
                </div>
                """
            )
