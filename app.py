import io
import os
import re
from typing import List

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 55%,
            #374151 100%
        );
        color: white;
        box-shadow: 0 12px 35px rgba(17,24,39,.16);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
    }

    .hero p {
        margin: .55rem 0 0;
        color: #d1d5db;
        font-size: 1.05rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 105px;
    }

    .metric-label {
        color: #6b7280;
        font-size: .82rem;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 750;
        margin-top: .25rem;
    }

    .good {
        color: #047857;
    }

    .warn {
        color: #b45309;
    }

    .bad {
        color: #b91c1c;
    }

    .small-note {
        color: #6b7280;
        font-size: .88rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📄 Resume ATS Analyzer</h1>
        <p>
            Upload your resume, optionally add a target job description,
            and get an ATS-style score with practical improvements
            powered by Gemini Flash.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GEMINI STRUCTURED RESPONSE SCHEMA
# ============================================================

class CategoryScore(BaseModel):
    score: int = Field(ge=0, le=100)
    feedback: str


class ResumeAnalysis(BaseModel):
    summary: str

    ats_readiness: int = Field(
        ge=0,
        le=100
    )

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

    # Streamlit Cloud Secrets
    try:
        secret = st.secrets.get("GEMINI_API_KEY")

        if secret:
            return str(secret).strip()

    except Exception:
        pass

    # Local environment variable
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_pdf(file_bytes: bytes) -> str:

    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is not installed."
        )

    text_parts = []

    with fitz.open(
        stream=file_bytes,
        filetype="pdf"
    ) as pdf:

        for page in pdf:
            text_parts.append(
                page.get_text("text")
            )

    return "\n".join(text_parts)


def extract_docx(file_bytes: bytes) -> str:

    if Document is None:
        raise RuntimeError(
            "python-docx is not installed."
        )

    doc = Document(
        io.BytesIO(file_bytes)
    )

    parts = []

    # Paragraphs
    for paragraph in doc.paragraphs:

        text = paragraph.text.strip()

        if text:
            parts.append(text)

    # Tables
    for table in doc.tables:

        for row in table.rows:

            cells = [
                cell.text.strip()
                for cell in row.cells
            ]

            if any(cells):
                parts.append(
                    " | ".join(cells)
                )

    return "\n".join(parts)


def extract_resume(uploaded_file) -> str:

    data = uploaded_file.getvalue()

    size_mb = len(data) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:

        raise ValueError(
            f"File is {size_mb:.1f} MB. "
            f"Please upload a file under "
            f"{MAX_FILE_MB} MB."
        )

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):

        text = extract_pdf(data)

    elif filename.endswith(".docx"):

        text = extract_docx(data)

    elif filename.endswith(".txt"):

        text = data.decode(
            "utf-8",
            errors="replace"
        )

    else:

        raise ValueError(
            "Unsupported format. "
            "Please upload PDF, DOCX, or TXT."
        )

    # Clean excessive whitespace
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    ).strip()

    if not text:

        raise ValueError(
            "No readable text was found. "
            "If this is a scanned/image-only PDF, "
            "please use a text-based PDF or DOCX."
        )

    return text[:MAX_TEXT_CHARS]


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_prompt(
    resume_text: str,
    job_description: str
) -> str:

    if job_description.strip():

        jd_block = job_description.strip()

    else:

        jd_block = (
            "No job description was supplied. "
            "Evaluate keyword alignment against "
            "common ATS expectations and the resume's "
            "stated target role and skills."
        )

    return f"""
You are an expert ATS resume reviewer and technical recruiter.

Analyze the resume below for ATS readiness.

IMPORTANT:
This is an advisory heuristic score, not a score from
a specific ATS vendor.

Be evidence-based.

Do NOT invent:
- experience
- skills
- employers
- degrees
- metrics
- certifications
- achievements
- keywords

as if the candidate already has them.

If suggesting a keyword, clearly treat it as a keyword
to consider adding only if it is truthful.

SCORING GUIDANCE:

keyword_alignment:
Evaluate how well the resume's skills, terminology,
and experience match the target job description.

If no job description is provided, judge general
role/skill targeting.

formatting:
Evaluate ATS parseability, conventional headings,
dates, layout simplicity, symbols, tables/text boxes,
headers/footers, and excessive styling when inferable
from extracted text.

experience_impact:
Evaluate action verbs, measurable outcomes,
specificity, relevance, and avoidance of
responsibility-only bullets.

skills:
Evaluate technical/hard skills clarity,
consistency, organization, and relevance.

completeness:
Evaluate whether important resume sections are
present and useful, including:
- contact/header
- summary when useful
- experience
- education
- skills
- dates
- certifications
- projects
- relevant additional sections

ats_readiness:
Give an overall score from 0 to 100.

Keep the score realistic.

Do NOT give 90+ unless the resume is genuinely strong.

Prioritize the highest-impact improvements first.

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

def analyze_resume(
    resume_text: str,
    job_description: str,
    api_key: str
) -> ResumeAnalysis:

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(

        model=MODEL_NAME,

        contents=build_prompt(
            resume_text,
            job_description
        ),

        config=types.GenerateContentConfig(

            temperature=0.2,

            response_mime_type="application/json",

            response_schema=ResumeAnalysis,
        ),
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    try:

        return ResumeAnalysis.model_validate_json(
            response.text
        )

    except Exception as exc:

        parsed = getattr(
            response,
            "parsed",
            None
        )

        if parsed is not None:

            if isinstance(
                parsed,
                ResumeAnalysis
            ):
                return parsed

            return ResumeAnalysis.model_validate(
                parsed
            )

        raise RuntimeError(
            "Could not parse Gemini's "
            f"structured response: {exc}"
        ) from exc


# ============================================================
# UI HELPERS
# ============================================================

def score_class(score: int) -> str:

    if score >= 80:
        return "good"

    if score >= 60:
        return "warn"

    return "bad"


def metric_card(
    label: str,
    value: int
) -> str:

    cls = score_class(value)

    return f"""
    <div class="metric-card">

        <div class="metric-label">
            {label}
        </div>

        <div class="metric-value {cls}">
            {value}/100
        </div>

    </div>
    """


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Settings")

    st.caption(
        "Powered by Google Gemini 2.5 Flash"
    )

    st.info(
        "For production, keep your Gemini API key "
        "in Streamlit Secrets. Never commit it to GitHub."
    )

    st.markdown(
        "**Accepted files:** PDF, DOCX, TXT"
    )

    st.markdown(
        f"**Maximum size:** {MAX_FILE_MB} MB"
    )

    st.markdown("---")

    st.caption(
        "ATS scores are estimates. Different employers "
        "and ATS platforms use different parsing and "
        "ranking rules."
    )


# ============================================================
# INPUT AREA
# ============================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


with left:

    uploaded_file = st.file_uploader(

        "Upload your resume",

        type=[
            "pdf",
            "docx",
            "txt"
        ],

        help=(
            "Text-based PDF, DOCX, or TXT. "
            "Scanned/image-only PDFs may not "
            "extract correctly."
        ),
    )


with right:

    job_description = st.text_area(

        "Target job description (optional)",

        height=190,

        placeholder=(
            "Paste the job description here "
            "for a more relevant keyword and "
            "ATS analysis..."
        ),
    )


if uploaded_file:

    st.success(
        f"Ready: {uploaded_file.name}"
    )


analyze_clicked = st.button(

    "🔍 Analyze Resume",

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

        st.error(
            "Gemini API key not found. "
            "Add GEMINI_API_KEY to Streamlit Secrets "
            "or your local environment."
        )

        st.stop()

    try:

        with st.spinner(
            "Extracting resume and analyzing ATS readiness..."
        ):

            resume_text = extract_resume(
                uploaded_file
            )

            if len(resume_text) < 100:

                st.warning(
                    "Very little text was extracted. "
                    "The document may be image-based "
                    "or empty."
                )

            analysis = analyze_resume(

                resume_text,

                job_description,

                api_key,
            )

            st.session_state["analysis"] = analysis

            st.session_state["resume_name"] = (
                uploaded_file.name
            )

    except Exception as exc:

        st.error(
            f"Analysis failed: {exc}"
        )

        st.info(
            "Check your API key, internet connection, "
            "file type, and Gemini API quota. "
            "Then try again."
        )


# ============================================================
# RESULTS
# ============================================================

analysis = st.session_state.get(
    "analysis"
)


if analysis:

    st.markdown("---")

    st.subheader(
        f"Analysis for "
        f"{st.session_state.get('resume_name', 'resume')}"
    )


    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    cols = st.columns(5)

    scores = [

        (
            "ATS readiness",
            analysis.ats_readiness
        ),

        (
            "Keywords",
            analysis.keyword_alignment.score
        ),

        (
            "Formatting",
            analysis.formatting.score
        ),

        (
            "Experience",
            analysis.experience_impact.score
        ),

        (
            "Completeness",
            analysis.completeness.score
        ),

    ]

    for col, (label, value) in zip(
        cols,
        scores
    ):

        with col:

            st.markdown(
                metric_card(
                    label,
                    value
                ),
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.markdown(
        "### Overall assessment"
    )

    st.write(
        analysis.summary
    )


    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(

        [
            "🚨 Priority fixes",
            "💪 Strengths",
            "🔑 Keywords & skills",
            "✍️ Rewrite ideas",
        ]

    )


    # ========================================================
    # PRIORITY FIXES
    # ========================================================

    with tab1:

        if analysis.critical_issues:

            for item in analysis.critical_issues:

                st.error(item)

        else:

            st.success(
                "No critical issues were identified."
            )


        st.markdown(
            "#### Recommended improvements"
        )


        for i, item in enumerate(
            analysis.improvements,
            1
        ):

            st.write(
                f"**{i}.** {item}"
            )


    # ========================================================
    # STRENGTHS
    # ========================================================

    with tab2:

        for item in analysis.strengths:

            st.success(item)


        st.markdown(
            "#### Detailed category feedback"
        )


        detail_cols = st.columns(4)


        details = [

            (
                "Keyword alignment",
                analysis.keyword_alignment
            ),

            (
                "Formatting",
                analysis.formatting
            ),

            (
                "Experience impact",
                analysis.experience_impact
            ),

            (
                "Skills",
                analysis.skills
            ),

        ]


        for col, (title, item) in zip(
            detail_cols,
            details
        ):

            with col:

                st.markdown(
                    f"**{title}: {item.score}/100**"
                )

                st.write(
                    item.feedback
                )


    # ========================================================
    # KEYWORDS
    # ========================================================

    with tab3:

        if analysis.missing_keywords:

            st.markdown(
                "**Keywords/skills to consider "
                "adding (only if truthful):**"
            )

            st.write(
                " · ".join(
                    analysis.missing_keywords
                )
            )

        else:

            st.success(
                "No major missing keywords "
                "were identified."
            )


        st.markdown(
            "#### Sections to consider"
        )


        for item in analysis.recommended_sections:

            st.write(
                f"• {item}"
            )


    # ========================================================
    # BULLET REWRITES
    # ========================================================

    with tab4:

        if analysis.suggested_bullet_rewrites:

            for item in (
                analysis.suggested_bullet_rewrites
            ):

                st.info(item)

        else:

            st.info(
                "No bullet rewrites were suggested."
            )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        "### Important note"
    )

    st.caption(
        "This analyzer estimates ATS readiness from "
        "extracted resume text and, when supplied, "
        "the target job description. It cannot "
        "reproduce every employer's ATS ranking "
        "algorithm and should not be treated as "
        "a hiring prediction."
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        "### How it works"
    )

    steps = st.columns(3)


    with steps[0]:

        st.markdown(
            "**1. Upload**"
        )

        st.write(
            "Add a PDF, DOCX, or TXT resume."
        )


    with steps[1]:

        st.markdown(
            "**2. Match**"
        )

        st.write(
            "Optionally paste the target job description."
        )


    with steps[2]:

        st.markdown(
            "**3. Improve**"
        )

        st.write(
            "Get a score, issues, missing keywords, "
            "and rewrite ideas."
        )
