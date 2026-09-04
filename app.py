import io
import os
import re
from typing import List

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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       APP
    -------------------------------------------------------- */

    .stApp {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        margin-bottom: 1.5rem;

        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 55%,
            #374151 100%
        );

        color: white;

        box-shadow:
            0 12px 35px rgba(17, 24, 39, 0.16);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin: 0.6rem 0 0;
        color: #d1d5db;
        font-size: 1.05rem;
        line-height: 1.6;
    }


    /* --------------------------------------------------------
       GENERAL
    -------------------------------------------------------- */

    .small-note {
        color: #6b7280;
        font-size: 0.88rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }


    /* --------------------------------------------------------
       SCORE CARDS
    -------------------------------------------------------- */

    .metric-wrapper {
        width: 100%;
        box-sizing: border-box;
    }

    .metric-card {
        width: 100%;
        min-height: 125px;
        box-sizing: border-box;

        background: #ffffff;

        border: 1px solid #e5e7eb;
        border-radius: 18px;

        padding: 1.15rem 1rem;

        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;

        text-align: center;

        box-shadow:
            0 4px 14px rgba(15, 23, 42, 0.05);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);

        box-shadow:
            0 8px 22px rgba(15, 23, 42, 0.09);
    }

    .metric-label {
        color: #6b7280;

        font-size: 0.76rem;
        font-weight: 700;

        text-transform: uppercase;
        letter-spacing: 0.06em;

        line-height: 1.3;
    }

    .metric-value {
        margin-top: 0.45rem;

        font-size: 2rem;
        line-height: 1;

        font-weight: 800;
        letter-spacing: -0.03em;
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


    /* --------------------------------------------------------
       RESULT BOXES
    -------------------------------------------------------- */

    .result-box {
        background: white;

        border: 1px solid #e5e7eb;
        border-radius: 18px;

        padding: 1.3rem 1.4rem;

        margin-bottom: 1rem;

        box-shadow:
            0 4px 14px rgba(15, 23, 42, 0.04);
    }


    /* --------------------------------------------------------
       FILE AREA
    -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 16px;
    }


    /* --------------------------------------------------------
       BUTTON
    -------------------------------------------------------- */

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 46px;
    }


    /* --------------------------------------------------------
       TABS
    -------------------------------------------------------- */

    button[data-baseweb="tab"] {
        font-weight: 650;
    }


    /* --------------------------------------------------------
       MOBILE
    -------------------------------------------------------- */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 18px;
        }

        .hero h1 {
            font-size: 1.9rem;
        }

        .hero p {
            font-size: 0.95rem;
        }

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
            powered by Google Gemini Flash.
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

        secret = st.secrets.get(
            "GEMINI_API_KEY"
        )

        if secret:

            return str(secret).strip()

    except Exception:

        pass


    # Local environment
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file_bytes: bytes) -> str:

    if pymupdf is None:

        raise RuntimeError(
            "PyMuPDF is not installed. "
            "Please check requirements.txt."
        )

    text_parts = []

    try:

        with pymupdf.open(
            stream=file_bytes,
            filetype="pdf"
        ) as pdf:

            for page in pdf:

                page_text = page.get_text(
                    "text"
                )

                if page_text:

                    text_parts.append(
                        page_text
                    )

    except Exception as exc:

        raise RuntimeError(
            f"Could not read PDF: {exc}"
        ) from exc

    return "\n".join(text_parts)


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file_bytes: bytes) -> str:

    if Document is None:

        raise RuntimeError(
            "python-docx is not installed. "
            "Please check requirements.txt."
        )

    try:

        doc = Document(
            io.BytesIO(file_bytes)
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not read DOCX file: {exc}"
        ) from exc


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


# ============================================================
# RESUME EXTRACTION
# ============================================================

def extract_resume(uploaded_file) -> str:

    data = uploaded_file.getvalue()

    size_mb = len(data) / (
        1024 * 1024
    )

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
You are an expert ATS resume reviewer,
technical recruiter, and career advisor.

Analyze the resume below for ATS readiness.

IMPORTANT:

This is an advisory heuristic score.

It is NOT a score from a specific ATS vendor.

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

If suggesting a keyword, clearly treat it as
a keyword to consider adding only if it is truthful.

============================================================
SCORING GUIDANCE
============================================================

KEYWORD ALIGNMENT

Evaluate how well the resume's:

- skills
- terminology
- technologies
- experience
- role-specific language

match the target job description.

If no job description is provided,
judge general role and skill targeting.

------------------------------------------------------------

FORMATTING

Evaluate ATS parseability, including:

- conventional section headings
- readable dates
- layout simplicity
- tables
- text boxes
- headers
- footers
- unusual symbols
- excessive styling
- inconsistent formatting

Only infer formatting issues when supported
by the extracted resume text.

------------------------------------------------------------

EXPERIENCE IMPACT

Evaluate:

- action verbs
- measurable outcomes
- specificity
- relevance
- accomplishments
- responsibility-only bullets
- technical depth

Prioritize evidence of impact.

------------------------------------------------------------

SKILLS

Evaluate:

- technical/hard skills
- clarity
- organization
- consistency
- relevance
- role-specific technologies

------------------------------------------------------------

COMPLETENESS

Evaluate whether useful resume sections are present,
such as:

- contact/header
- professional summary
- experience
- education
- skills
- dates
- certifications
- projects
- relevant additional sections

------------------------------------------------------------

ATS READINESS

Give an overall score from 0 to 100.

Keep the score realistic.

Do NOT give 90+ unless the resume is genuinely strong.

Prioritize the highest-impact improvements first.

Return concise and actionable feedback.

============================================================
TARGET JOB DESCRIPTION
============================================================

{jd_block}

============================================================
RESUME TEXT
============================================================

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

    try:

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

                response_mime_type="application/json",

                response_schema=ResumeAnalysis,

            ),
        )


    except Exception as exc:

        error_text = str(exc)

        if "404" in error_text:

            raise RuntimeError(
                f"Gemini model '{MODEL_NAME}' "
                "is not available for this API key/project. "
                "Check the current Gemini model available "
                "to your Google AI project."
            ) from exc


        if "401" in error_text:

            raise RuntimeError(
                "Gemini API authentication failed. "
                "Check your GEMINI_API_KEY."
            ) from exc


        if "403" in error_text:

            raise RuntimeError(
                "Gemini API access was denied. "
                "Check your API key, project permissions, "
                "and Gemini API access."
            ) from exc


        if "429" in error_text:

            raise RuntimeError(
                "Gemini API quota/rate limit reached. "
                "Please wait and try again."
            ) from exc


        raise RuntimeError(
            f"Gemini request failed: {error_text}"
        ) from exc


    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if not response.text:

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
            "Gemini returned an empty response."
        )


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

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
            "Could not parse Gemini's structured "
            f"response: {exc}"
        ) from exc


# ============================================================
# SCORE HELPERS
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

    # IMPORTANT:
    # This HTML is rendered using st.html()
    # instead of st.markdown().

    return f"""
    <div class="metric-wrapper">

        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value {cls}">
                {value}/100
            </div>

        </div>

    </div>
    """


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.caption(
        "Powered by Google Gemini 3.6 Flash"
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

        "📄 Upload your resume",

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

        "💼 Target job description (optional)",

        height=190,

        placeholder=(
            "Paste the target job description here "
            "for more relevant keyword and ATS analysis..."
        ),
    )


if uploaded_file:

    st.success(
        f"Ready to analyze: {uploaded_file.name}"
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
            "Gemini API key not found."
        )

        st.info(
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


        st.success(
            "Resume analysis completed successfully."
        )


    except Exception as exc:

        st.error(
            f"Analysis failed: {exc}"
        )

        st.info(
            "If this problem continues, check your "
            "Gemini API key, model availability, API quota, "
            "file type, and Streamlit logs."
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
        f"📊 Analysis for "
        f"{st.session_state.get('resume_name', 'resume')}"
    )


    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    cols = st.columns(
        5,
        gap="medium"
    )


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

            # IMPORTANT:
            # Use st.html() instead of st.markdown()
            # so the HTML cannot appear as plain text.

            st.html(
                metric_card(
                    label,
                    value
                )
            )


    # --------------------------------------------------------
    # OVERALL ASSESSMENT
    # --------------------------------------------------------

    st.markdown(
        "### Overall assessment"
    )

    st.markdown(
        f"""
        <div class="result-box">
            {analysis.summary}
        </div>
        """,
        unsafe_allow_html=True,
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

            st.markdown(
                "#### Critical issues"
            )

            for item in analysis.critical_issues:

                st.error(item)

        else:

            st.success(
                "No critical issues were identified."
            )


        st.markdown(
            "#### Recommended improvements"
        )


        if analysis.improvements:

            for i, item in enumerate(
                analysis.improvements,
                1
            ):

                st.write(
                    f"**{i}.** {item}"
                )

        else:

            st.info(
                "No additional improvements were suggested."
            )


    # ========================================================
    # STRENGTHS
    # ========================================================

    with tab2:

        if analysis.strengths:

            for item in analysis.strengths:

                st.success(item)

        else:

            st.info(
                "No specific strengths were identified."
            )


        st.markdown(
            "#### Detailed category feedback"
        )


        detail_cols = st.columns(
            4,
            gap="medium"
        )


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
                "adding — only if truthful:**"
            )


            # Display each keyword individually
            # instead of one large text string.

            keyword_text = " • ".join(
                analysis.missing_keywords
            )

            st.info(
                keyword_text
            )

        else:

            st.success(
                "No major missing keywords "
                "were identified."
            )


        st.markdown(
            "#### Sections to consider"
        )


        if analysis.recommended_sections:

            for item in analysis.recommended_sections:

                st.write(
                    f"• {item}"
                )

        else:

            st.info(
                "No additional sections were recommended."
            )


    # ========================================================
    # BULLET REWRITES
    # ========================================================

    with tab4:

        if analysis.suggested_bullet_rewrites:

            st.markdown(
                "#### Suggested bullet improvements"
            )


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


    steps = st.columns(
        3,
        gap="large"
    )


    with steps[0]:

        st.markdown(
            "### 1️⃣ Upload"
        )

        st.write(
            "Add a PDF, DOCX, or TXT resume."
        )


    with steps[1]:

        st.markdown(
            "### 2️⃣ Match"
        )

        st.write(
            "Optionally paste the target job description."
        )


    with steps[2]:

        st.markdown(
            "### 3️⃣ Improve"
        )

        st.write(
            "Get an ATS score, issues, missing keywords, "
            "and actionable rewrite ideas."
        )
