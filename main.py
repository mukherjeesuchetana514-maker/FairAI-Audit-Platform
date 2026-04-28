import io
import os
import json
from google import genai
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Gemini configuration — secured via .env
#
# Create a .env file in the same directory as this file and add:
#   GEMINI_API_KEY=your-actual-key-here
#
# The .env file must NEVER be committed to version control.
# Add it to .gitignore immediately:
#   echo ".env" >> .gitignore
# ---------------------------------------------------------------------------

# load_dotenv() reads the .env file and injects its values into os.environ.
# It is a no-op if the file does not exist, so CI/CD environments that inject
# secrets via real environment variables are unaffected.
load_dotenv()

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Create a .env file with GEMINI_API_KEY=your-key, "
        "or export the variable in your shell before starting the server."
    )

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Project FAIR-AI — Fairness Analysis API",
    description=(
        "Uploads a CSV, auto-detects the protected attribute and outcome column, "
        "computes the Disparate Impact Ratio, and returns an AI-generated "
        "Bias Mitigation Report from Gemini."
    ),
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # Explicit localhost origins for local frontend development.
    # Add your production domain here (e.g. "https://fairai.example.com") before deploying.
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",   # React / Next.js dev server
        "http://localhost:5173",   # Vite dev server
        "http://localhost:8080",   # Vue CLI / generic static server
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost",
         "http://127.0.0.1:3000",
        
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOUR_FIFTHS_THRESHOLD = 0.80

# Keywords used to identify protected-attribute columns by header name.
# Checked as case-insensitive substrings against each column name.
PROTECTED_ATTRIBUTE_KEYWORDS: list[str] = [
    "gender",
    "sex",
    "race",
    "ethnicity",
    "nationality",
    "age",
    "disability",
    "religion",
    "marital",
    "orientation",
]

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class GroupStats(BaseModel):
    group_label: str
    total_applicants: int
    total_selected: int
    selection_rate: float


class DisparateImpactResult(BaseModel):
    metric: str
    protected_attribute_used: str
    outcome_column_used: str
    privileged_group: GroupStats
    unprivileged_group: GroupStats
    all_group_rates: dict[str, float]
    disparate_impact_ratio: float
    four_fifths_threshold: float
    bias_detected: bool
    interpretation: str


class UploadResponse(BaseModel):
    status: str
    rows_processed: int
    columns_detected: list[str]
    auto_detected_protected_attribute: str
    auto_detected_outcome_column: str
    fairness_metrics: DisparateImpactResult
    gemini_bias_mitigation_report: str


class TextAuditRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": (
                        "Our AI assistant performs best for users in Western countries "
                        "and may provide less accurate responses for other regions."
                    )
                }
            ]
        }
    }


class BiasFlag(BaseModel):
    bias_type: str        # e.g. "Gender", "Racial", "Cultural"
    severity: str         # "Low" | "Medium" | "High"
    evidence: str         # The specific phrase or pattern that triggered the flag


class TextAuditResponse(BaseModel):
     
    status: str
    is_correct: bool
    analysis_report: str
    corrected_version: str    # The full Gemini plain-English report


# ---------------------------------------------------------------------------
# Step 1 — Auto-detection logic
# ---------------------------------------------------------------------------


def detect_protected_attribute(df: pd.DataFrame) -> str:
    """
    Scans column headers for known protected-attribute keywords (case-insensitive).
    Returns the name of the first matching column.

    Priority follows the order of PROTECTED_ATTRIBUTE_KEYWORDS, then column
    order, so 'gender' is preferred over 'age' when both are present.
    """
    col_lower_map: dict[str, str] = {col.lower(): col for col in df.columns}

    for keyword in PROTECTED_ATTRIBUTE_KEYWORDS:
        for lower_col, original_col in col_lower_map.items():
            if keyword in lower_col:
                return original_col

    raise HTTPException(
        status_code=422,
        detail=(
            "Could not auto-detect a protected attribute column. "
            f"Looked for keywords: {PROTECTED_ATTRIBUTE_KEYWORDS}. "
            f"Columns found in CSV: {df.columns.tolist()}. "
            "Please rename the relevant column so it contains one of the keywords above."
        ),
    )


def detect_outcome_column(df: pd.DataFrame, exclude_col: str) -> str:
    """
    Scans all numeric columns (excluding the already-identified protected
    attribute) for a strictly binary 0/1 distribution.

    To avoid false positives on columns that happen to be binary by accident
    (e.g. a boolean flag), the function also requires that BOTH 0 and 1 are
    present — a column of all-zeros or all-ones carries no signal.
    """
    for col in df.columns:
        if col == exclude_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals == {0, 1}:
            return col

    raise HTTPException(
        status_code=422,
        detail=(
            "Could not auto-detect a binary outcome column. "
            "Ensure the CSV contains exactly one numeric column with only 0 and 1 values "
            f"(excluding '{exclude_col}'). "
            f"Columns found: {df.columns.tolist()}"
        ),
    )


# ---------------------------------------------------------------------------
# Step 2 — Disparate Impact calculation
# ---------------------------------------------------------------------------


def compute_disparate_impact(
    df: pd.DataFrame,
    protected_attribute: str,
    outcome_column: str,
) -> DisparateImpactResult:
    """
    Disparate Impact Ratio = P(Outcome=1 | Unprivileged) / P(Outcome=1 | Privileged)

    Privileged group  → highest selection rate in the data (derived, not assumed).
    Unprivileged group → lowest selection rate (worst-case adverse impact signal).

    A ratio below 0.80 triggers the Four-Fifths Rule (EEOC adverse impact standard).
    """
    # Normalise string columns; leave numeric group keys (e.g. age brackets) as-is
    if pd.api.types.is_string_dtype(df[protected_attribute]):
        df[protected_attribute] = df[protected_attribute].str.strip().str.title()

    # Build per-group stats table
    group_stats: dict[str, dict] = {}
    for group_label, group_df in df.groupby(protected_attribute):
        total = len(group_df)
        selected = int(group_df[outcome_column].sum())
        rate = selected / total if total > 0 else 0.0
        group_stats[str(group_label)] = {
            "group_label": str(group_label),
            "total_applicants": total,
            "total_selected": selected,
            "selection_rate": round(rate, 4),
        }

    if len(group_stats) < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{protected_attribute}' must contain at least 2 distinct groups "
                f"to compute a ratio. Groups found: {list(group_stats.keys())}"
            ),
        )

    privileged_label = max(group_stats, key=lambda g: group_stats[g]["selection_rate"])
    unprivileged_label = min(group_stats, key=lambda g: group_stats[g]["selection_rate"])

    priv = group_stats[privileged_label]
    unpriv = group_stats[unprivileged_label]

    if priv["selection_rate"] == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"All groups in '{protected_attribute}' have a selection rate of 0 — "
                "the Disparate Impact Ratio is mathematically undefined."
            ),
        )

    ratio = round(unpriv["selection_rate"] / priv["selection_rate"], 4)
    bias_detected = ratio < FOUR_FIFTHS_THRESHOLD

    if bias_detected:
        interpretation = (
            f"⚠️  Bias detected. For '{protected_attribute}', the '{unprivileged_label}' group "
            f"receives a positive '{outcome_column}' outcome at only {ratio * 100:.1f}% of the "
            f"rate of the '{privileged_label}' group, falling below the Four-Fifths (80%) "
            "threshold. This constitutes a strong signal of adverse impact under EEOC guidelines."
        )
    else:
        interpretation = (
            f"✅  No adverse impact detected. For '{protected_attribute}', the '{unprivileged_label}' "
            f"group receives a positive '{outcome_column}' outcome at {ratio * 100:.1f}% of the "
            f"rate of the '{privileged_label}' group, meeting or exceeding the Four-Fifths (80%) threshold."
        )

    all_group_rates = {
        label: stats["selection_rate"] for label, stats in group_stats.items()
    }

    return DisparateImpactResult(
        metric="Disparate Impact Ratio",
        protected_attribute_used=protected_attribute,
        outcome_column_used=outcome_column,
        privileged_group=GroupStats(**priv),
        unprivileged_group=GroupStats(**unpriv),
        all_group_rates=all_group_rates,
        disparate_impact_ratio=ratio,
        four_fifths_threshold=FOUR_FIFTHS_THRESHOLD,
        bias_detected=bias_detected,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Step 3 — Gemini Bias Mitigation Report
# ---------------------------------------------------------------------------


def generate_bias_report(metrics: DisparateImpactResult) -> str:
    """
    Serialises the computed fairness metrics to JSON and passes them to
    Gemini with a structured prompt. Returns a 3-sentence plain-English
    Bias Mitigation Report written from the perspective of an AI Ethics Officer.
    """
    metrics_json = json.dumps(metrics.model_dump(), indent=2)

    prompt = f"""
You are a senior AI Ethics Officer preparing a formal Bias Mitigation Report
for a non-technical executive audience.

You have been provided with the following Disparate Impact Analysis results
computed from a real dataset:

{metrics_json}

Write a concise Bias Mitigation Report in EXACTLY 3 sentences:
1. Sentence 1 — Summarise the disparity in plain English: which group is
   disadvantaged, by how much, and what the metric shows.
2. Sentence 2 — Explain the most likely root cause of this discrepancy
   (e.g. historical bias in training data, proxy variables, structural
   inequality in the pipeline).
3. Sentence 3 — Recommend one concrete, actionable remediation step the
   organisation should take immediately to reduce the bias.

Use clear, professional language. Do not use bullet points or headers.
Do not repeat the raw numbers beyond what is necessary. Output only the
3-sentence report — no preamble, no sign-off.
""".strip()

    try:
        response = gemini_client.models.generate_content(
        model='gemini-flash-latest',
        contents=prompt
        )
        return response.text.strip()
    except Exception as exc:
        # Surface Gemini errors gracefully — the math results are still valid
        raise HTTPException(
            status_code=502,
            detail=(
                f"Fairness metrics were computed successfully, but the Gemini API "
                f"returned an error while generating the Bias Mitigation Report: {exc}. "
                "Check that your GEMINI_API_KEY is valid and has not exceeded its quota."
            ),
        )


def generate_text_audit(text: str) -> TextAuditResponse:
    prompt = f"""
You are an expert Automated Fact-Checker and Logic Validator.
Your job is to analyze the text, statement, or mathematical equation below and determine if it is factually and logically correct.

INPUT TO AUDIT:
\"\"\"
{text}
\"\"\"

CRITICAL INSTRUCTION: You MUST respond with ONLY a valid JSON object matching the exact schema below. Do not use markdown formatting outside the JSON.

{{
  "is_correct": <boolean: true if the input is entirely correct, false if there are factual, mathematical, or logical errors>,
  "analysis_report": "<string: A clear 2-to-3 sentence explanation of why the input is correct or incorrect. Detail the exact error if one exists.>",
  "corrected_version": "<string: If incorrect, provide the exact corrected sentence, fact, or mathematical solution. If it is already correct, output 'The input is accurate as provided.'>"
}}
""".strip()

    try:
        response = gemini_client.models.generate_content(
            model='gemini-flash-latest',
            contents=prompt
        )
        raw = response.text.strip()
        print(f"--- GEMINI RAW RESPONSE ---\n{raw}\n---------------------------")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        return TextAuditResponse(
            status="success",
            is_correct=parsed.get("is_correct", False),
            analysis_report=parsed.get("analysis_report", "No analysis returned."),
            corrected_version=parsed.get("corrected_version", "No correction provided.")
        )

    except json.JSONDecodeError as exc:
        print(f"JSON Parse Error. Raw text was: {raw}")
        return TextAuditResponse(
            status="partial_success",
            is_correct=False,
            analysis_report=f"Formatting Error: The AI responded with non-standard text: {raw}",
            corrected_version="N/A"
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error during text audit: {exc}")


# ---------------------------------------------------------------------------
# Endpoint — Text Audit
# ---------------------------------------------------------------------------


@app.post(
    "/audit-text",
    response_model=TextAuditResponse,
    summary="Submit raw text for generative AI bias analysis",
)
async def audit_text(request: TextAuditRequest):
    """
    Accepts a raw string of text (e.g. a GenAI-generated response, a job
    description, a policy document) and returns a structured bias audit.

    **What is detected:**
    - Gender bias (stereotyping, exclusionary language, pronoun assumptions)
    - Racial & ethnic bias (implicit associations, coded language)
    - Cultural bias (Western-centric framing, exclusion of non-dominant groups)
    - Socioeconomic bias (class assumptions, accessibility gaps)
    - Age bias (ageist language or assumptions)

    **Response includes:**
    - A structured list of `bias_flags` with type, severity, and evidence
    - A plain-English `audit_report` from the AI Ethics auditor persona
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="The 'text' field is required and cannot be empty.",
        )

    if len(request.text) > 50_000:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Text is too long ({len(request.text):,} characters). "
                "Maximum allowed is 50,000 characters per request."
            ),
        )

    return generate_text_audit(request.text)


@app.post(
    "/upload-data",
    response_model=UploadResponse,
    summary="Upload a CSV — columns are auto-detected, metrics computed, AI report generated",
)
async def upload_data(
    file: UploadFile = File(..., description="CSV dataset to analyse."),
):
    """
    A fully automated fairness analysis pipeline — no column names required.

    **Pipeline:**
    1. Parse the uploaded CSV.
    2. **Auto-detect** the protected attribute column (keyword scan on headers).
    3. **Auto-detect** the binary outcome column (0/1 value scan).
    4. Compute the **Disparate Impact Ratio** via pandas.
    5. Send the results to **Gemini** for a plain-English Bias Mitigation Report.
    6. Return all data — math metrics + AI report — as a single JSON response.

    **CSV requirements:**
    - At least one column whose name contains a protected-attribute keyword
      (e.g. `gender`, `race`, `age`, `ethnicity`, `sex`).
    - At least one numeric column containing only `0` and `1` (the outcome).
    """

    # --- Validate file type ---
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a valid .csv file.",
        )

    # --- Parse CSV ---
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the CSV file: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail="The uploaded CSV is empty.",
        )

    # --- Step 1: Auto-detect columns ---
    protected_attribute = detect_protected_attribute(df)
    outcome_column = detect_outcome_column(df, exclude_col=protected_attribute)

    # --- Clean key columns ---
    df = df.dropna(subset=[protected_attribute, outcome_column])
    df[outcome_column] = df[outcome_column].astype(int)

    # --- Step 2: Compute Disparate Impact ---
    fairness_metrics = compute_disparate_impact(
        df,
        protected_attribute=protected_attribute,
        outcome_column=outcome_column,
    )

    # --- Step 3: Generate Gemini report ---
    bias_report = generate_bias_report(fairness_metrics)

    return UploadResponse(
        status="success",
        rows_processed=len(df),
        columns_detected=df.columns.tolist(),
        auto_detected_protected_attribute=protected_attribute,
        auto_detected_outcome_column=outcome_column,
        fairness_metrics=fairness_metrics,
        gemini_bias_mitigation_report=bias_report,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "service": "Project FAIR-AI Backend — Dual Audit System",
        "version": "5.0.0",
        "endpoints": {
            "tabular_audit": "POST /upload-data",
            "text_audit":    "POST /audit-text",
        },
        "gemini_key_configured": True,  # Server would not have started if key was absent
    }