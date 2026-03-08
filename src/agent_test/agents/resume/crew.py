"""CrewAI pipeline for the resume fit analyzer.

Four specialist agents run sequentially inside a single Crew:

  1. JD Parser        — Steps 2–3 from the prompt (seniority, ATS keywords,
                        must-haves vs nice-to-haves, implicit expectations)
  2. Resume Analyzer  — Step 4 (gap mapping, red flag detection, ATS alignment)
  3. Scorer           — Step 5 (weighted 0–100 % fit score)
  4. Report Generator — Steps 6–7 (full structured report + action plan)

Call :func:`run_resume_crew` to execute the pipeline and get back
the finished report as a plain string.
"""

from __future__ import annotations

import re

from crewai import Agent as CrewAgent, Crew, Process, Task

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

_JD_PARSER_ROLE = "Job Description Parser"
_JD_PARSER_GOAL = (
    "Extract every piece of structured information from a job description: "
    "seniority level, must-have requirements, nice-to-haves, required years of "
    "experience, technical stack, domain knowledge, soft skills, ATS-critical "
    "keywords, implicit expectations, and culture/environment signals."
)
_JD_PARSER_BACKSTORY = (
    "Expert technical recruiter with 15+ years parsing software and AI engineering "
    "job postings at startups and FAANG. You separate genuine requirements from "
    "boilerplate and know exactly which keywords ATS systems flag."
)

_RESUME_ANALYZER_ROLE = "Resume & Candidate Analyzer"
_RESUME_ANALYZER_GOAL = (
    "Map every requirement extracted from the job description to specific evidence "
    "in the candidate's resume. Classify each requirement as ✅ Strong Match, "
    "⚠️ Partial Match, or ❌ Gap. Check ATS keyword alignment exactly. Identify "
    "red flags such as unexplained employment gaps, frequent job-hopping, "
    "overqualification, or title inflation."
)
_RESUME_ANALYZER_BACKSTORY = (
    "Senior technical interviewer and hiring manager who reviewed thousands of "
    "engineering resumes. Unstated skills are gaps — no assumptions. Direct, "
    "evidence-based, immune to buzzwords."
)

_SCORER_ROLE = "Fit Scorer"
_SCORER_GOAL = (
    "Compute an Overall Fit Score (0–100 %) using a strict weighted rubric: "
    "40 % for core technical skills match, 25 % for years and domain experience "
    "relevance, 20 % for nice-to-haves and bonus qualifications, and 15 % for "
    "soft skills, culture fit, and implicit signals. Apply the correct seniority "
    "threshold calibrated by the JD Parser. Provide a clear numeric score and a "
    "2–3 sentence rationale tied to each rubric component."
)
_SCORER_BACKSTORY = (
    "Data-driven hiring analyst who quantifies candidate-role fit with precision. "
    "Built scoring rubrics used by top-tier engineering teams. Never inflates "
    "scores — accuracy first."
)

_REPORTER_ROLE = "Career Report Generator"
_REPORTER_GOAL = (
    "Produce a complete, formatted job fit analysis report following the exact "
    "Output_Format specification. The report must include all seven sections: "
    "Overall Fit Score with rationale, Strengths, Critical Gaps, ATS Keyword "
    "Analysis, Red Flags, Reality Check, Recommendation directive, and an "
    "itemised Action Plan. Every claim must reference explicit evidence from "
    "either the JD or the resume."
)
_REPORTER_BACKSTORY = (
    "Brutally honest career coach specializing in SWE and AI engineering placements. "
    "You deliver the unvarnished truth most coaches avoid — professional, "
    "constructive, and grounded in evidence."
)

# ---------------------------------------------------------------------------
# Output format template (injected into the reporter task description)
# ---------------------------------------------------------------------------

_OUTPUT_FORMAT = """<div class="fit-report">

  <div class="fit-header">
    <div class="fit-title-row">
      <span class="fit-title-text">🎯 JOB FIT ANALYSIS</span>
      <span class="fit-meta">[Job Title] &nbsp;·&nbsp; Level: [Seniority] &nbsp;·&nbsp; [Company or "Company not stated"]</span>
    </div>
    <div class="fit-score-row">
      <!-- Score class: excellent (≥85), good (70–84), marginal (60–69), poor (<60) -->
      <div class="fit-score-circle [score-class]">[X]<span>%</span></div>
      <p class="fit-rationale">[2–3 sentence rationale referencing the scoring rubric breakdown]</p>
    </div>
  </div>

  <div class="fit-section">
    <div class="fit-section-hdr fit-strengths">✅ Strengths — What Works in Your Favor</div>
    <div class="fit-body">
      <ul>
        <li>[Specific, evidence-backed strength 1]</li>
        <li>[Add more items as needed]</li>
      </ul>
    </div>
  </div>

  <div class="fit-section">
    <div class="fit-section-hdr fit-gaps">🚨 Critical Gaps — What Will Get You Rejected</div>
    <div class="fit-body">
      <ul>
        <li>[Hard gap 1 — specific and direct]</li>
        <li>[Add more items as needed]</li>
      </ul>
    </div>
  </div>

  <div class="fit-section">
    <div class="fit-section-hdr fit-ats">🤖 ATS Keyword Analysis</div>
    <div class="fit-body">
      <p><strong>Keywords present in JD but missing from resume:</strong></p>
      <div class="fit-keyword-chips">
        <span class="fit-keyword">[keyword1]</span>
        <span class="fit-keyword">[keyword2]</span>
      </div>
      <p class="fit-match-level">Resume keyword match estimate: <strong>[Weak / Moderate / Strong]</strong></p>
      <p>[1–2 sentences on likelihood of passing automated screening]</p>
    </div>
  </div>

  <div class="fit-section">
    <div class="fit-section-hdr fit-flags">⚠️ Red Flags</div>
    <div class="fit-body">
      <p>[Red flag signals, or: No significant red flags identified.]</p>
    </div>
  </div>

  <div class="fit-section">
    <div class="fit-section-hdr fit-reality">📊 Reality Check</div>
    <div class="fit-body">
      <p>[Honest interview likelihood and competitive positioning]</p>
    </div>
  </div>

  <div class="fit-section">
    <div class="fit-section-hdr fit-rec">📌 Recommendation</div>
    <div class="fit-body">
      <!-- Add class "active" to the chosen chip only; the other two keep class="fit-rec-chip" -->
      <div class="fit-rec-row">
        <span class="fit-rec-chip">APPLY NOW</span>
        <span class="fit-rec-chip">UPSKILL FIRST</span>
        <span class="fit-rec-chip">LOOK ELSEWHERE</span>
      </div>
      <p>[2–3 sentences explaining the directive]</p>
    </div>
  </div>

  <div class="fit-section">
    <div class="fit-section-hdr fit-action">🗺️ Action Plan</div>
    <div class="fit-body">
      <ol>
        <li>[ATS/resume fixes]</li>
        <li>[Skills to build]</li>
        <li>[Red flag mitigation]</li>
        <li>[Experience or credentials to pursue]</li>
        <li>[If fit is poor — 2–3 alternative role titles; otherwise a forward-looking next step]</li>
      </ol>
    </div>
  </div>

</div>"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_resume_crew(
    llm,
    job_description: str,
    resume_text: str,
    *,
    task_callback=None,
) -> str:
    """Run the four-agent sequential pipeline and return the finished report.

    Parameters
    ----------
    llm:
        A ``crewai.LLM`` instance shared by all four crew agents.
    job_description:
        The full job description text.
    resume_text:
        The candidate's resume or qualifications summary.

    Returns
    -------
    str
        The complete formatted job fit analysis report.
    """

    def _shim(output) -> None:
        """Forward task-completion events to the per-request callback, if any."""
        if task_callback is not None:
            task_callback(output)

    # --- Agent instantiation ---

    jd_parser = CrewAgent(
        role=_JD_PARSER_ROLE,
        goal=_JD_PARSER_GOAL,
        backstory=_JD_PARSER_BACKSTORY,
        llm=llm,
        verbose=False,
    )

    resume_analyzer = CrewAgent(
        role=_RESUME_ANALYZER_ROLE,
        goal=_RESUME_ANALYZER_GOAL,
        backstory=_RESUME_ANALYZER_BACKSTORY,
        llm=llm,
        verbose=False,
    )

    scorer = CrewAgent(
        role=_SCORER_ROLE,
        goal=_SCORER_GOAL,
        backstory=_SCORER_BACKSTORY,
        llm=llm,
        verbose=False,
    )

    reporter = CrewAgent(
        role=_REPORTER_ROLE,
        goal=_REPORTER_GOAL,
        backstory=_REPORTER_BACKSTORY,
        llm=llm,
        verbose=False,
    )

    # --- Task definitions ---

    parse_task = Task(
        description=(
            "Parse the following job description and extract:\n"
            "• Seniority level (Intern / Junior / Mid / Senior / Staff / Principal / Lead)\n"
            "• Must-have requirements vs. nice-to-haves\n"
            "• Required years of experience, technical stack, domain knowledge, soft skills\n"
            "• All ATS-critical keywords (tools, languages, frameworks, methodologies)\n"
            "• Implicit expectations (on-call, cross-functional leadership, ambiguity, etc.)\n"
            "• Culture and environment signals (startup, enterprise, remote, fast-paced, etc.)\n\n"
            "Adjust scoring weights note:\n"
            "  Entry-level → weight potential and foundational skills\n"
            "  Mid-level   → balance technical depth with delivery track record\n"
            "  Senior+     → weight domain expertise, leadership, system-level thinking\n\n"
            "JOB DESCRIPTION:\n"
            "─────────────────\n"
            f"{job_description}"
        ),
        expected_output=(
            "A structured breakdown with clearly labelled sections: Seniority Level, "
            "Must-Have Requirements, Nice-to-Haves, Required Experience, Technical Stack, "
            "ATS Keywords (exact terms), Implicit Expectations, Culture Signals, and "
            "Seniority-Adjusted Scoring Notes."
        ),
        agent=jd_parser,
        callback=_shim,
    )

    analyze_task = Task(
        description=(
            "Using the JD analysis above, evaluate this candidate's resume.\n\n"
            "For every requirement identified by the JD Parser:\n"
            "• Map it to specific evidence in the resume\n"
            "• Classify as ✅ Strong Match / ⚠️ Partial Match / ❌ Gap\n"
            "• Any skill or experience NOT explicitly stated = Gap — no assumptions\n\n"
            "Additionally:\n"
            "• Check ATS keyword alignment: does the resume contain the exact JD terms?\n"
            "• Detect red flags: unexplained gaps, job-hopping (<12 months), "
            "overqualification, title inflation, mismatched trajectory\n\n"
            "CANDIDATE RESUME / QUALIFICATIONS:\n"
            "────────────────────────────────────\n"
            f"{resume_text}"
        ),
        expected_output=(
            "A detailed requirement-by-requirement gap analysis table "
            "(✅/⚠️/❌ for each item), a list of exact ATS keywords present and missing, "
            "and a clearly itemised red flag section."
        ),
        agent=resume_analyzer,
        callback=_shim,
    )

    score_task = Task(
        description=(
            "Using the gap analysis above, compute the Overall Fit Score (0–100 %) "
            "with this exact weighted rubric:\n"
            "  40 % → Core technical skills match\n"
            "  25 % → Years and domain experience relevance\n"
            "  20 % → Nice-to-haves and bonus qualifications\n"
            "  15 % → Soft skills, culture fit, and implicit signals\n\n"
            "Apply seniority-adjusted thresholds from the JD Parser output.\n\n"
            "Scoring guide:\n"
            "  85–100 % → Exceptionally strong fit. Apply immediately.\n"
            "  70–84 %  → Solid fit with minor gaps.\n"
            "  60–69 %  → Marginal fit. Viable only with strong framing.\n"
            "  Below 60 %→ Do not apply yet.\n\n"
            "Provide the numeric score, a per-component breakdown, and a "
            "2–3 sentence overall rationale."
        ),
        expected_output=(
            "A numeric Overall Fit Score (e.g. 73 %), a per-component score breakdown "
            "(technical / experience / nice-to-haves / soft skills), and a concise "
            "2–3 sentence rationale."
        ),
        agent=scorer,
        context=[parse_task],
        callback=_shim,
    )

    report_task = Task(
        description=(
            "Compile all outputs from the JD Parser, Resume Analyzer, and Scorer into "
            "a single complete report following EXACTLY this format:\n"
            f"{_OUTPUT_FORMAT}\n\n"
            "Rules:\n"
            "1. Be brutally honest — do not soften or hedge assessments\n"
            "2. Stay evidence-based — every claim must reference the JD or resume\n"
            "3. Be specific — no generic advice; tie everything to this exact role\n"
            "4. Never assume skills not explicitly stated\n"
            "5. If fit score < 60 %, do NOT recommend applying; tick LOOK ELSEWHERE "
            "or UPSKILL FIRST\n"
            "6. Action Plan must have exactly 5 numbered items\n"
            "7. If no red flags exist, state: 'No significant red flags identified.'\n"
            "8. Output ONLY the raw HTML block starting with <div class=\"fit-report\"> — "
            "no markdown fences, no introductory prose, no trailing comments"
        ),
        expected_output=(
            "A valid HTML block starting with <div class=\"fit-report\"> containing all "
            "eight sections: Fit Score, Strengths, Critical Gaps, ATS Keyword Analysis, "
            "Red Flags, Reality Check, Recommendation, and Action Plan. "
            "No markdown, no extra text outside the HTML block."
        ),
        agent=reporter,
        context=[parse_task, analyze_task],
        callback=_shim,
    )

    # --- Crew assembly ---

    crew = Crew(
        agents=[jd_parser, resume_analyzer, scorer, reporter],
        tasks=[parse_task, analyze_task, score_task, report_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return _strip_fences(str(result))


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap around HTML output."""
    # Strip ```html ... ``` or ``` ... ``` surrounding the whole output
    text = text.strip()
    stripped = re.sub(r'^```[a-zA-Z]*\s*', '', text)
    stripped = re.sub(r'\s*```$', '', stripped)
    return stripped.strip()
