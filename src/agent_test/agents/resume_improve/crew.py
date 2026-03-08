"""CrewAI pipeline for the resume improvement agent.

Four specialist agents run sequentially inside a single Crew:

  1. Enhancement Strategist — extracts targets from the Job Fit Analysis report:
     fit score, recommendation directive, strengths to amplify, addressable and
     unresolvable gaps, missing ATS keywords, and red flags to neutralize.
  2. Resume Auditor        — audits the current resume across six dimensions:
     structure/format, summary section, experience bullets, skills section,
     education/certifications, and red flag signals.
  3. Resume Rewriter       — rewrites and enhances the resume based on the
     strategy and audit outputs, applying ATS optimization, summary rewrite,
     bullet enhancement, skills rebuild, and red flag reframing.
  4. Report Generator      — compiles all outputs into a single HTML enhancement
     report following the Output_Format specification.

Call :func:`run_resume_improve_crew` to execute the pipeline and get back
the finished report as a plain string.
"""

from __future__ import annotations

import re

from crewai import Agent as CrewAgent, Crew, Process, Task

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

_STRATEGIST_ROLE = "Resume Enhancement Strategist"
_STRATEGIST_GOAL = (
    "Extract all actionable enhancement targets from the Job Fit Analysis report. "
    "Identify: overall fit score and per-component scoring breakdown, the "
    "Recommendation directive (APPLY NOW / UPSKILL FIRST / LOOK ELSEWHERE), "
    "strengths to amplify in the rewritten resume, critical gaps categorized as "
    "(a) addressable through reframing or (b) unresolvable without new experience, "
    "ATS keywords present in the JD but missing from the resume, and red flags "
    "requiring neutralization or reframing. Flag LOOK ELSEWHERE prominently."
)
_STRATEGIST_BACKSTORY = (
    "Strategic career positioning specialist who translates Job Fit Analysis reports "
    "into precise, executable resume improvement plans. Analytical, concise, and "
    "unsparing about what can and cannot be fixed through resume framing alone."
)

_AUDITOR_ROLE = "Resume Structure Auditor"
_AUDITOR_GOAL = (
    "Audit the candidate's current resume across six dimensions: "
    "(1) Structure and format — is it ATS-friendly, correct length, optimal section order? "
    "(2) Summary section — does it exist and is it targeted to the role? "
    "(3) Experience bullets — strong action verbs, quantified outcomes, ATS keyword alignment? "
    "(4) Skills section — does it include exact JD terms, any outdated or irrelevant skills? "
    "(5) Education and certifications — correctly placed and relevant credentials? "
    "(6) Red flag signals — unexplained employment gaps, short tenures (<12 months), "
    "overqualification markers, or career pivot coherence issues?"
)
_AUDITOR_BACKSTORY = (
    "Senior technical recruiter who has reviewed thousands of software and AI engineering "
    "resumes. Expert at spotting ATS killers, weak bullets, and structural problems that "
    "bury strong candidates. Evidence-based, no assumptions, no sugarcoating."
)

_REWRITER_ROLE = "Resume Rewriter"
_REWRITER_GOAL = (
    "Rewrite and enhance the resume based on the strategy and audit outputs. Apply: "
    "(1) ATS optimization — inject missing keywords naturally into bullets and skills, "
    "mirror exact JD terminology. "
    "(2) Summary rewrite — 3–4 targeted lines leading with years of relevant experience, "
    "core technical identity, and top 2 value drivers; embed 2–3 high-priority ATS keywords. "
    "(3) Bullet enhancement — [Action Verb] + [What You Did] + [Result/Impact] format, "
    "quantified outcomes where inferable, front-loaded by relevance to this role. "
    "(4) Skills rebuild — add all missing ATS-critical skills the candidate legitimately holds, "
    "remove outdated or irrelevant skills, organize into clear categories. "
    "(5) Red flag reframing — short tenures get context tags (Contract/Acquired/Laid Off), "
    "gaps get brief neutral notes, overqualification gets scoped language. "
    "INTEGRITY GUARDRAIL: Never invent skills, tools, degrees, or experiences not in the "
    "original resume. Flag unresolvable gaps explicitly with 'GAP NOTE: [description]'."
)
_REWRITER_BACKSTORY = (
    "Expert technical resume writer specializing in SWE and AI/ML engineering roles. "
    "Transforms buried talent into compelling, ATS-optimized documents. Every word earns "
    "its place. Length discipline is non-negotiable — 1 page under 5 years, 2 pages max."
)

_REPORTER_ROLE = "Enhancement Report Generator"
_REPORTER_GOAL = (
    "Compile all outputs from the Enhancement Strategist, Resume Auditor, and Resume "
    "Rewriter into a single complete HTML report following exactly the Output_Format "
    "specification. All sections must be present. Every claim must be grounded in the "
    "candidate's actual experience — never fabricate skills, metrics, or credentials."
)
_REPORTER_BACKSTORY = (
    "Precision technical writer who assembles structured career coaching reports. "
    "Exacting about format compliance, honest about limitations, and ensures the final "
    "HTML is valid, readable, and grounded in evidence."
)

# ---------------------------------------------------------------------------
# Output format template (injected into the reporter task description)
# ---------------------------------------------------------------------------

_OUTPUT_FORMAT = """<div class="improve-report">

  <div class="improve-header">
    <div class="improve-title-row">
      <span class="improve-title-text">📋 RESUME ENHANCEMENT REPORT</span>
      <span class="improve-meta">[Job Title] &nbsp;·&nbsp; Level: [Seniority] &nbsp;·&nbsp; [Company or "Company not stated"]</span>
    </div>
    <div class="improve-score-row">
      <div class="improve-score-block">
        <span class="improve-score-label">Original Score</span>
        <!-- Score class: excellent (≥85), good (70–84), marginal (60–69), poor (<60) -->
        <span class="improve-score-value [original-score-class]">[X]%</span>
      </div>
      <span class="improve-score-arrow">→</span>
      <div class="improve-score-block">
        <span class="improve-score-label">Projected Score</span>
        <span class="improve-score-value [projected-score-class]">[Y]%</span>
      </div>
    </div>
    <!-- Include the warning div ONLY if original fit score was below 60%: -->
    <!-- <div class="improve-low-score-warning">⚠️ Original fit score below 60% — resume enhancement may have limited impact on application odds. Proceed with realistic expectations.</div> -->
  </div>

  <div class="improve-section">
    <div class="improve-section-hdr improve-audit-hdr">🔍 Resume Audit Summary</div>
    <div class="improve-body">
      <p>[Brief assessment of the current resume's key weaknesses and opportunities across structure, content, ATS alignment, and red flags]</p>
    </div>
  </div>

  <div class="improve-section">
    <div class="improve-section-hdr improve-resume-hdr">✍️ Enhanced Resume</div>
    <div class="improve-body">
      <pre class="improve-resume-text">[Full rewritten resume in clean ATS-friendly plain text.
Use standard CAPS section headers: SUMMARY, EXPERIENCE, SKILLS, EDUCATION, CERTIFICATIONS.
No tables, columns, icons, graphics, or non-ASCII formatting characters.]</pre>
    </div>
  </div>

  <div class="improve-section">
    <div class="improve-section-hdr improve-changelog-hdr">📝 Change Log</div>
    <div class="improve-body">
      <ul>
        <li>→ [Section or bullet changed]: [One-line rationale for the change]</li>
        <li>[Add more items as needed]</li>
      </ul>
    </div>
  </div>

  <div class="improve-section">
    <div class="improve-section-hdr improve-ats-hdr">🤖 ATS Improvement Summary</div>
    <div class="improve-body">
      <p><strong>Keywords added:</strong></p>
      <div class="improve-keyword-chips">
        <span class="improve-keyword">[keyword1]</span>
        <span class="improve-keyword">[keyword2]</span>
      </div>
      <p class="improve-match-level">Keyword match: Before <strong>[Weak/Moderate/Strong]</strong> → After <strong>[Weak/Moderate/Strong]</strong></p>
    </div>
  </div>

  <div class="improve-section">
    <div class="improve-section-hdr improve-gaps-hdr">⚠️ Remaining Gaps (Cannot Be Fixed by Resume Alone)</div>
    <div class="improve-body">
      <ul>
        <li>[Specific skill, certification, or experience gap that cannot be resolved through resume framing — or state: No unresolvable gaps identified.]</li>
        <li>[Add more items as needed]</li>
      </ul>
    </div>
  </div>

  <div class="improve-section">
    <div class="improve-section-hdr improve-upskill-hdr">📚 Upskilling Priorities</div>
    <div class="improve-body">
      <ol>
        <li>[Highest-priority skill or certification to pursue — include a suggested resource, course, or project idea where relevant]</li>
        <li>[Add more items as needed]</li>
      </ol>
    </div>
  </div>

</div>"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_resume_improve_crew(
    llm,
    resume_text: str,
    fit_analysis: str,
    job_description: str,
    *,
    task_callback=None,
) -> str:
    """Run the four-agent sequential pipeline and return the enhancement report.

    Parameters
    ----------
    llm:
        A ``crewai.LLM`` instance shared by all four crew agents.
    resume_text:
        The candidate's current resume text.
    fit_analysis:
        The Job Fit Analysis report from the fit analyzer.
    job_description:
        The original job description text (may be empty if embedded in fit_analysis).

    Returns
    -------
    str
        The complete formatted resume enhancement report.
    """

    def _shim(output) -> None:
        """Forward task-completion events to the per-request callback, if any."""
        if task_callback is not None:
            task_callback(output)

    # --- Agent instantiation ---

    strategist = CrewAgent(
        role=_STRATEGIST_ROLE,
        goal=_STRATEGIST_GOAL,
        backstory=_STRATEGIST_BACKSTORY,
        llm=llm,
        verbose=False,
    )

    auditor = CrewAgent(
        role=_AUDITOR_ROLE,
        goal=_AUDITOR_GOAL,
        backstory=_AUDITOR_BACKSTORY,
        llm=llm,
        verbose=False,
    )

    rewriter = CrewAgent(
        role=_REWRITER_ROLE,
        goal=_REWRITER_GOAL,
        backstory=_REWRITER_BACKSTORY,
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

    # --- Build job description context string ---

    jd_context = (
        f"JOB DESCRIPTION:\n─────────────────\n{job_description}"
        if job_description.strip()
        else "(Job description is embedded in the Job Fit Analysis below — extract it from there.)"
    )

    # --- Task definitions ---

    strategy_task = Task(
        description=(
            "You have a Job Fit Analysis report and a job description.\n\n"
            "Extract the following enhancement targets:\n"
            "• Overall fit score and per-component scoring breakdown\n"
            "• Recommendation directive — APPLY NOW / UPSKILL FIRST / LOOK ELSEWHERE\n"
            "  → If LOOK ELSEWHERE, flag this prominently in your output and note that the "
            "report generator must include a low-score disclaimer\n"
            "• Strengths to amplify in the rewritten resume\n"
            "• Critical gaps — categorize as: (a) addressable through resume reframing, "
            "(b) unresolvable without new skills, experience, or credentials\n"
            "• ATS keywords present in the JD but missing from the resume\n"
            "• Red flags to neutralize (employment gaps, short tenures, overqualification, "
            "career pivots)\n\n"
            f"{jd_context}\n\n"
            "JOB FIT ANALYSIS:\n──────────────────\n"
            f"{fit_analysis}"
        ),
        expected_output=(
            "A structured enhancement strategy with clearly labelled sections: "
            "Fit Score & Breakdown, Recommendation Directive (flag LOOK ELSEWHERE "
            "prominently), Strengths to Amplify, Addressable Gaps, Unresolvable Gaps, "
            "Missing ATS Keywords, and Red Flags to Neutralize."
        ),
        agent=strategist,
        callback=_shim,
    )

    audit_task = Task(
        description=(
            "Audit the following resume across six dimensions:\n\n"
            "1. STRUCTURE & FORMAT — ATS-friendly? Correct length for experience level? "
            "Optimal section order?\n"
            "2. SUMMARY SECTION — Does it exist? Is it targeted to the role?\n"
            "3. EXPERIENCE BULLETS — Strong action verbs? Quantified outcomes? "
            "ATS keyword alignment?\n"
            "4. SKILLS SECTION — Exact JD terms present? Any outdated or irrelevant "
            "skills diluting focus?\n"
            "5. EDUCATION & CERTIFICATIONS — Correctly placed, relevant credentials?\n"
            "6. RED FLAG SIGNALS — Unexplained gaps, short tenures, overqualification "
            "markers, career pivot coherence?\n\n"
            "CURRENT RESUME:\n─────────────────\n"
            f"{resume_text}"
        ),
        expected_output=(
            "An audit report with specific findings for each of the six dimensions: "
            "Structure/Format, Summary, Experience Bullets, Skills Section, "
            "Education/Certs, and Red Flags. For each dimension, list specific "
            "weaknesses found and concrete opportunities to improve."
        ),
        agent=auditor,
        callback=_shim,
    )

    rewrite_task = Task(
        description=(
            "Using the enhancement strategy and resume audit above, rewrite the "
            "candidate's resume to maximize its effectiveness for the target role.\n\n"
            "Apply ALL of the following enhancements:\n"
            "• ATS OPTIMIZATION: Inject missing keywords naturally; mirror exact JD "
            "terminology (e.g. 'Machine Learning' vs 'ML' — use whichever the JD uses)\n"
            "• SUMMARY REWRITE: 3–4 targeted lines; lead with years of relevant experience, "
            "core technical identity, and top 2 value drivers; embed 2–3 ATS keywords\n"
            "• BULLET ENHANCEMENT: [Action Verb] + [What You Did] + [Result/Impact]; "
            "quantify outcomes where clearly inferable; front-load most relevant bullets; "
            "remove or condense bullets irrelevant to this role\n"
            "• SKILLS REBUILD: Add all missing ATS-critical skills the candidate legitimately "
            "holds; remove outdated/irrelevant skills; organize by category\n"
            "• RED FLAG REFRAMING: Short tenures → add context tags (Contract/Acquired/"
            "Laid Off/Project-Based); gaps → brief neutral note; overqualification → "
            "scope language to match role level\n\n"
            "INTEGRITY GUARDRAIL: Never invent skills, tools, degrees, or experiences "
            "not present in the original resume. Flag unresolvable gaps with "
            "'GAP NOTE: [description]'\n\n"
            "CURRENT RESUME:\n─────────────────\n"
            f"{resume_text}"
        ),
        expected_output=(
            "The complete rewritten resume in clean ATS-friendly plain text, using "
            "standard CAPS section headers (SUMMARY, EXPERIENCE, SKILLS, EDUCATION, "
            "CERTIFICATIONS). Followed by a change log listing every major edit and "
            "a one-line rationale (e.g., → Summary: fully rewritten — targeted to [role], "
            "injected 3 missing ATS keywords)."
        ),
        agent=rewriter,
        context=[strategy_task, audit_task],
        callback=_shim,
    )

    report_task = Task(
        description=(
            "Compile all outputs from the Enhancement Strategist, Resume Auditor, and "
            "Resume Rewriter into a single complete HTML report following EXACTLY this "
            "format:\n"
            f"{_OUTPUT_FORMAT}\n\n"
            "Rules:\n"
            "1. Include the FULL rewritten resume inside the "
            "<pre class=\"improve-resume-text\"> block\n"
            "2. Change Log must itemize every major edit with a one-line rationale\n"
            "3. ATS keywords added must be individual "
            "<span class=\"improve-keyword\"> chips\n"
            "4. Remaining Gaps must list only gaps that cannot be solved by resume framing\n"
            "5. Upskilling priorities must be ranked and include suggested resources\n"
            "6. If the original fit score was below 60%, uncomment and include the "
            "improve-low-score-warning div\n"
            "7. Score classes for improve-score-value: "
            "excellent (≥85), good (70–84), marginal (60–69), poor (<60)\n"
            "8. Output ONLY the raw HTML block starting with "
            "<div class=\"improve-report\"> — no markdown fences, no introductory "
            "prose, no trailing comments\n"
            "9. Never fabricate skills, experience, degrees, or metrics not in the "
            "original resume"
        ),
        expected_output=(
            "A valid HTML block starting with <div class=\"improve-report\"> containing "
            "all sections: Header (with original and projected scores), Resume Audit "
            "Summary, Enhanced Resume, Change Log, ATS Improvement Summary, Remaining "
            "Gaps, and Upskilling Priorities. No markdown, no extra text outside the "
            "HTML block."
        ),
        agent=reporter,
        context=[strategy_task, audit_task],
        callback=_shim,
    )

    crew = Crew(
        agents=[strategist, auditor, rewriter, reporter],
        tasks=[strategy_task, audit_task, rewrite_task, report_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return _strip_fences(str(result))


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap around HTML output."""
    text = text.strip()
    stripped = re.sub(r'^```[a-zA-Z]*\s*', '', text)
    stripped = re.sub(r'\s*```$', '', stripped)
    return stripped.strip()
