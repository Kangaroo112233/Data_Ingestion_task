Concept Note: Feedback Agent Modernization
Background

In the current workflow, once documents are processed through OCR and LLM extraction, UAT reviewers manually identify errors and communicate fixes back to the development team. This cycle is slow, inconsistent, and heavily dependent on manual analysis of error reports. As a result, prompt templates and post-processing rules evolve only after significant delays, and systematic issues (e.g., formatting mismatches, recurring false positives) persist across releases. The lack of automation in feedback handling limits the pace of improvement and slows down deployment from Dev → UAT → Prod.

1. Current Feedback Workflow

UAT reviewers validate extracted outputs manually against golden truth.

Errors are logged in spreadsheets or shared informally with Dev.

Dev team manually inspects differences, updates prompts/rules, and redeploys.

No central repository of error patterns → recurring issues re-discovered multiple times.

No automated metrics → business cannot track accuracy improvement release to release.

Pain Points:

Heavy reliance on manual inspection of diffs.

Slow turnaround (weeks) for incorporating fixes into prompts or rules.

No systematic error clustering or prioritization.

Feedback often lost or inconsistently applied.

Accuracy gains plateau due to lack of structured learning loop.

2. Proposed Feedback Agent Approach

The Feedback Agent automates the learning loop: Capture → Compare → Cluster → Suggest → Approve → Apply.
It transforms scattered UAT feedback into structured, actionable improvements for prompts and post-processing rules.

Key Components:

Automated Capture

Collect UAT-validated ground truth alongside model outputs in a standardized schema.

Store in feedback repository (DB/Parquet) with doc_id, field, predicted_value, gt_value, confidence, flags.

Automated Comparison & Clustering

Diff engine compares prediction vs. golden truth at field level.

Identifies mismatches and clusters by error type (regex fail, only-digits, comma-drop, mis-label, etc.).

Generates summary metrics: accuracy per field, edit-rate, false-positive/false-negative counts.

Rule/Prompt Suggestion Engine

For simple errors: auto-generate candidate regex/normalization rules (e.g., “strip commas in currency”, “convert parentheses to negatives”).

For complex errors: flag to prompt engineer/Copilot Studio with contextual examples and propose prompt modifications.

Use Copilot Studio or an internal LLM to draft YAML rule packs/scripts for review.

Approval & Application

Candidate changes presented in dashboard with before/after diffs.

Dev/QA approves → Feedback Agent updates rules_config.yaml or prompt template in Dev repo.

CI/CD pipeline promotes changes through Dev → UAT → Prod after validation.

Dashboard & Notifications

Feedback metrics published to dashboards: accuracy trendlines, top recurring errors, fields needing prompt re-design.

Notifications sent to Slack/Teams/email summarizing weekly feedback outcomes and proposed fixes.

3. Roles & Responsibilities

Business/UAT: Continue reviewing outputs; provide ground truth corrections.

Feedback Agent (automation): Capture corrections, generate diffs, propose fixes, maintain metrics.

Gen AI Team: Review and approve automated suggestions; integrate complex prompt updates.

Infra/DevOps: Support CI/CD pipeline that promotes validated feedback changes into production.

4. Benefits of Feedback Agent

✅ Converts manual, ad-hoc feedback into a structured, automated loop.
✅ Accelerates turnaround for incorporating fixes (days → hours).
✅ Builds a historical knowledge base of error patterns and applied solutions.
✅ Improves accuracy continuously through iterative rule/prompt refinement.
✅ Provides visibility to business on accuracy improvements with quantitative metrics.
✅ Lays foundation for semi-autonomous self-healing pipelines in future phases.

📌 Conclusion:
The Feedback Agent transforms feedback from a manual bottleneck into an automated engine of continuous improvement. By systematizing the capture, analysis, and application of UAT feedback, it ensures that each round of review makes the extraction pipeline smarter, faster, and more reliable for production use.



Current (Manual)

Business/UAT → [Manual Review] → Excel/Spreadsheets → [Manual] → Dev Team → [Manual Diff Analysis] → Prompt/Rule Update → [Manual] → Redeploy to UAT → [Manual] → Track Accuracy

Proposed (Automated)

Business/UAT → [Auto Capture Service] → Feedback Repository (Predictions + GT) → [Auto Diff Engine] → Error Clusters + Metrics → [Auto Suggestion Engine] → Candidate Rules/Prompt Updates → [Auto Validation/Shadow Run] → Approval Dashboard → [Auto Apply via CI/CD] → Dev/UAT → [Auto Metrics Dashboard]
