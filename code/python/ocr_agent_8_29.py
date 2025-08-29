Concept Note: OCR Agent Modernization
Background
The current OCR process for document ingestion and extraction involves multiple manual handoffs
across environments (P8, Instabase UAT, Network Drive, Dev, OCR). This approach is fragile,
repetitive, and not scalable for large document volumes (e.g., 250+ docs). Turnaround per
document type currently stretches to 2–3 months, creating significant delays for downstream
annotation, extraction, and business validation.
1. Current OCR Workflow
• Business intermediary extracts ~500 sample documents and places them in Instabase UAT.
• Files are manually downloaded from Instabase UAT → Network Drive.
• Dev team manually pushes files into Instabase Dev for OCR.
• OCR runs manually, producing CSV outputs per doc (word-level, lacking bounding
box/confidence).
• Dev team manually consolidates CSVs into a single dataset.
• Consolidated results are moved back to the Network Drive for downstream steps.
Pain Points: 4+ manual hops, repetitive file handling, CSV-only outputs, no automation,
non-scalable, and turnaround time of 2–3 months.
2. Proposed OCR Agent Approach
The OCR Agent automates the end-to-end flow: Ingest → OCR → Consolidate → Route →
Dashboard. It replaces manual steps with a fully automated, logged, and auditable pipeline.
Automated Movement: Detects new docs in Instabase UAT automatically, copies them to Dev
input and Network Drive. Implemented via Python transfer service with checksums, scheduled via
cron/systemd, Task Scheduler, or Airflow/Prefect.
OCR Trigger: No more manual runs. Two modes: API (trigger via Instabase APIs/SDKs) or
Watched-folder (Instabase auto-runs OCR when files arrive). Output = structured JSON (doc_id,
page_id, word, bbox, confidence).
Automated Consolidation: OCR JSON outputs parsed and consolidated automatically into a
standard schema (JSON/Parquet). Summary stats logged: docs processed, pages, words, average
confidence.
Routing: Artifacts automatically routed to Network Drive/object store and made available to
Annotation Agent.
Dashboard & Notifications: Processing metrics logged and exposed to dashboards. Run
summaries sent to Slack/Teams/email.
3. Roles & Responsibilities
• Infra/DevOps: Deploy and schedule the agent (cron/systemd/Airflow/Task Scheduler).
• Instabase Admins: Configure OCR for auto-run and enable JSON output.
• Gen AI Team: Build Python automation for transfer, trigger, consolidation, routing, and
notifications.
• Business Team: Continue supplying documents into Instabase UAT (no change).
4. Benefits of OCR Agent
■ Eliminates 4 manual hops across environments.
■ Produces structured JSON/Parquet outputs ready for downstream ML.
■ Scales seamlessly beyond 250 docs.
■ Reduces turnaround from months to days/weeks.
■ Provides audit logs, notifications, and dashboards for visibility.
■ Establishes foundation for daily DAG-based automation.
Conclusion
The OCR Agent transforms a fragile, manual process into a robust, automated pipeline, significantly
improving scalability, accuracy, and speed for data extraction workflows.

Current (Manual)
Business → Instabase UAT → [Manual] → Network Drive → [Manual] → Dev → [Manual] → OCR → CSVs → [Manual] → Consolidate → [Manual] → Network Drive


Proposed (Automated)
Business → Instabase UAT → [Auto Transfer Service] → Dev/Input → [Auto OCR Trigger] → JSON → [Auto Consolidation] → Network Drive → [Auto Routing] → Dashboard
