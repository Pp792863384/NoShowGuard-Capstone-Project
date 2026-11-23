# CareLoop Clinics — No-Show Guard

**Public GitHub Repository:** https://github.com/Pp792863384/NoShowGuard-Capstone-Project

## 4-Line Summary
1. **Part 1 (Python & Data Foundations):** Cleaned 1,020 raw booking records down to 990 valid appointments (removing 20 exact duplicates, 6 unreadable dates, and 4 blank statuses), capped 26 distance and 3 fee outliers, enriched all 8 clinics with Open-Meteo temperatures, and established an overall no-show rate of 22.0% (218 of 990).
2. **Part 2 (SQL Verification):** Reproduced the 990 appointments and 218 no-shows (22.02% overall) in MySQL, showing that no-shows escalate sharply with booking lead time (16.2% for 1–7 days vs 30.0% for 15–30 days and 60.5% for 31+ days) and peak among high-volume clinics at Hadapsar Care Centre (27.3%, 143 bookings) and Malviya Nagar Clinic (26.8%, 142 bookings), while Gomti Nagar Care Point has the highest overall rate at 33.3% (90 bookings).
3. **Part 3 (Machine Learning):** Selected a scaled Logistic Regression pipeline strictly via 5-fold cross-validation (CV ROC-AUC 0.774; single test-set ROC-AUC 0.79) at a 0.35 risk threshold, flagging 49 of 198 test appointments (`test_risk_scores.csv`) to achieve 0.57 recall and 0.57 precision (up from 0.347 recall at the default 0.50 cutoff).
4. **Part 4 (GenAI Follow-up Review):** Analysed the first 10 call-centre notes using Gemini Flash with CRAFT prompting and Pydantic validation (9 valid replies, 1 blank note skipped on N004, 14 API calls total across Tasks 4.1–4.3), confirming that 5 of the 10 notes had `reminder_sent = No` and both `forgot_or_no_reminder` cases (2 of 2: N001 and N007) received no reminder.

## Business Recommendation
CareLoop Clinics should first enforce evening-before automated reminders and deploy targeted front-desk confirmation calls for high-risk bookings—specifically patients booking 15 or more days ahead, patients with prior no-shows, and appointments lacking a reminder—starting at Hadapsar Care Centre (27.3% no-show rate), Malviya Nagar Clinic (26.8%), and Gomti Nagar Care Point (33.3% across 90 bookings). Across the 990 cleaned appointments, the 22.0% baseline no-show rate climbs to 30.0% for 15–30 day lead times and 60.5% for 31+ day bookings, and in the fitted Logistic Regression model (holding other features constant), appointments with a reminder are associated with roughly 69% lower odds of a no-show (odds ratio 0.3094; `lead_days` coefficient +0.6145 and `prior_no_shows` +0.6496). Call-centre notes reinforce why patients miss visits: alongside distance/transport (2 notes) and work/schedule conflicts (2 notes), 2 of 2 patients classified as `forgot_or_no_reminder` had `reminder_sent = No`. Running the Logistic Regression risk model (`test_risk_scores.csv`) each evening at the 0.35 probability threshold lets the front desk focus outreach on the riskiest ~25% of next-day bookings (49 of 198 test appointments), catching 57% of likely no-shows without overwhelming staff.

## Model Choice & Threshold
- **Chosen Model:** `Pipeline([StandardScaler, LogisticRegression(max_iter=1000)])`.
- **Why (Training Data Only):** Models were compared using 5-fold cross-validated ROC-AUC on the 792 training rows without touching the test set. Logistic Regression scored highest (CV ROC-AUC 0.774), outperforming the tuned Random Forest (`max_depth=3`, `n_estimators=200`, CV ROC-AUC 0.767), the default Random Forest (0.747), and the Depth-3 Decision Tree (0.678). Evaluated once on the 198-row test set, Logistic Regression achieved a ROC-AUC of 0.79 (0.789 unrounded).
- **Threshold Choice (0.35):** Selected from out-of-fold training probabilities because lowering the cutoff from 0.50 (OOF Precision 0.58, Recall 0.25, F1 0.35, 73 flagged) to 0.35 substantially improves F1 and nearly doubles recall (OOF Precision 0.51, Recall 0.47, F1 0.49, 154 flagged). On the single final test evaluation at 0.35, the model flagged 49 bookings with 0.57 precision and 0.57 recall. Recall is prioritised because a missed no-show leaves a doctor slot empty, whereas an extra reminder call costs only a minute of front-desk time (in production, clinics would choose this cutoff based on evening call capacity against the cost of an empty slot).

## Limitations
1. **Small note sample:** The 10 analysed call-centre notes serve only as an illustration of the GenAI pipeline and cannot represent the true clinic-wide distribution of missed-appointment reasons.
2. **Pre-split cleaning and random split leakage:** Medians, quartiles, and outlier fences in Part 1 were computed across all 990 rows before the train/test split; in addition, a random split allows the same patient to appear in both train and test, whereas a production model must be trained on past dates and evaluated on future bookings.
3. **Observational data and operational threshold:** Because booking records are observational rather than from a randomised trial, the lower no-show odds associated with reminders do not prove causation; further, a clinic's actual production threshold must be calibrated to front-desk evening call capacity against the cost of an empty slot.

## Repository Contents
- `01_ingest_clean.ipynb` — Part 1 ingestion, validation, cleaning, NumPy summary, Open-Meteo enrichment, and SQL export
- `rejected_rows.json`, `appointments_clean.csv`, `clinics_enriched.csv`, `load_data.sql` — Part 1 generated hand-off files
- `02_queries.sql` — Part 2 MySQL DDL, verification checks (including duplicate-key error comment), and analytical queries
- `03_model.ipynb`, `test_risk_scores.csv` — Part 3 modelling, cross-validation, threshold selection, and final test risk scores
- `04_genai.ipynb`, `notes_analysed.csv` — Part 4 Gemini Flash structured extraction, Pydantic validation, prompt comparison, and cross-tabulation
- `data/` — Original unedited source files (`appointments.csv`, `clinics.csv`, `followup_notes.csv`)
