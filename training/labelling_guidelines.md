# Labelling Guidelines — Medical Text Security Classification

**Version 1 — 2026-05-12**

These guidelines define how to assign one of four security tiers (Highly Confidential / Confidential / Restricted / Public) to a piece of medical text. They are the labelling rubric used to construct the training dataset for the Data Pillar classifier, AND the prompt used by Claude to apply labels at scale.

The goal is **consistency, not legal correctness in absolute terms**. Two annotators (human or model) applying these guidelines should produce nearly identical labels on the same text.

---

## Decision rule (read top to bottom — first match wins)

Assign the **highest** tier whose triggers fire. Read the triggers in order: HC → C → R → P. Stop at the first match.

### 1. Highly Confidential (HC) — extra-sensitive medical content

Trigger if the text discusses any of the following, **at the level of a specific patient's clinical content** (not general education):

- **Mental health & psychiatric conditions** — depression, anxiety, bipolar, schizophrenia, PTSD, suicidal ideation, eating disorders.
- **Substance use & addiction** — alcoholism, drug abuse, opioid dependency, smoking cessation in addiction context.
- **HIV, AIDS, STDs** — sexually transmitted infections regardless of stage.
- **Reproductive / sexual health (sensitive subset only)** — abortions, miscarriages, fertility treatment (IVF, infertility workup), sexual dysfunction, gender dysphoria / gender-affirming care, contraception counselling, sexually-transmitted infections, sexual assault. **NOT** routine gynecological surgery (TOT, hysterectomy for fibroids, gynecologic cancer surgery — the cancer trigger handles that), routine obstetric care, or pelvic-area trauma. Those are C.
- **Genetic / hereditary disease** — genetic testing results, hereditary cancer mutations (BRCA, Lynch), Huntington's, cystic fibrosis, sickle cell, rare genetic syndromes.
- **Oncology** — any cancer diagnosis (confirmed malignancy), biopsy results indicating malignancy, chemotherapy, cancer surveillance, **active cancer workup** (imaging + biopsy ordered for suspected cancer). NOT every benign mass excision — if the workup was driven by benign-imaging findings and the biopsy confirms benign, label C. The trigger is "was this patient genuinely investigated for cancer?" not "did the doctor cut something out?"
- **Severe / terminal / palliative conditions** — end-stage disease, hospice care, palliative-only treatment, expected mortality.
- **Pediatric serious illness** — pediatric oncology (confirmed or active workup), pediatric genetic/hereditary disorders, congenital disabilities with significant functional impact. Routine pediatric care (dental, common infections, minor injuries, benign masses) → C.
- **Severe disability** — full physical or cognitive impairment limiting independent living.
- **Domestic violence, abuse, assault** — any reference to abuse as part of the medical presentation.
- **Rare-disease-level identifiability** — clinical detail so specific it could re-identify the patient (rare syndromes with small populations).

If multiple categories overlap (e.g. a patient with both depression and cancer), it's still HC.

### 2. Confidential (C) — standard patient-care content

Trigger if HC didn't fire AND the text is **specific medical content tied to a patient context**:

- **Common chronic conditions** — hypertension, diabetes (type 2 standard cases), asthma, hyperlipidemia, GERD.
- **Acute care / trauma** — fractures, surgeries (routine), emergency presentations not falling into HC.
- **Cardiology** — myocardial infarction, arrhythmia, coronary artery disease, valve disease.
- **Orthopedics, GI, urology, ENT, dermatology** — standard conditions and procedures.
- **Lab values & imaging** — blood tests, CT, MRI, X-rays describing a patient's results.
- **Routine medications** — prescriptions, dosages, treatment plans for non-HC conditions.
- **Outpatient visits, hospital admissions, surgical reports** — standard care narratives.
- **Billing & administrative records** — patient-identifiable admin info, scheduling, insurance details.

Default for clinical narratives that don't trigger HC.

### 3. Restricted (R) — research, aggregate, methodological

Trigger if HC and C didn't fire AND the text is about **research methods, cohort statistics, or population-level findings**, not a single patient's care:

- Clinical trial protocols, methods, intermediate results.
- Aggregated cohort statistics (e.g. "n=243 patients, mean BP 134/86").
- Comparative effectiveness studies (drug A vs drug B across populations).
- Genomic studies at population scale.
- Public health surveillance results.
- Meta-analyses, systematic reviews of clinical questions.

R is about *who/what the data describes*, not about whether the diagnosis is sensitive. Aggregate cancer trial results are R, not HC.

### 4. Public (P) — non-sensitive, general

Trigger if none of the above fire AND the text is:

- **General health education** — "drink water, eat vegetables, exercise" advice without patient specifics.
- **Public health bulletins** — flu season notices, vaccination drives, general advisories.
- **Administrative metadata** — generic billing categories, scheduling system info, *without* patient-identifiable content.
- **Generic medical knowledge** — textbook-style explanations of how a body system works, with no patient tied to it.
- **Truly de-identified, low-detail summaries** — "annual wellness check was normal."

If a piece of text discusses any specific patient with any specific condition, it is **not** Public — drop to C at minimum.

---

## Edge cases

**Mixed content (HC + C in same document):** label the document at the *highest* tier whose triggers fire. A discharge summary that includes a depression history alongside a routine hypertension follow-up → **HC** (depression triggers it).

**Suspected diagnosis that was ruled out:** if the text describes a workup for an HC-tier condition (e.g. cancer suspected, ruled out), label **HC** — the discussion itself is sensitive and the patient was investigated for it.

**De-identified case reports (PMC-style):** the absence of patient name doesn't change the tier. Tier is about content sensitivity, not identifiability. A published PMC case report describing a patient's bipolar disorder is still HC content.

**Pediatric content:** lower threshold for HC. A pediatric case with genetic workup, developmental delay, or oncology → HC.

**Ambiguous between HC and C:** when in doubt about whether a condition is sensitive (e.g. obesity, infertility-adjacent), go **HC** if the text discusses it in detail; **C** if it's a brief mention.

**Multiple plausible tiers:** prefer the higher tier (HC > C > R > P). The access policy generator is designed to handle over-classification gracefully; under-classification is a security failure.

---

## Output format expected from labeller

For each input text, output:

```
{
  "label": "HC" | "C" | "R" | "P",
  "primary_trigger": "<short phrase, e.g. 'pediatric oncology' or 'routine cardiology'>",
  "confidence": "high" | "medium" | "low"
}
```

`confidence` flags low-quality decisions for human review. `primary_trigger` provides the single most important reason for the chosen tier — used in failure analysis.

---

## Anti-patterns to avoid

- ✗ Labelling based on document length, format, or stylistic features.
- ✗ Treating "the patient" as a name (it's not identifying).
- ✗ Calling a generic textbook explanation HC just because it mentions cancer.
- ✗ Calling a detailed cancer case C because "everyone gets cancer eventually."
- ✗ Distinguishing HC and C based on which hospital department wrote the note.

---

## Standards anchors (for paper §2)

- **HIPAA Privacy Rule** (45 CFR 164.514): defines PHI; our HC tier maps to "data with PHI identifiers + sensitive condition".
- **GDPR Article 9**: special-category data includes "data concerning health, sex life, sexual orientation, genetic data, biometric data." Our HC tier targets the high-risk subset within this.
- **NIST 800-171 / SP 800-66r2**: HIPAA implementation guidance. Cited for completeness.
- **ISO 27001 information classification (Annex A)**: 4-tier classification is standard in this framework.

---

## Versioning

- v1 (2026-05-12): initial.
- Future revisions will be additive; labels already assigned should remain valid unless explicitly migrated.
