"""
GynecologistGemma (ObGynGemma) — Gradio chat interface.

Talks directly to the vLLM endpoint (no backend API required to run standalone).
Serves mock clinical answers when vLLM is not yet available (training in progress).

Launch:
  python frontend/gradio/obgyn/launch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

GRADIO_PORT    = int(os.getenv("OBGYN_GRADIO_PORT", "7921"))
VLLM_BASE_URL  = os.getenv("OBGYN_VLLM_BASE_URL", "http://localhost:8105/v1")

SYSTEM_PROMPT = """You are GynecologistGemma, an AI assistant fine-tuned on 50,000 obstetrics
and gynaecology medical cases covering antenatal care, labour and delivery, postpartum management,
gynaecological conditions (PCOS, endometriosis, fibroids, ovarian pathology), gynaecological
oncology, reproductive medicine, and menopause.

Provide detailed, clinically structured, and evidence-based answers grounded in standard OB/GYN
practice (ACOG guidelines, NICE guidelines, RCOG Green-top guidelines, WHO recommendations).
When discussing medications in pregnancy, include safety classification and dosing.
When discussing surgical interventions, outline indications, prerequisites, and complications.

Always remind users to consult a licensed obstetrician or gynaecologist for actual clinical decisions."""

# ── Example questions — deep/specific, drawn from training data topics ──────

EXAMPLE_QUESTIONS = [
    # Obstetric emergencies
    [
        "A 28-year-old primigravida at 34+2 weeks presents with severe headache, visual "
        "disturbances (scotomata), and RUQ pain. BP is 168/112 mmHg on two readings 4 hours apart. "
        "Urine dipstick shows 3+ proteinuria. Platelet count 88,000/µL, ALT 142 U/L. "
        "Diagnose this condition, distinguish it from HELLP syndrome, and provide a full "
        "management protocol including antihypertensives, magnesium sulfate dosing, and delivery timing."
    ],
    [
        "A G3P2 delivers a 3.8 kg baby vaginally. Immediately after placental delivery, "
        "the midwife notes blood loss of 900 mL with a boggy uterus. The bleeding continues "
        "despite fundal massage. Walk through the HAEMOSTASIS protocol for postpartum hemorrhage: "
        "stepwise pharmacological management (oxytocin, ergometrine, carboprost, misoprostol, "
        "tranexamic acid), surgical options (B-Lynch suture, balloon tamponade, UAE, hysterectomy), "
        "and massive transfusion protocol triggers."
    ],
    [
        "A 24-year-old presents to A&E with 6 weeks amenorrhoea, a positive urine pregnancy test, "
        "beta-hCG 2,400 IU/L, and no intrauterine pregnancy on transvaginal ultrasound. She is "
        "haemodynamically stable with mild right-sided pelvic tenderness. Explain the discriminatory "
        "zone concept, classify this as non-diagnostic vs likely ectopic vs definite ectopic, and "
        "provide the full management algorithm: expectant vs methotrexate (indications, dosing, "
        "follow-up) vs surgical laparoscopy (salpingotomy vs salpingectomy)."
    ],
    # Fetal monitoring / labour
    [
        "A primigravida at 38+4 weeks is in active labour at 7 cm cervical dilation. The "
        "continuous CTG shows: baseline FHR 155 bpm, baseline variability 3 bpm, no accelerations "
        "for 50 minutes, and repetitive late decelerations with each contraction. No uterine "
        "hyperstimulation. Using the NICE/FIGO CTG classification: classify each feature, "
        "assign an overall category, and provide a step-by-step management algorithm including "
        "intrauterine resuscitation manoeuvres and threshold for operative delivery."
    ],
    [
        "A 32-year-old G1 at 28+0 weeks presents with 4 regular contractions per hour for "
        "2 hours. Speculum exam shows no ruptured membranes; cervix is 2.5 cm dilated. "
        "fFN swab is positive. Outline: (1) antenatal corticosteroid regimen (betamethasone vs "
        "dexamethasone — dose, timing, why), (2) tocolytic options (nifedipine vs atosiban — "
        "mechanism, duration, evidence), (3) magnesium sulfate for fetal neuroprotection "
        "(MAGPIE-like protocol — dose, duration, monitoring for toxicity), and (4) GBS "
        "prophylaxis considerations."
    ],
    # Antepartum conditions
    [
        "Contrast placenta praevia and placental abruption in a structured comparison: pathophysiology, "
        "clinical presentation (type of bleeding, pain, uterine tone, fetal status), ultrasound "
        "findings, grading/classification systems, and management algorithm for each — "
        "including the role of the 'double set-up examination', conservative vs active management, "
        "and indications for emergency caesarean section."
    ],
    [
        "A 30-year-old G2P1 at 26 weeks' gestation has a 75 g OGTT result: fasting 5.4 mmol/L, "
        "1-hour 11.1 mmol/L, 2-hour 9.2 mmol/L. Diagnose using WHO 2013 / IADPSG criteria. "
        "Describe: (1) initial dietary and exercise management, (2) blood glucose monitoring "
        "targets, (3) insulin initiation criteria and regimen (NPH vs analogue), (4) fetal "
        "surveillance schedule, (5) intrapartum management (glucose-insulin infusion), and "
        "(6) postpartum follow-up (HbA1c, conversion risk to T2DM)."
    ],
    # Gynaecological conditions
    [
        "A 22-year-old woman presents with a 2-year history of irregular periods (cycle 35–75 days), "
        "hirsutism (Ferriman-Gallwey score 14), acne, and BMI 31.5 kg/m². Pelvic ultrasound shows "
        "bilateral ovaries with 14 follicles per ovary (each <10 mm). Apply the Rotterdam 2003 criteria "
        "to confirm PCOS. What is the full metabolic and hormonal workup? Compare treatment strategies "
        "for: (a) metabolic management (lifestyle, metformin, inositol), (b) menstrual regulation "
        "(OCP vs progestin), (c) hyperandrogenism (spironolactone, flutamide), and (d) ovulation "
        "induction for fertility (clomiphene, letrozole, gonadotrophins, IVF)."
    ],
    [
        "A 32-year-old nulliparous woman presents with a 5-year history of progressive "
        "dysmenorrhea, deep dyspareunia, dyschezia, and 18 months of subfertility. CA-125 is "
        "elevated at 68 U/mL. Laparoscopy reveals peritoneal implants, an ovarian endometrioma "
        "(4 cm), and partial obliteration of the pouch of Douglas. Apply the revised ASRM "
        "staging. Detail: (1) medical management (GnRH agonist vs dienogest vs combined OCP — "
        "mechanism, add-back therapy), (2) surgical approach to the endometrioma (drainage vs "
        "cystectomy — recurrence rates, impact on ovarian reserve, AMH implications), and "
        "(3) fertility treatment pathway (IUI vs IVF)."
    ],
    [
        "Explain the current cervical cancer screening programme: (1) when to start, "
        "frequency, and when to stop; (2) HPV primary screening vs co-testing vs cytology alone — "
        "sensitivity/specificity comparison; (3) how to interpret results: HPV-positive/cytology "
        "negative, ASC-US, LSIL, HSIL, ASC-H; (4) colposcopy — what the colposcopist looks "
        "for (acetowhite epithelium, punctation, mosaic, atypical vessels), biopsy indications, "
        "and CIN classification; (5) LLETZ/LEEP procedure — indications, technique, and "
        "obstetric consequences (cervical insufficiency risk)."
    ],
    [
        "A 26-year-old woman presents to A&E with sudden onset severe right iliac fossa pain "
        "for 3 hours, nausea, and vomiting. Pregnancy test is negative. Pelvic ultrasound "
        "shows a 6.5 cm right ovarian cyst with absent Doppler flow within the ovary. "
        "Confirm the diagnosis of ovarian torsion. What is the differential (appendicitis, "
        "ruptured cyst, PID, ectopic)? Describe the surgical management — laparoscopy vs "
        "laparotomy, detorsion alone vs oophoropexy vs salpingo-oophorectomy, and how the "
        "decision changes based on the appearance of the ovary at laparoscopy."
    ],
    # Reproductive endocrinology / menopause
    [
        "A 51-year-old woman presents with 18 months of irregular periods, hot flushes 8–10 "
        "times per day (disrupting sleep), vaginal dryness, dyspareunia, and mood changes. "
        "FSH 65 IU/L, LH 42 IU/L, oestradiol <100 pmol/L. Confirm the diagnosis of menopause. "
        "Compare HRT formulations: (a) systemic combined vs oestrogen-only (who gets what), "
        "(b) oral vs transdermal route (VTE risk, why transdermal is preferred), "
        "(c) progestogen choice (MPA vs micronised progesterone — breast cancer risk data), "
        "(d) genitourinary syndrome management (vaginal oestrogen), and "
        "(e) absolute contraindications to HRT."
    ],
    [
        "A 34-year-old woman presents with her 36-year-old partner. They have been trying "
        "to conceive for 18 months without success. She has regular 28-day cycles; he has "
        "no previous children. Outline the complete evidence-based investigation pathway: "
        "(1) baseline female investigations (Day 3 FSH/LH/oestradiol, AMH, cycle Day 21 "
        "progesterone, pelvic US, hysterosalpingography — sensitivity for tubal pathology), "
        "(2) male factor evaluation (semen analysis — WHO 2021 reference values: morphology, "
        "motility, count; when to refer to andrology), (3) diagnosis and treatment of each "
        "cause identified (anovulation: clomiphene vs letrozole; tubal: laparoscopy vs IVF; "
        "male factor: IUI vs ICSI), and (4) when to move to IVF."
    ],
    # Labour and delivery
    [
        "A 29-year-old primigravida at 39+2 weeks has been in the second stage of labour for "
        "2.5 hours with no progress in the last 60 minutes. The fetal head is at +2 station, "
        "well-flexed, OA position. FHR is reassuring. You are considering operative vaginal "
        "delivery. Compare vacuum (ventouse) and forceps: indications where each is preferred, "
        "absolute prerequisites (station, position, engagement, anaesthesia, bladder), "
        "application technique, traction rules (Pajot's manoeuvre for forceps), maternal "
        "complications (3rd/4th degree tears, maternal trauma), neonatal complications "
        "(cephalhaematoma vs subgaleal haemorrhage — the dangerous one), and when to "
        "abandon and proceed to emergency caesarean."
    ],
    # Maternal sepsis
    [
        "A 32-year-old woman at 20 weeks' gestation presents with a 24-hour history of "
        "dysuria, frequency, and rigors. On examination: temperature 39.4°C, HR 134 bpm, "
        "BP 88/54 mmHg, RR 24/min, SpO₂ 95% on air. Urine dipstick: nitrites+, leucocytes+++. "
        "Apply the Modified Obstetric Early Warning Score (MEOWS). Diagnose septic shock in "
        "pregnancy. Describe the obstetric Sepsis-6 bundle (specific to pregnancy). "
        "Which organisms are most common in pregnancy-related sepsis? What are the antibiotic "
        "choices safe in pregnancy? How does sepsis physiology differ in pregnancy "
        "(increased cardiac output, immune tolerance, altered pharmacokinetics)? "
        "When does maternal sepsis mandate urgent delivery regardless of gestation?"
    ],
]

# ── Mock answers (served when vLLM is not yet up) ────────────────────────────

MOCK_ANSWERS: list[tuple[list[str], str]] = [

    # 1 — Preeclampsia / HELLP
    (
        ["preeclampsia", "hellp", "28-year-old", "headache", "bp 168", "magnesium", "antihypertensive"],
        """## Preeclampsia with Severe Features — Management Protocol

### Diagnosis
This is **preeclampsia with severe features** (ACOG 2019):
- BP ≥160/110 mmHg on two readings ≥4 hours apart ✓
- Proteinuria ≥300 mg/24h (3+ dipstick as surrogate) ✓
- Thrombocytopenia (platelets <100,000/µL) ✓
- Hepatic involvement (ALT >40 U/L, RUQ pain) ✓
- Severe headache / visual disturbances ✓

**Distinguish from HELLP syndrome:**

| Feature | Preeclampsia (severe) | HELLP Syndrome |
|---------|----------------------|----------------|
| Definition | BP ≥160/110 + severe symptoms | **H**aemolysis + **E**levated **L**iver enzymes + **L**ow **P**latelets |
| Haemolysis | Not required | Required (LDH >600, schistocytes) |
| Overlap | 70–80% of HELLP have preeclampsia | Subset of severe preeclampsia |
| Platelets | Any range | <100,000/µL (Class I: <50,000) |
| Prognosis | Severe | HELLP Class I carries highest mortality |

This patient with platelets 88,000 and ALT 142 **meets Tennessee criteria for partial HELLP** (Class II). Check: LDH, total bilirubin, peripheral smear for schistocytes.

---

### Antihypertensive Management

**Target: BP <160/110 mmHg** — acute severe hypertension (≥160/110) is a hypertensive emergency requiring treatment within 30–60 minutes.

| Drug | Dose | Route | Notes |
|------|------|-------|-------|
| **Labetalol** (first-line) | 20 mg IV → 40 mg → 80 mg q10min (max 300 mg) | IV | Avoid in asthma; preferred in UK/NICE |
| **Hydralazine** | 5–10 mg IV q20min (max 30 mg) | IV | Reflex tachycardia common |
| **Nifedipine** (oral) | 10–20 mg PO, repeat in 30 min | PO | Use immediate-release (NOT sustained-release in acute setting) |
| **Oral nifedipine** (maintenance) | 30–60 mg SR daily | PO | Maintenance once acute BP controlled |

---

### Magnesium Sulfate (Seizure Prophylaxis)

**Indication:** All preeclampsia with severe features — seizure prophylaxis (reduces eclampsia risk 58%, NNT ~50).

**Pritchard Protocol:**
- Loading dose: **4 g IV** over 20 minutes
- Maintenance: **1–2 g/hour** continuous IV infusion

**Monitoring (magnesium toxicity):**

| Finding | Serum Mg level | Action |
|---------|---------------|--------|
| Therapeutic range | 4–7 mEq/L | Continue infusion |
| Loss of patellar reflexes | ~7 mEq/L | **Reduce infusion rate** |
| Respiratory depression | ~10 mEq/L | **Stop infusion**, give **calcium gluconate 1 g IV** (antidote) |
| Cardiac arrest | >15 mEq/L | Emergency resuscitation + calcium |

**Monitor:** Reflexes hourly, urine output >25 mL/hour, RR >12/min. Reduce dose in renal impairment (↑ Mg level risk).

---

### Delivery Timing

At **34+2 weeks with severe features**: **deliver after maternal stabilisation** (not conservative management):
- Acute BP control: ≤30–60 minutes
- Magnesium loading: start before delivery
- **Corticosteroids** (betamethasone 12 mg IM × 2 doses 24h apart) if time permits — but do not delay delivery >24–48h for steroids alone in severe disease
- Mode: Vaginal delivery preferred if cervix favourable (Bishop score) and no other contraindication; LSCS for obstetric indications

**Do NOT pursue expectant management beyond 34+0 weeks with severe features** — risk of HELLP progression, eclampsia, abruption, and maternal organ failure outweighs prematurity benefit.

---

### Postpartum
- Seizure risk peaks 24–48h postpartum — continue MgSO₄ for 24h post-delivery
- BP may worsen Day 3–5 postpartum (fluid remobilisation)
- Continue antihypertensives until BP stable for 72h without medication
- **Counsel on long-term cardiovascular risk** — preeclampsia confers 4× lifetime risk of hypertension, 2× risk of stroke

> ⚕️ *Preeclampsia with severe features is a multi-organ emergency. Simultaneous BP control, MgSO₄ initiation, and delivery planning must happen in parallel — call senior obstetric and anaesthetic teams immediately.*
""",
    ),

    # 2 — Postpartum hemorrhage
    (
        ["postpartum hemorrhage", "postpartum haemorrhage", "haemostasis", "oxytocin", "carboprost", "b-lynch", "boggy uterus"],
        """## Postpartum Haemorrhage — HAEMOSTASIS Protocol

### Definition and Diagnosis
- **PPH:** Blood loss ≥500 mL after vaginal delivery (or ≥1000 mL after CS)
- **Major PPH:** ≥1000 mL with signs of haemodynamic compromise
- **This patient:** 900 mL loss, boggy uterus → **primary PPH, most likely atonic (Tone = 80% of PPH)**

### The 4 T's of PPH
| Cause | Frequency | Signs |
|-------|-----------|-------|
| **Tone** (uterine atony) | 80% | Boggy, poorly contracted uterus |
| **Trauma** (lacerations) | 10–15% | Uterus firm; bleeding from cervix/vagina |
| **Tissue** (retained placenta) | 5% | Incomplete placenta, membranes |
| **Thrombin** (coagulopathy) | 1–2% | Oozing from IV sites, DIC picture |

---

### HAEMOSTASIS Protocol

**H — Help and IV access**
- Call senior midwife, obstetrician, anaesthetist
- 2 large-bore IVs (16G), bloods: FBC, coagulation, fibrinogen, G&S, X-match 4 units
- Catheterise — monitor urine output

**A — Assess (vital signs, 4 T's) + uterotonics**

| Drug | Dose | Mechanism | Notes |
|------|------|-----------|-------|
| **Oxytocin** 1st line | 10 IU IM or slow IV, then 40 IU infusion | Myometrial contraction | Avoid rapid IV bolus (hypotension, tachycardia) |
| **Ergometrine** | 0.5 mg IM/IV | Sustained uterine contraction | Contraindicated in hypertension, PET |
| **Syntometrine** | 1 amp IM | Oxytocin + ergometrine combined | |
| **Carboprost** (15-methyl PGF2α) | 250 µg IM q15min (max 2 mg / 8 doses) | PG-mediated contraction | Contraindicated in asthma; causes bronchospasm |
| **Misoprostol** | 800–1000 µg PR/SL | PGE1 analogue | Use if above unavailable or failed |
| **Tranexamic acid (TXA)** | **1 g IV** over 10 min, repeat 1 g if needed | Antifibrinolytic | **Give within 3 hours of delivery** (WOMAN trial — reduces death from PPH 31%) |

**E — Examine under anaesthesia** (bimanual compression, explore uterine cavity, inspect cervix/vagina for lacerations)

**M — Massage + Bimanual compression** — external + internal bimanual for atony while uterotonics take effect

**O — OT / Theatre** — if ≥2 uterotonics failed

**S — Surgical haemostasis:**
1. **Intrauterine balloon tamponade** (SOS Bakri balloon, 300–500 mL) — "tamponade test": if bleeding stops, avoid surgery
2. **Compression sutures** — B-Lynch suture (external longitudinal compression), Cho square sutures
3. **Bilateral uterine artery ligation** (O'Leary sutures) — reduces uterine blood flow 90%
4. **Iliac artery ligation** (internal iliac) — technically demanding
5. **Uterine artery embolisation (UAE)** — if stable and IR available; preserves fertility
6. **Peripartum hysterectomy** — definitive; life-saving last resort

**T — Transfusion / MTP**
- Activate **Massive Transfusion Protocol (MTP)** if blood loss >1500 mL or haemodynamic instability
- Target ratio: RBC:FFP:Platelets = **1:1:1** (CRASH-2 principles)
- Target fibrinogen >2 g/L — give cryoprecipitate if fibrinogen <1.5 g/L (high risk DIC with PPH)
- **Calcium** (calcium gluconate 10 mL 10%) with each 4 units RBC — citrate in blood products chelates calcium

**A — Anaesthesia** — regional preferred; GA if haemodynamically unstable

**S — SOS** (call for help early: haematology, vascular, ICU)

**I — ICU admission** for ongoing monitoring if ≥2 L blood loss, massive transfusion, or coagulopathy

**S — Secondary survey** — once stabilised: review all notes, discuss with patient

> ⚕️ *Time is muscle. Activate the PPH protocol at 500 mL — do not wait. TXA must be given within 3 hours. Hysterectomy is a life-saving, not a last-last resort.*
""",
    ),

    # 3 — Ectopic pregnancy
    (
        ["ectopic", "beta-hcg", "2400", "transvaginal", "methotrexate", "salpingectomy", "discriminatory"],
        """## Ectopic Pregnancy — Discriminatory Zone and Management Algorithm

### Discriminatory Zone
The **discriminatory zone** is the beta-hCG threshold above which an IUP should be visible on transvaginal ultrasound (TVUS):
- With a **single TVUS probe**: ~1,500–2,000 IU/L
- With a **high-quality TVUS**: ~3,000–3,500 IU/L (ACOG)

**This patient: hCG 2,400 IU/L, no IUP on TVUS**

At this hCG level (within or just above discriminatory zone), and with no IUP and pelvic tenderness:
- Diagnosis: **Pregnancy of unknown location (PUL) vs ectopic** — cannot be categorically confirmed until either an IUP is seen or an ectopic visualised

**Management after 48h:**
- Repeat hCG: rising >66% in 48h → IUP expected; rising <53% or falling → abnormal IUP or ectopic
- Repeat TVUS in 48–72h once hCG rises above discriminatory zone

---

### Management Algorithm

```
Haemodynamically stable?
         │
     YES ├──► TVUS shows adnexal mass + no IUP + hCG ≥1500?
         │              │
         │           YES ├──► Methotrexate if criteria met
         │              │     Laparoscopy if criteria not met
         │              │
         │           NO  └──► Serial hCG + repeat TVUS in 48h
         │
     NO  └──► EMERGENCY LAPAROSCOPY
              (haemoperitoneum, shock)
```

---

### Expectant Management
**Criteria:** hCG <1000 IU/L, declining, TVUS shows no haemorrhage, no cardiac activity, fully counselled, reliable follow-up.
- **Success rate:** 57–100% depending on initial hCG
- Monitor hCG twice weekly until negative

---

### Methotrexate (MTX) — Medical Management

**ACOG Eligibility Criteria (ALL must be met):**
- Haemodynamically stable
- No significant abdominal pain (not ruptured)
- Ectopic mass ≤3.5 cm (ACOG) / <4 cm (NICE/RCOG)
- No fetal cardiac activity
- hCG ideally <5,000 IU/L (relative; some use <10,000)
- Adequate renal, hepatic, haematologic function
- No breastfeeding, immunodeficiency, blood dyscrasias, active pulmonary disease

**Contraindications:** Breastfeeding, peptic ulcer, hepatic/renal disease, haematologic abnormalities, pulmonary disease, immunodeficiency.

**Single-dose protocol (most common):**
- MTX **50 mg/m²** IM single dose
- Baseline LFTs, FBC, hCG, transvaginal US
- Monitor hCG on Day 4 and Day 7: expect 15% fall Day 4–7
- If <15% fall: give second dose 50 mg/m²

**Counsel on "separation pain"** (Days 3–7) — normal mild cramping from fallopian tube reaction; instruct to attend A&E if severe pain (rupture risk).
**No NSAIDs** (interfere with MTX mechanism). Avoid folic acid supplements. Counsel against pregnancy for **3 months** post-MTX.

---

### Surgical Management

**Laparoscopy is the gold standard for ectopic treatment.**

**Salpingotomy (linear salpingotomy):**
- Incision along the antimesenteric border, ectopic evacuated
- Preserves tube — preferred if **contralateral tube damaged or absent**
- Risk: persistent trophoblast (5–20%) → monitor hCG post-op; may need MTX
- RCT evidence (ESEP study): salpingotomy vs salpingectomy — no significant difference in subsequent intrauterine pregnancy rates

**Salpingectomy (tube removal):**
- **Preferred** if contralateral tube is healthy (NICE/RCOG)
- Eliminates persistent trophoblast risk
- No significant fertility disadvantage when contralateral tube is normal

**Emergency laparotomy:** If haemodynamically unstable, haemoperitoneum, or laparoscopy not available.

---

### Follow-Up
- hCG monitoring to zero after all treatment modalities
- Rhesus D immunoglobulin (250 IU / 500 IU if >20 weeks) if Rh-negative
- Discuss implications for future fertility (IVF if bilateral tubal damage)
- Emotional support — pregnancy loss, fertility anxiety

> ⚕️ *Ectopic pregnancy is the leading cause of first-trimester maternal death. Haemodynamic instability = operating theatre immediately — do not delay for serial hCGs.*
""",
    ),

    # 4 — CTG interpretation
    (
        ["ctg", "cardiotocograph", "late deceleration", "baseline variability", "figo", "nice", "fetal monitoring"],
        """## CTG Interpretation — NICE/FIGO Classification and Management

### Feature-by-Feature Analysis

| CTG Feature | This Trace | Classification |
|-------------|-----------|----------------|
| **Baseline FHR** | 155 bpm (normal: 110–160) | ✅ Normal |
| **Baseline variability** | 3 bpm (normal: 5–25 bpm) | ❌ Reduced (<5 bpm for ≥50 min) → **Non-reassuring** |
| **Accelerations** | None for 50 min (normal: ≥2/20 min) | ❌ Absent → **Non-reassuring** (in active labour) |
| **Decelerations** | Repetitive late decelerations | ❌ **Abnormal** — late decelerations are NEVER reassuring |

---

### NICE 2022 CTG Category System

| Category | Definition | Action |
|----------|-----------|--------|
| **Normal** | All 4 features reassuring | Continue CTG; no immediate action |
| **Suspicious** | 1 non-reassuring OR 1 abnormal feature | Correct reversible causes; consider fetal blood sampling (FBS) |
| **Pathological** | ≥2 non-reassuring OR ≥1 abnormal feature | **Immediate action**: FBS or delivery |

**This CTG:** Reduced variability (non-reassuring) + absent accelerations (non-reassuring) + repetitive late decelerations (ABNORMAL) = **PATHOLOGICAL**

---

### Pathophysiology of Late Decelerations
Late decelerations = FHR nadir >30 seconds after peak of contraction → **uteroplacental insufficiency**
- Mechanism: uterine contraction → ↓ intervillous blood flow → fetal hypoxia → vagal response → bradycardia (late because placenta buffers the hypoxic signal)
- Even shallow late decelerations (<15 bpm) that are repetitive are **abnormal**
- Associated with: IUGR, post-dates, placental abruption, preeclampsia, maternal hypotension

---

### Intrauterine Resuscitation Manoeuvres (immediate — do BEFORE calling theatre)

1. **Maternal repositioning** — left lateral decubitus (relieves aortocaval compression; ↑ placental perfusion)
2. **Stop oxytocin** (if running) — reduce uterine hyperstimulation
3. **IV fluid bolus** — 500 mL Hartmann's if epidural or hypotension present
4. **Supplemental oxygen** — 15 L/min via non-rebreather mask (controversial evidence but low harm in acute setting)
5. **Exclude cord prolapse** — urgent vaginal examination
6. **Tocolysis** (terbutaline 0.25 mg SC) — if uterine hyperstimulation causing decelerations

---

### Decision Algorithm

```
Pathological CTG
       │
Apply intrauterine resuscitation
       │
       ├──► Improved (normal/suspicious) → Continue CTG, reassess
       │
       └──► Persistent pathological
                    │
              Cervix ≥7 cm dilated, experienced operator
                    │
              Perform Fetal Blood Sampling (FBS)
                    │
                pH ≥7.25 → Continue; repeat FBS in 1h
                pH 7.20–7.24 → Repeat FBS in 30 min
                pH <7.20 → IMMEDIATE DELIVERY
                    │
              Cervix not suitable for FBS or FBS unavailable
                    │
              IMMEDIATE OPERATIVE DELIVERY
              (ventouse/forceps if ≥+2 station; CS otherwise)
```

---

### Fetal Blood Sampling (FBS)
- Scalp sample: pH + base excess
- Lactate (alternative to pH; threshold: <4.1 normal, ≥4.8 = immediate delivery)
- Contraindicated in: HIV, hepatitis B/C, herpes, fetal blood disorder (thrombocytopenia, haemophilia)

> ⚕️ *A pathological CTG in active labour requires you to be physically present at the bedside. Document your assessment, management steps, and decision-making time-stamped. CTG interpretation errors are the most common cause of intrapartum litigation.*
""",
    ),

    # 5 — Preterm labour
    (
        ["preterm labor", "preterm labour", "28 weeks", "betamethasone", "nifedipine", "atosiban", "magnesium", "neuroprotection", "ffn"],
        """## Preterm Labour at 28 Weeks — Full Management Protocol

### Diagnosis
Preterm labour (PTL) = regular contractions causing cervical change at 20+0–36+6 weeks.
**This patient:** 28+0 weeks, 4 contractions/hour, cervix 2.5 cm dilated, positive fFN.
- **fFN positive (>200 ng/mL)** + cervix ≥1.5 cm dilated: **positive predictive value 77%** for delivery within 7 days → active intervention warranted

---

### 1. Antenatal Corticosteroids (ACS)

**Purpose:** Fetal lung maturation (surfactant), reduces RDS by 50%, IVH by 46%, NEC by 54%.

| Regimen | Dose | Route | Timing |
|---------|------|-------|--------|
| **Betamethasone** (preferred UK/RCOG) | 12 mg × 2 doses, 24h apart | IM | Maximum benefit at 24–48h after first dose; partial benefit even if <24h elapses |
| **Dexamethasone** (preferred USA/WHO) | 6 mg × 4 doses, 12h apart | IM | Equivalent efficacy |

**Window:** 24+0–33+6 weeks (strong evidence); consider 34+0–36+6 weeks (late preterm steroids — ALPS trial: modest NNT for respiratory morbidity, no harm).

**Repeat course:** Single rescue course if first course >7 days ago and likely delivery within 7 days (do not routinely repeat).

---

### 2. Tocolysis (buys 48h for ACS to work + maternal transfer)

Tocolysis does NOT improve perinatal outcomes beyond facilitating ACS completion and in-utero transfer. Do NOT tocolyse beyond 48h or beyond 34 weeks.

#### Nifedipine (Calcium Channel Blocker) — **First-line UK**

| Parameter | Detail |
|-----------|--------|
| Mechanism | Blocks L-type Ca²⁺ channels in myometrial smooth muscle → ↓ uterine contractility |
| Dose | 20 mg PO loading, then 10–20 mg q6h for up to 48h (modified-release) |
| Advantages | Oral, cheap, most evidence for PTL |
| Maternal SE | Hypotension, headache, flushing, palpitations |
| Fetal | No adverse effects established |
| Contraindications | Pre-existing hypotension, aortic stenosis, cardiac disease, concurrent MgSO₄ (hypotension risk) |

#### Atosiban (Oxytocin Receptor Antagonist) — **First-line some European centres**

| Parameter | Detail |
|-----------|--------|
| Mechanism | Competitive antagonist at myometrial oxytocin receptors → ↓ contractions |
| Dose | 6.75 mg IV bolus → 18 mg/hour × 3h → 6 mg/hour × 45h (max 48h course) |
| Advantages | Fewer maternal side effects than nifedipine or ritodrine; specific mechanism |
| Disadvantages | IV only, expensive; no benefit over nifedipine in RCTs |
| Fetal | Rare fetal/neonatal reactions; not licensed <24 weeks |

**Beta-2 agonists (ritodrine, salbutamol):** Effective but abandoned in many centres due to serious maternal cardiovascular side effects. Used if nifedipine/atosiban contraindicated.

**Indomethacin (COX inhibitor):** Effective; used 24–32 weeks. Risk of premature closure of ductus arteriosus (avoid >32 weeks) and neonatal renal effects.

---

### 3. Magnesium Sulfate for Fetal Neuroprotection

**Indication:** Imminent birth <34 weeks (ideally <32 weeks — highest NNT benefit).
**Effect:** Reduces risk of **cerebral palsy** by 32% (Cochrane, NNT ~50 at <34 weeks).

**Protocol (Crowther/MAGPIE-derived):**
- Loading dose: **4 g IV** over 20–30 minutes
- Maintenance: **1 g/hour** for up to 24 hours (or until delivery)

**Monitoring:** Same as for eclampsia prophylaxis — reflexes, RR, urine output. Antidote: calcium gluconate 1 g IV.

**Note:** MgSO₄ is NOT a tocolytic in this context (used specifically for neuroprotection, separate from tocolysis).

---

### 4. Group B Streptococcus (GBS) Prophylaxis

- If GBS status unknown and delivery imminent at <37 weeks: **IV penicillin G** 3 MU loading then 1.5 MU q4h (or ampicillin 2 g → 1 g q4h)
- Aim for ≥4 hours before delivery for adequate prophylaxis

---

### Additional Actions
- In-utero transfer to tertiary unit with Level 3 NICU (if not already there)
- Neonatology team briefed — parental counselling re: 28-week outcomes (survival ~90%, significant morbidity risk)
- Anaesthetic team aware for possible LSCS
- Serial cervical length if contractions settle (can guide discharge)

> ⚕️ *The window to give corticosteroids is 24–48h. If delivery looks imminent, give the first betamethasone dose immediately — partial benefit is better than none. Never delay ACS for tocolysis assessment.*
""",
    ),

    # 6 — PCOS
    (
        ["pcos", "polycystic ovary", "rotterdam", "irregular periods", "hirsutism", "ferriman", "metformin", "letrozole"],
        """## Polycystic Ovary Syndrome — Rotterdam Criteria, Workup, and Treatment

### Rotterdam 2003 Diagnostic Criteria
Requires **2 of 3** features (after excluding other causes):
1. **Oligo-/anovulation** — cycles >35 days or <8 cycles/year ✓
2. **Clinical or biochemical hyperandrogenism** — FG score ≥8 (this patient: 14), or elevated free testosterone/DHEAS ✓
3. **Polycystic ovary morphology (PCOM)** on ultrasound — ≥12 follicles 2–9 mm per ovary (2003 threshold; **2018 update: ≥20 follicles/ovary on high-frequency TVUS**) ✓

**This patient: All 3 features present → confirmed PCOS.**

Exclusions to rule out before confirming PCOS:
- Thyroid dysfunction (TSH)
- Hyperprolactinaemia (prolactin)
- Non-classical congenital adrenal hyperplasia (17-OHP)
- Cushing's syndrome (24h urinary free cortisol if clinically suspected)

---

### Metabolic and Hormonal Workup

**Hormonal:**
- LH, FSH, LH:FSH ratio (often >2 in PCOS, not diagnostic)
- Total testosterone, SHBG → calculate free androgen index (FAI)
- DHEAS (elevated in 50% of PCOS)
- 17-OHP (rule out NCAH — elevated in NCAH)
- Prolactin, TSH

**Metabolic (MANDATORY — metabolic syndrome in 30–40% of PCOS):**
- Fasting glucose + 75g OGTT (impaired glucose tolerance in 30%, T2DM in 10%)
- Fasting lipid profile
- BP measurement, waist circumference
- **AMH** — elevated in PCOS (~3–4× normal; reflects increased antral follicle count)

---

### Treatment by Goal

#### A. Metabolic Management

| Intervention | Evidence |
|-------------|---------|
| **Lifestyle modification** (5–10% weight loss) | First-line — restores ovulation in 50%, improves insulin resistance, reduces androgens |
| **Metformin** 500–2000 mg/day | Reduces insulin resistance, improves menstrual regularity; modest effect on hirsutism; not inferior to OCP for metabolic outcomes |
| **Inositol** (myo-inositol 4g + D-chiro-inositol) | Emerging evidence; insulin sensitiser; safe; some data for ovulation induction |

#### B. Menstrual Regulation and Contraception

| Drug | Mechanism | Notes |
|------|-----------|-------|
| **Combined OCP** | Suppress LH → ↓ androgen; progestin opposes endometrial proliferation | First-line for cycle regulation + hirsutism; any combined OCP works; drospirenone/cyproterone acetate have added anti-androgen effect |
| **Cyclic progestogen** (norethisterone 5 mg days 16–25) | Induces withdrawal bleeds | If OCP contraindicated; protects endometrium from hyperplasia (unopposed oestrogen risk) |

#### C. Hyperandrogenism (Hirsutism, Acne)

| Drug | Dose | Mechanism | Notes |
|------|------|-----------|-------|
| **Spironolactone** | 100–200 mg/day | Androgen receptor antagonist + weak 5α-reductase inhibitor | Diuretic; teratogenic — use with contraception |
| **Cyproterone acetate** | 2 mg (in OCP Dianette/Diane-35) | Anti-androgen progestogen | Cannot use long-term as sole OCP (VTE risk) |
| **Flutamide** | 250–500 mg/day | Androgen receptor antagonist | Hepatotoxicity risk — monitor LFTs |
| **Eflornithine cream** | Apply BD to face | Inhibits ornithine decarboxylase in hair follicles | Adjunct to systemic therapy |

#### D. Ovulation Induction (Fertility)

**Step 1: Lifestyle — weight loss to BMI <30 before pharmacological OI (improves live birth rates).**

| Agent | Dose | Mechanism | Success rate | Notes |
|-------|------|-----------|-------------|-------|
| **Letrozole** (first-line) | 2.5–7.5 mg/day D2–6 | Aromatase inhibitor → ↑ FSH | 61% live birth (NEJM 2014 RCT) | Higher live birth rate than clomiphene in PCOS; not teratogenic at OI doses |
| **Clomiphene citrate** | 50–150 mg/day D2–6 | Anti-oestrogenic → ↑ FSH | 46% live birth | Max 6 cycles (anti-oestrogenic endometrial effect; theoretical neoplasm risk) |
| **Metformin + clomiphene** | Combined | Sensitises to clomiphene | Modest additional benefit in obese PCOS | |
| **Gonadotrophins** | FSH/LH SC injections | Direct FSH stimulation | High ovulation, ↑OHSS risk | Requires US monitoring; step-up protocol |
| **IVF** | — | — | For gonadotrophin-failure or bilateral tubal disease | OHSS risk — freeze-all strategy preferred in PCOS |

**Ovarian drilling** (laparoscopic diathermy): alternative to gonadotrophins; restores ovulatory cycles in 80%; avoids OHSS; limited to women failing clomiphene without male factor.

---

### Long-term Follow-up
- Annual OGTT (progression to T2DM)
- Cardiovascular risk assessment (BP, lipids, weight)
- Endometrial monitoring if oligomenorrhoea (risk of endometrial hyperplasia/cancer with chronic anovulation + ↑ oestrogen without progestogen opposition)

> ⚕️ *PCOS is a lifelong condition, not just a fertility problem. Metabolic management (weight, insulin resistance) has the highest long-term impact on health outcomes.*
""",
    ),

    # 7 — Endometriosis
    (
        ["endometriosis", "endometrioma", "dysmenorrhea", "dyspareunia", "rasrm", "gnh agonist", "dienogest", "asrm staging"],
        """## Endometriosis — Staging, Medical, and Surgical Management

### revised ASRM Staging (rASRM)
Endometriosis is staged by laparoscopic findings using a point scoring system:

| Stage | Name | Score | Key features |
|-------|------|-------|-------------|
| I | Minimal | 1–5 | Isolated implants, no significant adhesions |
| II | Mild | 6–15 | More implants, superficial ovarian involvement |
| III | Moderate | 16–40 | Endometriomas, peritubal adhesions |
| **IV** | **Severe** | **>40** | Large endometriomas, dense adhesions, obliterated POD |

**This patient:** Peritoneal implants + 4 cm endometrioma + partial obliteration of POD → **Stage III–IV**.

**Important caveat:** rASRM stage correlates poorly with pain severity and fertility outcomes. It is a surgical staging tool, not a pain staging tool.

---

### Medical Management

#### First-Line Pain Management
- NSAIDs (naproxen, mefenamic acid) for dysmenorrhoea — symptomatic only
- **Combined OCP (continuous — skip placebo pills)** — suppresses ovulation and reduces menstruation; first-line for most women not seeking immediate fertility

#### Hormonal Suppression Therapy

| Drug | Mechanism | Evidence | Notes |
|------|-----------|---------|-------|
| **Dienogest** 2 mg/day | Progestogen; direct antiproliferative on ectopic endometrium | High-quality RCT evidence; equivalent to GnRH agonist for pain; no add-back needed | Preferred oral therapy; minimal bone loss; not for fertility use |
| **GnRH agonist** (leuprorelin, goserelin) | Pituitary downregulation → surgical menopause state | Gold standard for pain before/after surgery | **Requires add-back therapy** after 3–6 months (tibolone or oestrogen + progestogen) to prevent bone loss; not for use >6 months without add-back |
| **GnRH antagonist** (elagolix, linzagolix) | Oral; dose-dependent FSH/LH suppression | Newer; ELARIS trials for pain | Titratable; add-back available |
| **Danazol** | Androgen derivative; suppresses HPO axis | Effective but androgenic side effects (acne, hirsutism, voice change) | Rarely used now |
| **Levonorgestrel IUD (Mirena)** | Local endometrial progestogen | Reduces dysmenorrhoea; limited effect on implants outside uterus | Good long-term option |

---

### Surgical Management

#### Endometrioma (Ovarian Cyst)

The key debate: **Drainage + ablation vs Cystectomy**

| Approach | Technique | Recurrence | Ovarian reserve impact |
|---------|-----------|-----------|----------------------|
| **Drainage + ablation** | Empty cyst, ablate lining by laser/diathermy | Higher (30–40% at 1 year) | Less damage to follicles |
| **Laparoscopic cystectomy** (stripping) | Excise cyst wall with cleavage plane | Lower (10–15% at 1 year) | **AMH drop 30–50%** (normal ovarian tissue removed with cyst wall — Somigliana phenomenon) |

**ESHRE/ASRM recommendation:** Cystectomy preferred for cysts ≥3 cm — lower recurrence; despite AMH reduction, IVF outcomes are NOT significantly worse after cystectomy in most studies.

**Special consideration for fertility:** If AMH already low and cystectomy would significantly compromise reserve → consider direct IVF without surgery (or drain under US guidance to retrieve eggs).

#### Peritoneal Disease
- Laparoscopic excision > ablation (deeper excision removes glands, better pain outcomes)
- Obliterated POD: radical excision ('shaving' or full-thickness bowel resection if bowel involved) by experienced surgeon

#### Deep Infiltrating Endometriosis (DIE)
- Requires multidisciplinary approach (colorectal surgeon if rectum involved)
- Urological input if ureters affected
- High surgical morbidity; refer to specialist endometriosis centre

---

### Fertility Management

**Step 1:** Surgical treatment of endometrioma and pelvic disease (improves oocyte access, reduces inflammatory milieu)
**Step 2:** Depending on ovarian reserve (AMH), age, and tubal status:
- If reserve adequate + tubes patent + no male factor → **IUI** (up to 3–6 cycles)
- If reserve borderline / bilateral disease / male factor / >38 years → **IVF directly**
- GnRH agonist down-regulation before IVF in endometriosis may improve outcomes (ultralong protocol — ESHRE guideline)

---

### Follow-Up
- Annual clinical review; CA-125 trend useful for monitoring but not diagnostic
- Counsel that endometriosis is chronic — treatment goal is symptom control and fertility preservation, not cure
- Psychological support — often 7–10 year delay to diagnosis; chronic pelvic pain impacts quality of life profoundly

> ⚕️ *Endometriosis is a progressive disease — early treatment to prevent adhesion formation and endometrioma growth is the most impactful intervention for long-term fertility outcomes.*
""",
    ),

    # 8 — Gestational diabetes
    (
        ["gestational diabetes", "gdm", "ogtt", "insulin", "glucose monitoring", "who 2013", "iadpsg"],
        """## Gestational Diabetes Mellitus — Diagnosis and Full Management

### Diagnosis using WHO 2013 / IADPSG Criteria

**75g OGTT thresholds (venous plasma glucose):**

| Time point | Normal | GDM (≥1 value meets) |
|-----------|--------|----------------------|
| Fasting | <5.1 mmol/L | **≥5.1 mmol/L** |
| 1-hour | <10.0 mmol/L | **≥10.0 mmol/L** |
| 2-hour | <8.5 mmol/L | **≥8.5 mmol/L** |

**This patient:** Fasting 5.4 (≥5.1), 1h 11.1 (≥10.0), 2h 9.2 (≥8.5) → **GDM diagnosed** (all 3 thresholds exceeded — overt GDM picture).

---

### Tier 1: Dietary and Lifestyle Management (Start Immediately)

**Dietary advice:**
- Low glycaemic index carbohydrates (wholegrains, legumes) — avoid refined sugars
- Distribute carbohydrate across 3 meals + 2–3 snacks (prevents postprandial spikes)
- Carbohydrate ~40–45% of total calories
- Dietitian referral within 1 week

**Physical activity:**
- 30 minutes moderate exercise most days (walking after meals blunts postprandial glucose)

---

### Tier 2: Blood Glucose Monitoring Targets (NICE/ADA)

| Measurement | Target |
|------------|--------|
| Fasting | **≤5.3 mmol/L** (NICE) |
| 1 hour post-meal | **≤7.8 mmol/L** |
| 2 hours post-meal | **≤6.4 mmol/L** (if tested) |

**Monitoring frequency:** Fasting + 1h post each main meal = 4 readings/day minimum.

---

### Tier 3: Insulin Initiation

**When to start insulin (NICE 2015):**
- Fasting glucose ≥7.0 mmol/L at diagnosis → **start insulin immediately** (bypass dietary trial)
- Fasting glucose 6.0–6.9 mmol/L → insulin if not controlled within 1–2 weeks of diet
- Post-meal glucose persistently above target after 1–2 weeks dietary modification
- Metformin: alternative to insulin (NICE allows; ACOG considers off-label) — ~50% of GDM women achieve targets on metformin alone

**Insulin Regimen Options:**

| Scenario | Regimen |
|---------|---------|
| **High fasting glucose** | **Bedtime NPH (isophane)** 0.1–0.2 units/kg at 22:00 |
| **High post-meal glucose** | **Rapid-acting analogue** (NovoRapid/Humalog) before meals: 4–6 units, titrate |
| **Both** | **Basal-bolus:** NPH at night + rapid-acting before meals |

**Titration:** Increase by 2 units every 2–3 days until targets met. Review at every visit.

---

### Fetal Surveillance

| Test | Timing/Frequency | Reason |
|------|-----------------|--------|
| Detailed anomaly scan | 20 weeks (if GDM diagnosed later, reassess) | Structural anomalies |
| Fetal growth USS + liquor + Doppler | 28, 32, 36 weeks | Macrosomia (AC >95th percentile), polyhydramnios, IUGR if vascular disease |
| Umbilical artery Doppler | As indicated | If IUGR pattern |
| Kick counts | From 28 weeks | Maternal awareness |

---

### Intrapartum Management

**Timing of delivery:**
- Diet-controlled GDM: deliver by **40+6 weeks** (NICE; same as non-diabetic)
- Insulin-requiring GDM: **38+6 weeks** (NICE; individualise — earlier if macrosomia or poor control)

**Glucose-insulin infusion protocol (if blood glucose >7.0 mmol/L in labour):**
- Start GKI: 10% dextrose 500 mL/h with 10 units regular insulin titrated to glucose 4–7 mmol/L hourly
- Post-delivery: stop insulin immediately after placenta delivered (rapid normalisation)

**Shoulder dystocia counselling:** Macrosomia >4.5 kg → discuss risks; consider planned CS if EFW >4.5 kg (NICE/ACOG).

---

### Postpartum Follow-Up

- Discontinue insulin and glucose monitoring immediately after delivery
- **6-week 75g OGTT** (not HbA1c alone — misses IGT): 50% of GDM women revert to normal; 15–20% have pre-diabetes
- Annual fasting glucose or HbA1c thereafter
- **Lifetime T2DM risk: 50–70%** — lifestyle modification (Mediterranean diet, exercise) reduces risk by 58% (DPP trial equivalent)
- Contraception counselling — ensure adequate glucose control before next pregnancy

> ⚕️ *GDM management is team-based: obstetric team, dietitian, and diabetes nurse/midwife. Set up structured follow-up from diagnosis — uncontrolled GDM doubles the rate of macrosomia, birth injury, and neonatal hypoglycaemia.*
""",
    ),

    # 9 — Operative vaginal delivery
    (
        ["forceps", "vacuum", "ventouse", "operative vaginal", "second stage", "pajot", "subgaleal", "cephalhaematoma"],
        """## Operative Vaginal Delivery — Vacuum vs Forceps

### Prerequisites (BOTH instruments — ALL must be confirmed)
1. **Fully dilated cervix** (10 cm)
2. **Ruptured membranes**
3. **Fetal head engaged** — presenting part ≤0 station (at or below ischial spines)
4. **Position known** — must know exact position before applying instrument
5. **Bladder empty** — catheterise immediately before
6. **Adequate analgesia** — epidural (ideal) or pudendal block (vacuum at low station)
7. **Consent** — maternal consent obtained
8. **Paediatrician present** (especially for mid-cavity deliveries)
9. **Operator trained** in the specific instrument selected
10. **Back-up plan** — if instrument fails → emergency CS; theatre must be immediately available

---

### Vacuum vs Forceps — When to Choose

| Indication | Vacuum (Ventouse) | Forceps |
|-----------|------------------|---------|
| **Position uncertainty** | Not ideal | Forceps preferred (examine station precisely before applying) |
| **Need for rotation** | Kiwi/Bird cup — rotate ≤45° | Kjelland forceps for rotation (specialist only) |
| **Premature (<34 weeks)** | **Avoid** — scalp too fragile; subgaleal risk | Preferred |
| **Face presentation** | Contraindicated | Possible with specific type |
| **After-coming head in breech** | Contraindicated | **Preferred (Piper forceps)** |
| **Rapid delivery needed** | Slower (if cup dislodges) | Faster — more traction control |
| **Regional anaesthesia unavailable** | Vacuum possible with pudendal | Requires adequate analgesia |
| **Maternal effort impaired** (exhaustion) | Either | Forceps offer more traction |

---

### Forceps Application — Traction Technique (Pajot's Manoeuvre)
1. Apply left blade first (into left maternal side), then right blade
2. Lock blades — recheck position, station, and blade application
3. **Pajot's manoeuvre (axis traction):**
   - One hand on shanks applying traction in axis of the pelvic curve
   - Other hand (Pajot's hand) placed on shank top — presses DOWN to direct traction along pelvic axis
   - Traction only during contractions with maternal pushing
4. Disengage blades as head crowns; deliver head and shoulders normally

---

### Maternal Complications

| Complication | Vacuum | Forceps |
|-------------|--------|---------|
| Perineal trauma (3rd/4th degree tear) | Less (8%) | More (**19%** vs vacuum) |
| Maternal pelvic floor injury | Less | More |
| Cervical/vaginal lacerations | Less | More |
| Failed instrument | 15% fail → CS | Lower fail rate |

---

### Neonatal Complications

| Complication | Vacuum | Forceps |
|-------------|--------|---------|
| Cephalhaematoma (subperiosteal) | More common (10%) | Less common |
| **Subgaleal haemorrhage** | **Higher risk — life-threatening** | Lower risk |
| Facial nerve palsy | Uncommon | More common (pressure on facial nerve) |
| Scalp lacerations | More | Less |
| Intracranial injury | Similar (rare) | Similar |
| Neonatal jaundice | More | Less |

### ⚠️ Subgaleal Haemorrhage — The Dangerous One
- Blood accumulates between epicranial aponeurosis and periosteum — **no tissue boundary to limit expansion**
- Can hold the **entire neonatal blood volume** (250–300 mL)
- Signs: boggy scalp swelling crossing suture lines, progressive pallor, tachycardia, poor tone
- Onset: Hours after delivery
- Management: Treat haemorrhagic shock; CT head; paediatric ICU; FFP if coagulopathy
- **Any baby with vacuum delivery: examine scalp every hour for 4 hours post-delivery**

---

### When to Abandon
- If no descent after 3 pulls with correct technique
- Instrument dislodges twice (vacuum)
- Estimated fetal weight >4 kg and instrument not advancing
- **Proceed to emergency CS** — sequential instruments (vacuum then forceps) are associated with higher fetal injury and are generally not recommended

> ⚕️ *The most dangerous error in operative delivery is proceeding without knowing the exact fetal position. Take time to establish position — a mis-applied instrument causes the complications.*
""",
    ),

    # 10 — Cervical cancer screening
    (
        ["cervical screening", "pap smear", "hpv", "colposcopy", "cin", "lletz", "leep", "acus", "hsil", "lsil"],
        """## Cervical Cancer Screening — HPV Testing, Cytology, and Colposcopy

### Screening Programme (UK/NHS; note US differences)

| Parameter | UK (NHS CSPS) | USA (ACOG/USPSTF) |
|-----------|-------------|------------------|
| Start age | **25 years** | **21 years** |
| 25–49 years | HPV primary screening every **5 years** | Pap smear every 3 years OR Pap+HPV co-test every 5 years (age 30+) |
| 50–64 years | HPV every **5 years** | Co-test every 5 years or Pap every 3 years |
| Stop age | **64 years** (if last 3 normal) | **65 years** (if adequate prior screening) |

---

### HPV Primary Screening vs Co-testing vs Cytology Alone

| Method | Sensitivity for CIN2+ | Specificity | Colposcopy referral rate |
|--------|----------------------|-------------|--------------------------|
| Cytology alone | ~55–70% | High | Low |
| **HPV primary** | **90–97%** | Moderate (↑ referrals) | Higher |
| Co-testing (HPV+Cytology) | ~99% | Moderate | Moderate |

**HPV primary screening is preferred** (NHSCSP, UK) — highest sensitivity; negative HPV = very low cancer risk over 5 years.

---

### Interpreting Results

| Result | Meaning | Action |
|--------|---------|--------|
| **HPV negative** | Very low cancer risk | Routine recall (5 years) |
| **HPV positive, cytology negative** | HPV infection without cellular changes | Repeat HPV at **12 months** |
| **HPV positive, cytology inadequate** | — | Repeat at 3 months |
| **HPV positive + ASC-US or LSIL** | Low-grade changes | **Colposcopy** |
| **HPV positive + HSIL/ASC-H** | High-grade changes | **Urgent colposcopy (2-week wait)** |
| **Abnormal glandular cells (AGC)** | Endocervical/endometrial concern | **Colposcopy + endometrial sampling** |

---

### Colposcopy — What the Colposcopist Sees

**Procedure:** Acetic acid (3–5%) applied → abnormal cells turn acetowhite (↑ nuclear protein density).

**Colposcopic features:**

| Feature | Significance |
|---------|-------------|
| **Acetowhite epithelium** | Abnormal nuclear:cytoplasmic ratio — degree and density correlates with CIN grade |
| **Punctation** | Dilated capillary loops running vertically to surface — fine = CIN1, coarse = CIN2/3 |
| **Mosaic pattern** | Capillaries arranged in tiles — coarse mosaic = high-grade |
| **Atypical vessels** | Irregular branching, hairpin vessels → **cancer until proven otherwise** |
| **Iodine (Lugol's) test** | Normal glycogen-rich cells stain mahogany brown; abnormal cells = iodine-pale ("Schiller positive") |

**CIN Classification:**
- **CIN1** (LSIL equivalent): mild dysplasia; 60% regress spontaneously; observe
- **CIN2** (HSIL): moderate dysplasia; treat (especially if confirmed, >25 years)
- **CIN3** (HSIL): severe dysplasia/carcinoma-in-situ; **always treat**
- **Microinvasive carcinoma**: LLETZ + cone biopsy for staging

---

### LLETZ (Large Loop Excision of Transformation Zone) / LEEP

**Indications:**
- CIN2 or CIN3 confirmed on biopsy
- Inadequate colposcopy (transformation zone not fully visualised)
- High-grade colposcopic appearance with positive HPV (treatment at colposcopy — "see and treat")

**Technique:**
- Loop diathermy electrode; size depends on transformation zone type
- Under local anaesthesia (1% lidocaine + adrenaline into cervical stroma)
- Excise transformation zone to depth ≥5–7 mm (to include cervical crypts)

**Histology:** Always send specimen — margin status is key (clear margins → risk of residual disease 1–3%; involved margins → ~20% residual)

**Obstetric consequences:**
- Single LLETZ: RR of preterm birth 1.7× (excision >10 mm depth doubles risk further)
- Cervical incompetence and second-trimester loss
- Counsel about cervical length surveillance in future pregnancy (16–24 weeks; progesterone/cerclage if <25 mm)

---

### Follow-Up After Treatment
- Test of cure: HPV test at **6 months** post-LLETZ
  - HPV negative at 6 months → return to routine 5-year recall
  - HPV positive at 6 months → colposcopy
- 10-year surveillance if HPV positive at 6 months, regardless of cytology

> ⚕️ *HPV vaccination (Gardasil 9: HPV types 6, 11, 16, 18, 31, 33, 45, 52, 58) prevents 90% of cervical cancers. Screening remains essential — vaccine does not cover all oncogenic HPV types, and many women were not vaccinated before exposure.*
""",
    ),
]


def _find_mock_answer(message: str) -> str | None:
    """Return the best-matching mock answer for the message, or None if no match."""
    q_lower = message.lower()
    best_score = 0
    best_answer = None
    for triggers, answer in MOCK_ANSWERS:
        score = sum(1 for t in triggers if t.lower() in q_lower)
        if score > best_score:
            best_score = score
            best_answer = answer
    return best_answer if best_score >= 2 else None


def _get_vllm_client() -> OpenAI | None:
    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key="NONE")
        models = client.models.list()
        if models.data:
            return client
    except Exception:
        pass
    return None


# ── Speech-to-text (Whisper large-v3-turbo via transformers) ─────────────────
# Uses openai/whisper-large-v3-turbo — same encoder as large-v3 (best accuracy)
# with a 4-layer decoder (vs 32-layer). RTFx ~200x. Works on CUDA 13 / PyTorch 2.9
# via transformers. faster-whisper/WhisperX are NOT used — both require CUDA 12.x
# (CTranslate2 has no CUDA 13 wheel as of Apr 2026).

_asr_pipeline = None


def _get_asr_pipeline():
    global _asr_pipeline
    if _asr_pipeline is None:
        import torch
        from transformers import pipeline as hf_pipeline
        _asr_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device="cuda",
        )
    return _asr_pipeline


def transcribe_audio(audio: tuple | None) -> str:
    """Transcribe microphone audio (sample_rate, numpy_array) → text via Whisper."""
    if audio is None:
        return ""
    try:
        import numpy as np
        sr, data = audio
        # Normalise int16 → float32
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype != np.float32:
            data = data.astype(np.float32)
        # Stereo → mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        pipe = _get_asr_pipeline()
        result = pipe({"raw": data, "sampling_rate": sr})
        return result.get("text", "").strip()
    except Exception as e:
        return f"[Transcription error: {e}]"


def _model_status() -> tuple[str, str]:
    """Return (status_text, color_css_class)."""
    client = _get_vllm_client()
    if client:
        try:
            models = client.models.list()
            name = models.data[0].id if models.data else "unknown"
            return f"✅  Model online: {name}", "status-online"
        except Exception:
            pass
    return "⏳  Model not yet serving (training planned after PediatricianGemma ~Apr 22 2026)", "status-training"


def respond(message: str, history: list) -> str:
    """Gradio streaming chat function."""
    client = _get_vllm_client()
    if client is None:
        mock = _find_mock_answer(message)
        if mock:
            yield mock
        else:
            yield (
                "⏳ **GynecologistGemma is not yet trained.**\n\n"
                "Training is planned immediately after PediatricianGemma completes (~Apr 22 2026).\n\n"
                "Please try one of the example questions on the right — those have "
                "detailed preview answers available."
            )
        return

    try:
        models = client.models.list()
        model_id = models.data[0].id
    except Exception:
        yield "Error: could not retrieve model name from vLLM."
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        role = msg.get("role") if isinstance(msg, dict) else None
        raw_content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(raw_content, list):
            text = next((c for c in raw_content if isinstance(c, str)), None)
        else:
            text = str(raw_content) if raw_content else None
        if role in ("user", "assistant") and text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            stream=True,
        )
        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full_response += delta
                yield full_response
    except Exception as e:
        yield f"Error calling model: {e}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
.status-online   { color: #16a34a; font-weight: 600; }
.status-training { color: #d97706; font-weight: 600; }
.model-badge     { background: #7c3aed; color: white; padding: 4px 12px;
                   border-radius: 999px; font-size: 0.85em; font-weight: 600; }
.specialty-tag   { background: #fdf4ff; border: 1px solid #d8b4fe; color: #6b21a8;
                   padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
#chatbox         { height: 520px; }
.example-btn     { text-align: left !important; font-size: 0.82em !important; }
.audio-input     { border: 1px dashed #c4b5fd !important; border-radius: 8px !important;
                   background: #faf5ff !important; }
"""

DESCRIPTION = """
## GynecologistGemma
<span class="model-badge">Gemma-4-31B</span>&nbsp;
<span class="specialty-tag">Obstetrics & Gynaecology</span>

**Base model:** `google/gemma-4-31b-it` &nbsp;|&nbsp;
**Fine-tuned on:** 50,000 OB/GYN Q&A cases (antenatal care · labour & delivery · gynaecological conditions · reproductive medicine · menopause) &nbsp;|&nbsp;
**Served via:** vLLM (OpenAI-compatible)

> ⚕️ For clinical decision support only. Always consult a licensed obstetrician or gynaecologist for actual medical decisions.
"""


def build_interface():
    status_text, _ = _model_status()

    with gr.Blocks(title="GynecologistGemma") as demo:

        gr.Markdown(DESCRIPTION)
        gr.Markdown(f"**Status:** {status_text}")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="GynecologistGemma",
                    elem_id="chatbox",
                    render_markdown=True,
                    avatar_images=(None, "🩺"),
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask an obstetrics or gynaecology clinical question…",
                        show_label=False,
                        scale=9,
                        container=False,
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 Or speak your question — click the mic, speak, then stop recording",
                        show_download_button=False,
                        elem_classes=["audio-input"],
                    )

                with gr.Row():
                    clear_btn = gr.Button("Clear conversation", size="sm")

            with gr.Column(scale=1):
                gr.Markdown("### 📋 Example Questions")
                gr.Markdown(
                    "*Click any question to load it. These are representative of the "
                    "50k OB/GYN cases the model was trained on.*"
                )
                for q in EXAMPLE_QUESTIONS:
                    display = q[0][:120] + "…" if len(q[0]) > 120 else q[0]
                    gr.Button(display, size="sm", elem_classes=["example-btn"]).click(
                        fn=lambda text=q[0]: text,
                        outputs=msg_box,
                    )

        def user_submit(message, history):
            return "", history + [{"role": "user", "content": message}]

        def bot_respond(history):
            if not history:
                return history
            raw = history[-1]["content"]
            if isinstance(raw, list):
                user_msg = next((c for c in raw if isinstance(c, str)), str(raw))
            else:
                user_msg = str(raw)
            history = history + [{"role": "assistant", "content": ""}]
            for partial in respond(user_msg, history[:-1]):
                history[-1]["content"] = partial
                yield history

        msg_box.submit(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue=False).then(
            bot_respond, chatbot, chatbot
        )
        send_btn.click(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue=False).then(
            bot_respond, chatbot, chatbot
        )
        clear_btn.click(lambda: [], outputs=chatbot)

        # Voice input: transcribe when recording stops, put text in msg_box
        audio_input.stop_recording(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[msg_box],
        )

    return demo
