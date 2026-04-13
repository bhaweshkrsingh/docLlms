"""
PediatricianGemma — Gradio chat interface.

Talks directly to the vLLM endpoint (no backend API required to run the UI standalone).
When the backend is up, it can optionally use /api/chat for routing.

Launch:
  python frontend/gradio/pediatrician/launch.py
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

GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7920"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8101/v1")

SYSTEM_PROMPT = """You are PediatricianGemma, an AI assistant fine-tuned on 50,000 pediatric
medical cases covering neonatal care, child development, vaccinations, congenital conditions,
pediatric emergencies, adolescent medicine, and pediatric pharmacology.

Provide detailed, clinically structured, and accurate answers grounded in standard pediatric
practice (Nelson's Textbook, AAP guidelines, ACIP recommendations). When discussing medications,
include weight-based dosing formulas. When discussing developmental concerns, reference appropriate
milestone tables.

Always remind users to consult a licensed pediatrician for actual clinical decisions."""

# ── Example questions — deep/specific, drawn from training data topics ──────

EXAMPLE_QUESTIONS = [
    # Well-child / developmental
    [
        "A 14-month-old boy has no words and doesn't point to objects. His mother says he doesn't "
        "consistently respond to his name. What developmental red flags are present, how do you "
        "evaluate him, and what are your differential diagnoses for this presentation?"
    ],
    [
        "Compare the expected developmental milestones for a healthy child at 6 months, 12 months, "
        "18 months, and 24 months across the four domains: gross motor, fine motor, language, "
        "and social/adaptive. What single red flag at each age should prompt immediate referral?"
    ],
    # Neonatal
    [
        "A 36-hour-old term neonate has a total serum bilirubin of 18 mg/dL. His birth weight was "
        "3.4 kg and he is exclusively breastfed. How do you assess the need for phototherapy versus "
        "exchange transfusion, and what are the risk factors that lower the intervention threshold?"
    ],
    [
        "An 8-week-old premature infant (32 weeks GA) in the NICU develops increasing abdominal "
        "distension, bloody stools, feeding intolerance, and temperature instability. X-ray shows "
        "pneumatosis intestinalis with portal venous gas. Outline the diagnosis, staging (Bell's "
        "criteria), and full management including surgical indications."
    ],
    # Emergencies
    [
        "A 2-year-old presents with a 3-hour history of sudden-onset inconsolable crying, drawing "
        "up his legs in paroxysms every 15–20 minutes, vomiting, and has now passed a red 'currant "
        "jelly' stool. What is the diagnosis, how do you confirm it, and what is your management "
        "algorithm including when to proceed to surgical intervention?"
    ],
    [
        "A 5-year-old boy has had 5 days of fever (≥38.5°C), bilateral non-purulent conjunctivitis, "
        "cracked lips, strawberry tongue, a polymorphous truncal rash, and swollen, erythematous "
        "hands and feet. ESR 95, CRP 14, echo shows mild dilation of the left anterior descending "
        "coronary artery. Confirm the diagnosis, explain the Kobayashi score, and detail the "
        "IVIG + aspirin protocol. What coronary artery z-score thresholds guide long-term "
        "anticoagulation decisions?"
    ],
    [
        "An 18-month-old is brought in after a 2-minute generalised tonic-clonic seizure with fever "
        "(39.4°C). She is now post-ictal but rousable. How do you distinguish a simple febrile "
        "seizure from a complex one? What is the recommended workup, and what do you tell the parents "
        "about recurrence risk, driving implications, and whether anti-epileptic therapy is indicated?"
    ],
    # Gastroenterology / Surgery
    [
        "A 6-week-old boy presents with 4 days of progressive projectile non-bilious vomiting after "
        "every feed. He is hungry immediately after vomiting. On exam you palpate an olive-shaped "
        "mass in the right upper quadrant. What is the diagnosis, what ultrasound findings confirm "
        "it, and describe the pre-operative electrolyte correction and the Ramstedt pyloromyotomy procedure."
    ],
    # Oncology / Haematology
    [
        "A 4-year-old girl presents with 6 weeks of progressive fatigue, pallor, and easy bruising. "
        "She has hepatosplenomegaly. CBC: WBC 85,000/µL (80% blasts), Hgb 6.2 g/dL, platelets "
        "18,000/µL. Describe your diagnostic approach including bone marrow biopsy, immunophenotyping, "
        "cytogenetics, and outline the Berlin-Frankfurt-Münster induction chemotherapy protocol for "
        "paediatric ALL."
    ],
    # Infectious disease
    [
        "An unvaccinated 3-year-old presents with 5 days of fever, coryza, conjunctivitis, and "
        "Koplik spots on the buccal mucosa followed by a maculopapular rash starting at the hairline "
        "and spreading cephalocaudally. Confirm the diagnosis and describe in detail: "
        "(1) the major complications including subacute sclerosing panencephalitis, "
        "(2) post-exposure prophylaxis for unvaccinated contacts, "
        "(3) vitamin A supplementation indications."
    ],
    [
        "A 3-month-old infant presents with 2 weeks of paroxysmal coughing episodes followed by "
        "an inspiratory 'whoop', post-tussive vomiting, and cyanosis during coughing fits. "
        "Lymphocyte count is 22,000/µL. What is the diagnosis, which organism and toxin are "
        "responsible, how do you confirm it microbiologically, and what are the treatment and "
        "household contact prophylaxis protocols?"
    ],
    # Vaccination
    [
        "Walk me through the complete ACIP vaccination schedule for a healthy term infant from "
        "birth through 18 months. For each visit, list the vaccines due, the route and site of "
        "administration, the key contraindications, and explain why the schedule is timed the way "
        "it is (maternal antibody waning, immunological maturity). Include catch-up guidance for "
        "a 12-month-old who received only the birth hepatitis B dose."
    ],
    # Nutrition / Growth
    [
        "A 6-month-old breastfed infant has dropped from the 25th to the 8th weight-for-age "
        "percentile since her 4-month visit. She is not yet on solids. Her mother reports "
        "frequent nursing (10×/day) and no vomiting. Physical exam shows mild hypotonia. "
        "Define failure to thrive, classify this as organic vs non-organic, outline your "
        "diagnostic workup, and detail the nutritional rehabilitation plan including timing "
        "of solid food introduction and supplementation."
    ],
    # Pharmacology
    [
        "A 20 kg, 6-year-old child with community-acquired pneumonia (lobar consolidation, "
        "fever 39°C, RR 35/min, SpO₂ 93% on room air) requires antibiotic therapy. "
        "Compare IV ampicillin vs oral amoxicillin vs azithromycin — indications, weight-based "
        "dosing calculations for this child, duration, and criteria for stepping down from IV "
        "to oral. When would you add coverage for atypical organisms?"
    ],
    # Endocrinology
    [
        "A 10-year-old girl has had 3 months of polyuria, polydipsia, and unintentional 4 kg "
        "weight loss. Random glucose is 340 mg/dL, pH 7.28, bicarbonate 14 mEq/L, glucose in "
        "urine, and moderate ketonuria. Distinguish DKA from HHS in paediatrics. Provide the "
        "step-by-step DKA management protocol including fluid resuscitation rates, insulin "
        "infusion, potassium replacement, and monitoring for cerebral oedema."
    ],
]

# ── Mock answers (served when vLLM is not yet up) ────────────────────────────
# Each entry: (keyword_triggers, answer_markdown)
# Matching: score = number of triggers found in the lowercased user message.

MOCK_ANSWERS: list[tuple[list[str], str]] = [

    # 1 — Developmental delay / autism red flags (14-month-old, no words, no pointing)
    (
        ["14-month", "no words", "doesn't point", "name", "developmental"],
        """## Developmental Assessment: 14-Month-Old With No Words and No Pointing

### Red Flags Present
This child has **three simultaneous red flags** for autism spectrum disorder (ASD) and/or language delay:
1. **No words by 12 months** (should have ≥1 meaningful word by 12 months)
2. **No pointing by 12 months** (proto-declarative pointing is a critical joint-attention milestone)
3. **Inconsistent response to name** — a hallmark early sign of ASD

Any single one of these warrants evaluation; all three together make ASD the primary concern until ruled out.

---

### Developmental Milestones Expected at 14 Months
| Domain | Expected | This Child |
|--------|----------|------------|
| Language | 1–3 words + babbling with consonants | ❌ No words |
| Social | Points to share interest, waves bye-bye, imitates | ❌ No pointing |
| Response to name | Reliable 90%+ of the time | ❌ Inconsistent |
| Gross motor | Walking independently (range 9–15 mo) | Not stated |

---

### Evaluation Approach

**1. Validated Screening Tool**
- Administer the **M-CHAT-R/F** (Modified Checklist for Autism in Toddlers, Revised with Follow-up) immediately. A score ≥3 triggers referral.

**2. Hearing Assessment — Do First**
- Refer for **formal audiology** (ABR or conditioned play audiometry). Conductive hearing loss (chronic OME) or sensorineural loss can mimic all these findings.
- Do not wait for audiology before ASD referral — proceed in parallel.

**3. Developmental Paediatrics / Neurology Referral**
- Refer to a **developmental paediatrician or multidisciplinary ASD clinic** — do not wait for school age.

**4. Early Intervention (EI) Referral**
- In the USA, refer immediately under **IDEA Part C** (birth–3 programme). EI services can begin while awaiting formal diagnosis — early speech-language therapy is beneficial regardless of final diagnosis.

**5. History Deep-Dive**
- Regression? (loss of previously acquired skills = red flag for Rett syndrome or Landau-Kleffner)
- Eye contact, social smile, imitation, pretend play
- Repetitive behaviours, sensory sensitivities
- Pre/peri/postnatal history, family history of ASD or language delay

**6. Physical Examination**
- Head circumference (macrocephaly in ~20% of ASD; microcephaly suggests genetic syndrome)
- Dysmorphic features → chromosomal/metabolic workup
- Skin (ash-leaf macules for tuberous sclerosis → TSC is the most common single-gene cause of ASD)
- Neurological tone

---

### Differential Diagnoses (ranked by probability)

1. **Autism Spectrum Disorder (ASD)** — most likely given the triad
2. **Language disorder / late talker** — isolated expressive delay, but lack of pointing argues against simple late talking
3. **Sensorineural or conductive hearing loss** — must exclude first
4. **Global developmental delay** — if motor milestones also delayed
5. **Intellectual disability** (without ASD)
6. **Rett syndrome** — consider in girls if regression present
7. **Fragile X syndrome** — most common inherited cause of intellectual disability; test with FMR1 CGG repeat analysis
8. **Tuberous sclerosis** — check skin under Wood's lamp
9. **Landau-Kleffner syndrome** — acquired epileptic aphasia; EEG if regression present

---

### Key Message to Parents
Reassure that early intervention **before age 3** yields the greatest developmental gains. Avoid the "wait and see" approach — evidence strongly supports acting now.

> ⚕️ *This is educational content. Diagnosis requires formal assessment by a developmental specialist.*
""",
    ),

    # 2 — Developmental milestones 6/12/18/24 months
    (
        ["developmental milestones", "6 months", "12 months", "18 months", "24 months", "gross motor", "fine motor", "language", "social"],
        """## Developmental Milestones: 6, 12, 18, and 24 Months

### Overview
Development is assessed across four domains: **gross motor, fine motor/adaptive, language/communication, and social/personal-adaptive**.

---

### 6 Months
| Domain | Expected Milestones |
|--------|---------------------|
| Gross motor | Sits with support; rolls front-to-back and back-to-front; bears weight on legs when held standing |
| Fine motor | Reaches and grasps with both hands; transfers objects hand-to-hand; raking grasp |
| Language | Babbles with consonants (ba, da, ga); turns toward voice; laughs |
| Social | Recognises familiar faces; stranger anxiety beginning; social smile well established |

🚩 **Red flag at 6 months:** No social smile, not turning to sounds, no babbling, floppy tone

---

### 12 Months
| Domain | Expected Milestones |
|--------|---------------------|
| Gross motor | Pulls to stand; cruises furniture; may take first steps (range 9–15 mo) |
| Fine motor | Pincer grasp (thumb + index finger); releases objects voluntarily; bangs two objects |
| Language | 1–3 meaningful words ("mama", "dada" specific); understands "no"; responds to name |
| Social | Waves bye-bye; points (proto-declarative); separation and stranger anxiety peak |

🚩 **Red flag at 12 months:** No babbling, no gesturing (pointing, waving), no words by 16 months

---

### 18 Months
| Domain | Expected Milestones |
|--------|---------------------|
| Gross motor | Walks independently; stoops and recovers; beginning to run (stiffly) |
| Fine motor | Stacks 2–3 blocks; scribbles spontaneously; turns pages (several at a time); uses spoon |
| Language | ≥10 meaningful words; uses jargon; points to 1–2 body parts on request |
| Social | Parallel play; follows simple one-step commands; feeds self with spoon; symbolic play beginning |

🚩 **Red flag at 18 months:** Fewer than 6 words, no pointing, no imitating actions — triggers M-CHAT-R/F and ASD evaluation

---

### 24 Months
| Domain | Expected Milestones |
|--------|---------------------|
| Gross motor | Runs well; kicks a ball; walks up/down stairs holding rail (two feet per step) |
| Fine motor | Stacks 6 blocks; turns single pages; copies a vertical line |
| Language | ≥50 words; **2-word phrases** ("more milk", "daddy go"); 50% intelligible to strangers |
| Social | Parallel → associative play; uses "I" and "me"; follows 2-step commands; symbolic play established |

🚩 **Red flag at 24 months:** No 2-word phrases, vocabulary <50 words, loss of any previously acquired language or social skill

---

### Single Absolute Red Flags Requiring Immediate Referral (any age)
- **Any regression** of previously acquired skills (language, motor, social)
- No social smile by **2 months**
- No babbling by **12 months**
- No words by **16 months**
- No 2-word phrases by **24 months**
- Any loss of language or social skills at **any age**

> ⚕️ *Milestones are ranges. A single delayed milestone in isolation may be normal. Multiple delays or regression always warrant prompt evaluation.*
""",
    ),

    # 3 — Neonatal jaundice / hyperbilirubinaemia
    (
        ["bilirubin", "neonat", "jaundice", "phototherapy", "exchange transfusion", "breastfed", "36-hour"],
        """## Neonatal Hyperbilirubinaemia: Phototherapy vs Exchange Transfusion

### Clinical Scenario
**36-hour-old term neonate, TSB 18 mg/dL, BW 3.4 kg, exclusively breastfed.**

---

### Step 1: Plot on the AAP Bhutani Nomogram
The **hour-specific bilirubin nomogram** is the cornerstone of management. At 36 hours of age:
- TSB 18 mg/dL plots in the **high-risk zone** (>95th percentile for age in hours)
- This exceeds the phototherapy threshold for a **≥38-week, low-risk infant** (approximately 12–13 mg/dL at 36h)
- **Phototherapy is indicated immediately**

---

### Phototherapy Decision (AAP 2022 Guidelines)
AAP 2022 replaced fixed thresholds with risk-stratified curves based on:
1. **Gestational age** (each week below 38 weeks lowers threshold by ~1 mg/dL)
2. **Neurotoxicity risk factors** (albumin <3.0 g/dL, isoimmune haemolytic disease, G6PD deficiency, sepsis, temperature instability, acidosis)

For this infant at 36h with TSB 18 mg/dL:
- **Phototherapy threshold**: ~12–14 mg/dL → **clearly exceeded → start phototherapy NOW**
- **Exchange transfusion threshold**: ~22–24 mg/dL for low-risk term infant → **not yet reached**

---

### Phototherapy Protocol
- **Intensive phototherapy**: special blue LED light (460–490 nm), irradiance ≥30 µW/cm²/nm
- Maximise skin exposure (remove clothing, eye protection only)
- Continue breastfeeding every 2–3 hours (do not supplement with formula unless dehydrated)
- Recheck TSB in **4–6 hours** to confirm declining trend
- Discontinue when TSB falls ≥4–5 mg/dL below threshold AND infant ≥48h old

---

### Risk Factors That Lower the Intervention Threshold
| Risk Factor | Effect |
|-------------|--------|
| Gestational age 35–36 weeks | Lower threshold by 1–2 mg/dL |
| Isoimmune haemolytic disease (ABO, Rh) | Lower threshold |
| G6PD deficiency | Lower threshold; can have sudden bilirubin spike |
| Sepsis / acidosis | Lower threshold (more permeable BBB) |
| Albumin <3.0 g/dL | More unbound bilirubin → more neurotoxic |
| Clinical jaundice within 24h of birth | Always pathological → immediate investigation |

---

### Exchange Transfusion Indications
- TSB approaching/exceeding exchange threshold despite intensive phototherapy
- Rate of rise >0.5 mg/dL/hour despite phototherapy
- Any signs of **acute bilirubin encephalopathy**: high-pitched cry, lethargy, hypotonia, retrocollis/opisthotonus, fever, poor feeding

Exchange transfusion replaces ~85% of circulating RBCs and removes bilirubin, haemolytic antibodies, and sensitised cells.

---

### Additional Workup for This Infant
- Blood type and Coombs (direct antiglobulin test) — rule out ABO/Rh isoimmunisation
- CBC with reticulocyte count — haemolysis?
- G6PD screen (if ethnic risk: Mediterranean, African, Asian)
- Serum albumin
- If sepsis suspected: blood culture, CRP

> ⚕️ *Management thresholds are individualised. Always use the AAP 2022 interactive tool or Bhutani nomogram with the infant's exact age in hours.*
""",
    ),

    # 4 — NEC
    (
        ["necrotizing enterocolitis", "nec", "pneumatosis", "premature", "32 weeks", "portal venous", "bell"],
        """## Necrotising Enterocolitis (NEC) — Staging, Diagnosis, and Management

### Diagnosis
This presentation is **NEC until proven otherwise**:
- 32-week premature infant (prematurity is the #1 risk factor)
- Abdominal distension, bloody stools, feeding intolerance, temperature instability
- **Pneumatosis intestinalis** (gas in bowel wall) — pathognomonic
- **Portal venous gas** — sign of advanced disease, high mortality

---

### Bell's Staging Criteria (modified)

| Stage | Name | Clinical | Radiological | Management |
|-------|------|----------|--------------|------------|
| I | Suspected NEC | Temperature instability, apnoea, feeding intolerance, mild abdominal distension | Normal or mild ileus | NPO, NG suction, IV antibiotics, serial exams |
| IIA | Definite NEC (mild) | Above + absent bowel sounds, mild tenderness | Pneumatosis intestinalis | NPO 7–10 days, IV antibiotics, TPN |
| IIB | Definite NEC (moderate) | Above + metabolic acidosis, thrombocytopenia | Pneumatosis + portal venous gas | NPO 14 days, broader antibiotics, surgical consult |
| IIIA | Advanced NEC (severe, bowel intact) | Peritonitis, DIC, shock, severe acidosis | Extensive pneumatosis, ascites | Surgical consult, stabilise |
| IIIB | Advanced NEC (perforated) | Sudden deterioration, "football sign" | **Free air** on AXR | **Emergency surgery** |

**This infant is Stage IIB–IIIA** given portal venous gas and systemic signs.

---

### Immediate Management

**1. NPO + NG Decompression**
- Nothing by mouth — stop all enteral feeds immediately
- Pass orogastric tube, leave on low intermittent suction

**2. IV Access + Resuscitation**
- Two IV lines (consider UAC/UVC if not already placed)
- Fluid bolus 10 mL/kg NS for haemodynamic instability; repeat as needed
- Start TPN via central line for 14+ days

**3. IV Antibiotics (broad spectrum)**
- **Ampicillin** 50 mg/kg/dose q8–12h (gram-positive coverage, GBS)
- **Gentamicin** 4–5 mg/kg/dose q24–48h (gram-negative enteric coverage)
- **Metronidazole** 15 mg/kg loading, then 7.5 mg/kg q12h (anaerobic coverage)
- Duration: 10–14 days for confirmed NEC IIB+

**4. Serial Abdominal Radiographs**
- Every 6 hours initially: look for **free air** (perforation = surgical emergency)
- Left lateral decubitus or cross-table lateral best for detecting free air

**5. Labs (every 6–12 hours)**
- CBC (watch platelets — thrombocytopenia indicates severe disease)
- ABG / VBG — worsening acidosis signals deterioration
- CMP, coagulation panel (DIC screen)
- Blood culture × 2

**6. Respiratory Support**
- Intubate if apnoea worsens or respiratory failure (abdominal distension splints diaphragm)

---

### Surgical Indications (IIIB)
- **Free pneumoperitoneum** — emergency laparotomy
- Clinical deterioration despite maximal medical therapy
- Abdominal wall erythema/induration (suggests perforation/gangrene)
- Fixed abdominal mass

**Surgical options:** resection of necrotic bowel with primary anastomosis vs stoma formation; peritoneal drain as temporising measure in extremely low birth weight infants.

---

### Prognosis
- Overall mortality: 15–30% (higher with perforation)
- **Short bowel syndrome** is the major long-term complication if >50% small bowel resected
- Neurodevelopmental follow-up essential — NEC survivors have higher rates of cognitive impairment

> ⚕️ *NEC is a surgical emergency. Maintain a low threshold for surgical consultation at any stage.*
""",
    ),

    # 5 — Intussusception
    (
        ["intussusception", "currant jelly", "paroxysmal", "legs", "crying", "2-year-old", "2 year old"],
        """## Intussusception — Diagnosis and Management

### Diagnosis
The clinical triad is **classic and diagnostic**:
1. **Paroxysmal colicky pain** (drawing up legs every 15–20 min) — due to peristalsis against the obstructed segment
2. **Vomiting** (initially non-bilious, becomes bilious with prolonged obstruction)
3. **"Currant jelly" stool** — blood and mucus from ischaemic bowel mucosa (late finding, present in only 50–60%)

**Most common age:** 3 months – 6 years; **peak 5–10 months.**
**Most common type:** ileocolic (90%) at the ileocaecal valve.
**Pathological lead point** (Meckel's diverticulum, polyp, lymphoma) more likely in children >5 years or adults.

---

### Confirmation

**1. Ultrasound (Investigation of Choice)**
- Sensitivity 98–100%, specificity 88–99%
- **"Target sign"** (transverse) or **"pseudokidney sign"** (longitudinal)
- Can detect free fluid (perforation risk)
- No radiation — preferred over contrast enema for diagnosis

**2. Plain AXR**
- May show paucity of bowel gas in RLQ, soft tissue mass, or features of obstruction
- Not diagnostic; useful to exclude perforation before enema

---

### Management Algorithm

```
Haemodynamically stable?
       │
   Yes ├──► IV access + fluid resuscitation
       │    NPO + NG tube
       │    Surgical team on standby
       │
       ├──► Ultrasound confirms diagnosis?
       │                │
       │             Yes ├──► Pneumatic or hydrostatic enema reduction
       │                │
       │             No  └──► Reconsider diagnosis
       │
   No  └──► Resuscitate first, then urgent imaging/surgical consult
```

---

### Non-Operative Reduction (First-Line)

**Pneumatic enema (air enema)** — preferred at most paediatric centres:
- Air insufflated per rectum under fluoroscopic or US guidance
- Pressure ≤120 mmHg in three attempts
- **Success rate 70–90%** in uncomplicated cases
- Signs of successful reduction: sudden release of air into terminal ileum, free flow of gas through ileocaecal valve, cessation of pain

**Hydrostatic enema** (water-soluble contrast or saline under US guidance):
- Alternative when fluoroscopy unavailable
- Slightly lower success rate

**Contraindications to enema reduction:**
- Free pneumoperitoneum (perforation)
- Peritonitis
- Haemodynamic instability unresponsive to resuscitation
- Prolonged symptoms >48–72h with signs of bowel ischaemia

---

### Surgical Indications
- Failed enema reduction (after 2–3 attempts)
- Perforation / peritonitis
- Pathological lead point suspected
- **Procedure:** manual reduction; if not possible → resection ± primary anastomosis

---

### Post-Reduction Care
- Admit for observation 24 hours
- **Recurrence rate 5–10%** after enema, 1–3% after surgical reduction
- Parents counselled: return immediately if symptoms recur

> ⚕️ *Do not delay enema in a stable child — the window for non-operative reduction closes with increasing duration of symptoms and bowel ischaemia.*
""",
    ),

    # 6 — Kawasaki disease
    (
        ["kawasaki", "conjunctivitis", "strawberry tongue", "coronary", "ivig", "aspirin", "fever 5 days"],
        """## Kawasaki Disease — Diagnosis, Kobayashi Score, and Management

### Diagnosis
This child meets **complete Kawasaki disease criteria**: fever ≥5 days PLUS ≥4 of 5 principal features.

**Principal Features (CRASH mnemonic):**
| Feature | This Patient |
|---------|-------------|
| **C**onjunctival injection (bilateral, non-purulent) | ✓ |
| **R**ash (polymorphous, truncal) | ✓ |
| **A**denopathy (cervical, >1.5 cm, usually unilateral) | Not stated |
| **S**trawberry tongue / lip changes (cracked, erythematous) | ✓ |
| **H**ands/feet (oedema, erythema; periungual desquamation in convalescence) | ✓ |

Echo finding of **coronary artery dilation (LAD z-score >2)** + all clinical features = treat without hesitation.

---

### Kobayashi Score (Predicts IVIG Resistance)
Score ≥5 predicts non-response to first IVIG dose (~requiring rescue therapy):

| Parameter | Points |
|-----------|--------|
| Na ≤133 mEq/L | 2 |
| Illness day at diagnosis ≤4 | 2 |
| AST ≥100 U/L | 2 |
| Days of fever before treatment | 1 per day >4 |
| CRP ≥10 mg/dL | 1 |
| Age <12 months | 1 |
| Platelet ≤300 × 10³/µL | 1 |

**Score ≥5** → consider primary combination therapy (IVIG + corticosteroids or infliximab).

---

### Treatment Protocol

**Phase 1: Acute (within 10 days of fever onset — URGENTLY)**

**IVIG 2 g/kg IV** over 10–12 hours (single infusion) — reduces coronary aneurysm risk from ~25% to ~4%.

**Aspirin** (dual role, dose changes by phase):
- **Acute phase**: High-dose **80–100 mg/kg/day ÷ q6h** (anti-inflammatory) until afebrile for 48h
- **Subacute/convalescent phase**: Low-dose **3–5 mg/kg/day once daily** (antiplatelet) — continue until echo normal at 6–8 weeks

**Phase 2: If IVIG-resistant** (fever persists/recurs ≥36h after IVIG):
- Second IVIG dose 2 g/kg
- OR **Infliximab** 5 mg/kg IV (preferred at many centres for second-line)
- OR **Methylprednisolone** 30 mg/kg IV × 3 days then prednisolone taper

---

### Coronary Artery Z-Score Thresholds and Long-Term Management

| Z-score | Classification | Management |
|---------|---------------|------------|
| <2 | Normal | Low-dose aspirin × 6–8 weeks, then stop |
| 2–2.5 | Dilation | Low-dose aspirin; echo at 2 weeks, 6 weeks, 1 year |
| 2.5–5 | Small aneurysm | Low-dose aspirin; echo every 3–6 months |
| 5–10 | Medium aneurysm | Aspirin ± anticoagulation (warfarin or LMWH) |
| **>10** | **Giant aneurysm** | **Warfarin (INR 2–3) + aspirin; highest thrombosis risk**; cardiology co-management; activity restriction |

**This child** (mild LAD dilation, likely z-score 2–2.5): standard IVIG + aspirin protocol; likely resolution with timely treatment.

---

### Follow-Up Echo Schedule (no/mild coronary involvement)
- **Baseline** (at diagnosis)
- **2 weeks** post-treatment
- **6–8 weeks** (if normal → discontinue aspirin)
- **1 year** (if all normal → discharge from cardiac follow-up)

> ⚕️ *Start IVIG within 10 days of fever onset — this is the window for preventing coronary artery aneurysms. After day 10, benefit is less certain but still treat active disease.*
""",
    ),

    # 7 — Febrile seizure
    (
        ["febrile seizure", "18-month", "post-ictal", "simple", "complex", "recurrence", "anti-epileptic"],
        """## Febrile Seizure — Classification, Workup, and Counselling

### Classification: Simple vs Complex

| Feature | Simple Febrile Seizure | Complex Febrile Seizure |
|---------|----------------------|------------------------|
| Duration | **<15 minutes** | ≥15 minutes (febrile status epilepticus) |
| Type | Generalised tonic-clonic | Focal onset, or focal features |
| Recurrence in 24h | **None** | Recurs within same febrile illness |
| Post-ictal period | Brief (<1h) | Prolonged (>1h) |
| Neurological exam | Normal | May have Todd's paresis or abnormality |

**This child (2-min GTC, post-ictal, no focal features, temperature 39.4°C) = Simple febrile seizure.**

Simple febrile seizures are **benign, common (2–5% of children 6mo–5yr), and require no acute investigation beyond identifying the source of fever.**

---

### Workup

**Simple febrile seizure in a well-appearing child ≥18 months with known fever source:**
- ✅ **Clinical evaluation only** — no LP, no EEG, no neuroimaging routinely indicated (AAP guidelines)
- ✅ Identify and treat the **cause of fever** (exam for AOM, pharyngitis, UTI, viral illness)

**When to deviate (do LP / neuroimaging):**
- Meningismus or signs of CNS infection
- Fully vaccinated? If not → lower LP threshold (Hib, pneumococcal meningitis risk)
- **Complex** febrile seizure → consider MRI brain (especially if focal)
- Age <12 months → LP more strongly considered (meningitis harder to exclude clinically)
- Persistent altered mental status beyond expected post-ictal period
- Focal neurological signs

**EEG:** Not indicated after simple febrile seizure. May be considered after complex or recurrent complex seizures.

---

### Recurrence Risk

| Risk Factor | Effect on Recurrence |
|-------------|---------------------|
| Age <18 months at first seizure | ↑ risk |
| Family history of febrile seizures | ↑ risk |
| Low-grade fever at time of seizure | ↑ risk |
| Short duration between fever onset and seizure | ↑ risk |

- **No risk factors**: ~15% recurrence
- **1–2 risk factors**: ~25–30%
- **≥3 risk factors**: ~50–65%

**This child (18 months, single episode)**: approximately 30–35% chance of a second febrile seizure.

---

### Risk of Epilepsy
- Simple febrile seizure → epilepsy risk is **1–2%** (barely above general population of 0.5–1%)
- Complex febrile seizure → epilepsy risk 4–6%
- Multiple complex febrile seizures → higher (10–15%)

---

### Anti-Epileptic Prophylaxis

**Not recommended** for simple febrile seizures:
- Continuous AED prophylaxis (phenobarbital, valproate) — side effects outweigh benefit
- Antipyretics do NOT prevent recurrent febrile seizures (Cochrane evidence)

**Rescue therapy (diazepam rectal 0.5 mg/kg or midazolam buccal 0.3 mg/kg):**
- May be prescribed for families with high anxiety, remote access to care
- Used only if seizure lasts >5 minutes

---

### Parent Counselling (Key Points)
1. **"This is not epilepsy"** — most children outgrow febrile seizures by age 5–6
2. **What to do if it happens again:** time the seizure, place child on their side, do not put anything in the mouth, call 999/911 if >5 minutes or child doesn't recover
3. Fever control (ibuprofen / paracetamol) for comfort — does not prevent seizure recurrence
4. No driving/activity restrictions

> ⚕️ *The key clinical skill is distinguishing simple from complex — this determines the workup and follow-up intensity.*
""",
    ),

    # 8 — Pyloric stenosis
    (
        ["pyloric stenosis", "projectile vomiting", "olive", "6-week", "pyloromyotomy", "alkalosis", "non-bilious"],
        """## Hypertrophic Pyloric Stenosis

### Diagnosis
**Classic presentation confirmed:** 6-week-old male, progressive projectile non-bilious post-prandial vomiting, hungry immediately after, olive-shaped mass in RUQ. This is hypertrophic pyloric stenosis (HPS) until proven otherwise.

- **Epidemiology:** 2–3/1000 live births; M:F = 4:1; firstborn males predominant; familial tendency
- **Pathophysiology:** Progressive hypertrophy of pyloric circular muscle → gastric outlet obstruction

---

### Ultrasound Findings (Investigation of Choice)
Pyloric ultrasound has **>97% sensitivity and specificity**:

| Measurement | Normal | HPS |
|-------------|--------|-----|
| **Pyloric muscle thickness** | <3 mm | **≥4 mm** (diagnostic) |
| **Pyloric channel length** | <14 mm | **≥17 mm** |
| **Pyloric diameter** | <13 mm | ≥14 mm |

Additional US signs: failure of antrum to empty ("antral nipple sign"), real-time lack of fluid passage through pylorus.

Upper GI contrast series (showing "string sign," "shoulder sign") is used if US is inconclusive.

---

### Pre-Operative Electrolyte Correction — CRITICAL

HPS causes **hypochloraemic hypokalaemic metabolic alkalosis** (the classic electrolyte picture):
- Vomiting → loss of HCl → ↑ HCO₃⁻, ↑ pH, ↓ Cl⁻
- Kidney conserves Na⁺ at expense of H⁺ and K⁺ → hypokalaemia, paradoxical aciduria

**Pre-op targets (do NOT operate until achieved):**
| Parameter | Target |
|-----------|--------|
| Serum bicarbonate | <26–30 mEq/L |
| Serum chloride | >100 mEq/L |
| Serum potassium | >3.5 mEq/L |
| Urine output | >1 mL/kg/h |

**IV fluid correction:**
- Start: **0.9% NaCl + 5% dextrose** at 1.5× maintenance
- Add **KCl 20–40 mEq/L** once urine output confirmed
- Correction typically takes **24–48 hours**
- NPO; consider NG tube to low suction if vomiting profuse
- Recheck electrolytes every 8–12 hours

---

### Ramstedt Pyloromyotomy

**Procedure:** Longitudinal incision through pyloric serosa and circular muscle down to mucosa, without entering the mucosa. Relieves obstruction without resection.

**Approaches:**
- Open (right upper quadrant or circumumbilical incision) — gold standard for decades
- Laparoscopic — equivalent outcomes, now preferred at most centres; 3-mm instruments, faster recovery, better cosmesis

**Key intraoperative step:** After myotomy, insufflate air via NG to confirm mucosal integrity (no bubbles = intact mucosa).

**Post-op feeding:**
- Begin **ad libitum feeding 6–8 hours** post-op (vs traditional graduated protocol — no difference in outcomes, Cochrane)
- Expect some vomiting in first 24–48h (mucosal oedema) — normal
- Usually home in 24–48 hours

**Complications (<5%):** Mucosal perforation (intra-op), incomplete myotomy (persistent vomiting), wound infection.

---

### Outcome
Near-100% cure rate. Long-term outcomes are excellent; no recurrence of obstruction.

> ⚕️ *The electrolyte correction is as important as the surgery. Operating on an infant with uncorrected alkalosis risks anaesthetic complications. Never rush to the OR.*
""",
    ),

    # 9 — Paediatric ALL
    (
        ["leukemia", "all", "blasts", "bfm", "chemotherapy", "bone marrow", "4-year-old", "wbc 85"],
        """## Paediatric Acute Lymphoblastic Leukaemia (ALL)

### Diagnostic Approach

**This presentation is ALL until proven otherwise:** 4-year-old, fatigue, pallor, bruising, hepatosplenomegaly, WBC 85,000/µL with 80% blasts, Hgb 6.2 g/dL, platelets 18,000/µL.

---

### Step 1: Stabilisation Before Workup
- Haematology/oncology referral **today**
- Transfuse pRBC if Hgb <7 or symptomatic — but **do NOT transfuse platelets** unless active bleeding or invasive procedure (functional platelet count may be better than number suggests)
- Monitor for **tumour lysis syndrome (TLS):** uric acid, LDH, phosphate, Ca²⁺, creatinine; IV hydration, allopurinol
- Avoid fever workup delays — these children are profoundly immunosuppressed

### Step 2: Diagnostic Workup

**Peripheral blood smear:** Confirm blasts morphology (lymphoblasts vs myeloblasts)

**Bone marrow biopsy and aspiration (definitive):**
- ≥25% lymphoblasts in marrow = ALL by WHO criteria
- Provides material for all downstream testing

**Immunophenotyping (flow cytometry) — classifies ALL type:**
| Type | Frequency | Key markers |
|------|-----------|-------------|
| B-ALL (precursor B) | ~85% | CD19, CD10, CD22, TdT |
| T-ALL | ~15% | CD3, CD7, CD5, TdT |

**Cytogenetics and FISH:**
- **Hyperdiploidy (>50 chromosomes)** → favourable prognosis
- **ETV6-RUNX1 (TEL-AML1) t(12;21)** → most common translocation; favourable
- **BCR-ABL1 t(9;22) "Philadelphia chromosome"** → high-risk; add TKI (imatinib/dasatinib)
- **KMT2A (MLL) rearrangements** → infant ALL; poor prognosis
- **iAMP21** → high-risk; intensified therapy

**Minimal residual disease (MRD)** by PCR or flow at Day 29 of induction: key prognostic marker and treatment modifier.

**CNS staging:** Lumbar puncture (combined with first intrathecal chemotherapy)
- CNS1: no blasts
- CNS2: <5 WBC + blasts
- CNS3 (CNS disease): ≥5 WBC + blasts, or cranial nerve palsy → intensified CNS treatment

**Additional:** chest X-ray (mediastinal mass in T-ALL), renal function, LFTs, coagulation

---

### Berlin-Frankfurt-Münster (BFM) Induction Protocol

The **BFM backbone** is used by most paediatric oncology groups (COG in USA uses similar):

**Phase 1A (weeks 1–4) — Four-drug induction:**

| Drug | Dose | Route | Days |
|------|------|-------|------|
| **Prednisolone** | 60 mg/m²/day | PO | Days 1–28 |
| **Vincristine** | 1.5 mg/m² (max 2 mg) | IV | Days 8, 15, 22, 29 |
| **Daunorubicin** | 30 mg/m² | IV | Days 8, 15, 22, 29 |
| **PEG-asparaginase** | 2500 IU/m² | IM | Day 12 (or 15) |
| **Intrathecal MTX** | Age-based | IT | Days 1, 15, 29 |

**Goal of induction:** achieve **complete remission** (blasts <5% in marrow) and MRD negativity by Day 29.

**Phase 1B (weeks 5–8):** Cyclophosphamide, 6-mercaptopurine, cytarabine, intrathecal therapy.

**Consolidation and maintenance** follow based on risk stratification (standard vs high risk).

---

### Risk Stratification (COG)
- **Standard risk:** B-ALL, age 1–9.99y, WBC <50,000, no CNS disease, favourable cytogenetics, MRD-negative Day 29
- **High risk:** age ≥10y, WBC ≥50,000, unfavourable cytogenetics, CNS disease, MRD-positive

**This patient:** WBC 85,000 → **high-risk** based on WBC alone; cytogenetics will further define.

---

### Overall Prognosis
- Paediatric B-ALL overall survival: **>90%** with modern therapy
- High-risk B-ALL: ~80–85% OS
- T-ALL: ~75–85% OS
- Ph+ ALL with TKI: outcomes approaching standard-risk

> ⚕️ *Immediate haematology-oncology involvement is essential. Do not delay diagnostic workup — TLS can develop rapidly as treatment begins.*
""",
    ),

    # 10 — Measles
    (
        ["measles", "koplik", "unvaccinated", "cephalocaudal", "subacute sclerosing", "sspe", "vitamin a"],
        """## Measles (Rubeola) — Complications, PEP, and Vitamin A

### Diagnosis Confirmed
**Measles (rubeola):** fever → prodrome (coryza, cough, conjunctivitis — the 3 Cs) → Koplik spots (pathognomonic, appear 1–2 days before rash) → maculopapular rash beginning at **hairline/behind ears**, spreading **cephalocaudally** to trunk then extremities over 3 days.

- Caused by **measles morbillivirus** (single-stranded RNA paramyxovirus)
- Infectious 4 days before to 4 days after rash onset
- Notifiable disease — report to public health immediately

---

### Major Complications

**1. Pulmonary**
- **Pneumonia** (most common cause of measles death): primary giant cell pneumonia or secondary bacterial (S. pneumoniae, S. aureus, H. influenzae)
- Croup, bronchiolitis

**2. Neurological**
- **Acute post-infectious encephalitis:** 1–3 days after rash, 1/1000 cases; 15% mortality, 25% permanent neurological damage. Mechanism: autoimmune demyelination.
- **Measles inclusion body encephalitis (MIBE):** 1–7 months post-infection; in immunocompromised patients; often fatal.
- **Subacute Sclerosing Panencephalitis (SSPE):**
  - Rare (1/10,000), latency 7–10 years after primary measles
  - Progressive, invariably fatal neurodegenerative disease
  - Stages: personality change → seizures → myoclonic jerks (periodic, every 5–8 seconds) → rigidity → coma → death
  - EEG: **Radermecker complexes** (periodic high-voltage complexes)
  - CSF: markedly elevated measles antibody titres
  - No effective treatment; isoprinosine may slow progression
  - Vaccination is the only prevention — **risk is eliminated in vaccinated individuals**

**3. Gastrointestinal**
- Diarrhoea and dehydration (leading cause of mortality in low-income settings)
- Appendicitis (measles can involve lymphoid tissue)

**4. Ear**
- Otitis media (1/10 cases)

**5. Immunosuppression ("Immune Amnesia")**
- Measles destroys pre-existing immune memory cells for 2–3 years post-infection
- Increases susceptibility to other infections long after recovery

---

### Post-Exposure Prophylaxis (PEP)

**Window:** Within 72 hours of exposure for MMR, within 6 days for IVIG.

| Contact Type | PEP | Timing |
|-------------|-----|--------|
| Healthy unvaccinated ≥12 months | **MMR vaccine** | Within 72h of exposure |
| Infants 6–11 months | MMR vaccine | Within 72h (re-vaccinate at 12–15m and 4–6y) |
| Immunocompromised (any age) | **IVIG 400 mg/kg IV** | Within 6 days |
| Infants <6 months | **IVIG 400 mg/kg IV** | Within 6 days |
| Pregnant unvaccinated | **IVIG 400 mg/kg IV** | Within 6 days |

Exposed unvaccinated healthy contacts ≥12 months who don't receive PEP within 72h should be excluded from school/childcare for 21 days.

---

### Vitamin A Supplementation

**WHO/AAP Indications (reduces mortality up to 50% in high-risk populations):**
- Age <2 years hospitalised with measles
- Immunodeficiency
- Evidence of vitamin A deficiency (xerophthalmia, Bitot's spots, night blindness)
- Children from areas where measles case fatality rate ≥1% (resource-limited settings)

**Dosing (WHO):**
| Age | Dose | Duration |
|-----|------|----------|
| <6 months | 50,000 IU | Once daily × 2 days |
| 6–11 months | 100,000 IU | Once daily × 2 days |
| ≥12 months | **200,000 IU** | Once daily × 2 days |
| Any age with ophthalmological signs | Repeat dose at 4 weeks |

Vitamin A is essential for epithelial integrity and immune function; deficiency worsens measles severity.

> ⚕️ *Measles is a notifiable disease. Isolate the patient (airborne precautions), report to public health, and initiate contact tracing immediately.*
""",
    ),

    # 11 — Pertussis (whooping cough)
    (
        ["pertussis", "whoop", "paroxysmal", "3-month", "lymphocyte", "bordetella", "azithromycin prophylaxis"],
        """## Pertussis (Whooping Cough) — Diagnosis and Management

### Diagnosis

**Classic pertussis in an unimmunised 3-month-old infant:**
- Paroxysmal cough → inspiratory whoop → post-tussive vomiting + cyanosis
- **Lymphocytosis 22,000/µL** — lymphocyte count >20,000/µL is characteristic (pertussis toxin causes lymphocyte recirculation)
- Infants <6 months may not have the classic whoop — presentation is **apnoea** and cyanosis with cough paroxysms

**Note:** Infants <2 months have the highest morbidity and mortality from pertussis — critical illness with pulmonary hypertension and lymphocytosis >100,000/µL ("pertussis hyperleucocytosis") can be fatal.

---

### Causative Organism and Toxin

**Organism:** *Bordetella pertussis* — small Gram-negative coccobacillus

**Key virulence factors:**
| Toxin/Factor | Role |
|-------------|------|
| **Pertussis toxin (PT)** | Inhibits Gi proteins → ↑ cAMP → lymphocyte recirculation (causes lymphocytosis), sensitisation to histamine; major vaccine antigen |
| Filamentous haemagglutinin (FHA) | Adhesion to respiratory epithelium |
| Adenylate cyclase toxin | Inhibits neutrophil/macrophage killing |
| Tracheal cytotoxin | Destroys ciliated epithelium → prolonged cough ("100-day cough") |
| Dermonecrotic toxin | Local tissue damage |

---

### Microbiological Confirmation

**1. Nasopharyngeal PCR (Investigation of Choice)**
- Sensitivity 97%, specificity >99% in first 3 weeks of illness
- NP swab or aspirate — posterior nasopharynx, not anterior nares
- Remains positive up to 4 weeks after cough onset

**2. Nasopharyngeal Culture**
- Gold standard for surveillance but slow (7–10 days), sensitivity lower (30–60%)
- Requires special media (Regan-Lowe, Bordet-Gengou)
- Best yield in catarrhal phase (before paroxysmal phase)

**3. Serology (anti-PT IgG)**
- Useful for diagnosis after 2–8 weeks of illness when PCR may be negative
- Single high titre (>100 IU/mL) or fourfold rise in paired sera

---

### Treatment

**Macrolide antibiotics** — eradicate organism, shorten infectivity; limited effect on paroxysms once established.

| Drug | Dose | Duration | Notes |
|------|------|----------|-------|
| **Azithromycin** (preferred) | 10 mg/kg/day (max 500 mg) × 5 days | 5 days | Drug of choice for all ages including infants |
| Clarithromycin | 7.5 mg/kg/dose q12h (max 500 mg/dose) | 7 days | Not for infants <1 month |
| Erythromycin | 10 mg/kg/dose q6h | 14 days | Associated with pyloric stenosis in infants <1 month — avoid |
| TMP-SMX (alternative if macrolide intolerant) | 8 mg/kg/day TMP ÷ q12h | 14 days | Not for <2 months |

**Supportive care:**
- Hospitalise infants <3–4 months, all with apnoea, cyanosis, or hypoxia
- Monitor cardiorespiratory continuously — apnoea monitors
- Minimal stimulation (paroxysms triggered by noise, feeding, suction)
- Humidified oxygen; avoid respiratory secretion suctioning where possible
- Severe lymphocytosis (>100,000) → exchange transfusion (leucapheresis) may reduce pulmonary hypertension risk

---

### Household Contact Prophylaxis

**Post-exposure prophylaxis (PEP) for ALL household contacts regardless of vaccination status:**

| Drug | Dose | Duration |
|------|------|----------|
| **Azithromycin** | Adults 500 mg/day; children 10 mg/kg/day (max 500 mg) | **5 days** |
| Clarithromycin | 7.5 mg/kg/dose q12h (adults 500 mg q12h) | 7 days |

**Administer within 21 days of exposure** (after 21 days, PEP unlikely to prevent disease but still reduces transmission).

**Priority contacts for PEP:**
- Infants <12 months (highest risk)
- Pregnant women in third trimester
- Immunocompromised individuals
- Household contacts who have contact with high-risk individuals

**Vaccination:** Tdap booster for adolescent and adult household contacts who haven't had it in the last 5 years ("cocooning" strategy for infant protection).

> ⚕️ *Report to public health. Pertussis is still a major cause of infant death worldwide despite vaccination — exclude from childcare until 5 days of antibiotics completed.*
""",
    ),

    # 12 — Vaccination schedule
    (
        ["vaccination schedule", "birth", "2 months", "4 months", "6 months", "12 months", "18 months", "acip", "catch-up"],
        """## ACIP Vaccination Schedule: Birth through 18 Months

### Why the Schedule Is Timed as It Is

Two biological principles govern timing:
1. **Maternal antibody waning:** Transplacentally transferred IgG wanes over 4–6 months. Vaccines that would be neutralised by maternal antibodies (e.g. MMR, varicella) are given at 12 months when maternal IgG has largely disappeared.
2. **Immunological maturity:** The infant immune system matures progressively; polysaccharide antigens require T-cell help (hence conjugate vaccines for Hib and pneumococcus); multiple doses are required to build immunological memory.

---

### Schedule by Visit

#### **Birth (before hospital discharge)**
| Vaccine | Route/Site | Key Contraindication |
|---------|-----------|----------------------|
| **Hepatitis B dose 1** | IM, anterolateral thigh | Birth weight <2 kg: delay if mother HBsAg-negative |

*Rationale:* HBV can be transmitted perinatally; early vaccination protects even before exposure risk is apparent.

---

#### **2 Months**
| Vaccine | Route/Site | Notes |
|---------|-----------|-------|
| **DTaP dose 1** | IM, anterolateral thigh | Defer if moderate-severe illness |
| **IPV dose 1** | IM or SQ | |
| **Hib dose 1** | IM, anterolateral thigh | |
| **PCV15 or PCV20 dose 1** | IM | |
| **Rotavirus dose 1** (RV1 or RV5) | Oral | Must start before 15 weeks; contraindicated in SCID or prior intussusception |
| **Hepatitis B dose 2** | IM | |

*Rationale:* Maternal antibody to DTaP antigens has waned; 2-month immune system can now mount T-cell dependent responses to conjugate vaccines.

---

#### **4 Months**
| Vaccine | Route/Site |
|---------|-----------|
| **DTaP dose 2** | IM |
| **IPV dose 2** | IM |
| **Hib dose 2** | IM |
| **PCV15/PCV20 dose 2** | IM |
| **Rotavirus dose 2** | Oral |

*Rationale:* Booster doses 4–8 weeks apart build primary series immunological memory.

---

#### **6 Months**
| Vaccine | Route/Site | Notes |
|---------|-----------|-------|
| **DTaP dose 3** | IM | |
| **IPV dose 3** | IM | Can be given 6–18 months |
| **Hepatitis B dose 3** | IM | Can be given 6–18 months |
| **PCV15/PCV20 dose 3** | IM | |
| **Hib dose 3** | IM | Only needed with PedvaxHIB |
| **Influenza** (annual) | IM or IN | First-time recipients need 2 doses 4 weeks apart |

---

#### **12–15 Months**
| Vaccine | Route/Site | Notes |
|---------|-----------|-------|
| **MMR dose 1** | SQ, upper arm | Delay until ≥12m: maternal IgG would neutralise live virus before |
| **Varicella dose 1** | SQ, upper arm | Same reasoning as MMR |
| **Hepatitis A dose 1** | IM | 2-dose series, 6+ months apart |
| **PCV15/PCV20 dose 4** | IM | |
| **Hib dose 4** | IM | |

*Rationale:* MMR and varicella are live-attenuated vaccines — maternal IgG neutralises them if given before 12 months.

---

#### **15–18 Months**
| Vaccine | Route/Site |
|---------|-----------|
| **DTaP dose 4** | IM |
| **Hepatitis A dose 2** | IM (≥6 months after dose 1) |

---

### Catch-Up: 12-Month-Old With Only Birth Hepatitis B Dose

This child is significantly behind. Using the **ACIP Catch-Up Schedule:**

**Today (12 months):**
- HepB dose 2
- DTaP dose 1
- IPV dose 1
- Hib dose 1
- PCV15/PCV20 dose 1
- MMR dose 1 ← can give now (≥12 months)
- Varicella dose 1 ← can give now
- HepA dose 1
- *(Rotavirus NOT given — max age for last dose is 8 months 0 days)*

**4 weeks later (13 months):**
- DTaP dose 2, IPV dose 2, Hib dose 2, PCV dose 2, HepB dose 3

**≥8 weeks after last dose:**
- DTaP dose 3, PCV dose 3, Hib dose 3

**12–15 months after DTaP/Hib/PCV series:**
- DTaP dose 4, Hib dose 4, PCV dose 4

**≥6 months after HepA dose 1:**
- HepA dose 2

**At 4–6 years:**
- DTaP dose 5, IPV dose 4, MMR dose 2, Varicella dose 2

> ⚕️ *Use the CDC's online catch-up scheduler (cdc.gov/vaccines) for complex schedules. Multiple vaccines can be given at the same visit — no evidence of immune overload.*
""",
    ),

    # 13 — Failure to thrive
    (
        ["failure to thrive", "percentile", "6-month-old", "breastfed", "hypotonia", "nutritional rehabilitation", "solid food"],
        """## Failure to Thrive (FTT) — Classification, Workup, and Nutritional Rehabilitation

### Definition

**Failure to thrive (FTT)** — now preferred term: **"faltering growth"**:
- Weight-for-age **crossing ≥2 major percentile lines** downward on standardised growth chart, OR
- Weight consistently **below 3rd percentile** for age and sex
- **This infant:** dropped from 25th → 8th percentile = 2+ percentile lines crossing → **FTT confirmed**

---

### Classification

**Traditional (aetiological):**
| Category | Description | % of Cases |
|----------|-------------|-----------|
| **Non-organic (functional)** | Inadequate caloric intake or absorption without underlying disease | ~70–80% |
| **Organic** | Underlying medical condition | ~20–30% |
| Mixed | Both components | Common |

**Modern approach:** classify by caloric mechanism:
1. Insufficient intake
2. Increased losses (vomiting, malabsorption)
3. Increased demands (metabolic, cardiopulmonary)

**This infant** — frequent nursing (10×/day), no vomiting, mild hypotonia — suggests:
- **Primary: Insufficient caloric transfer** (poor latch, insufficient milk supply, or breastfeeding difficulty)
- **Concern: Hypotonia** — raises possibility of organic cause (neuromuscular disorder, metabolic disease, hypothyroidism)

---

### Diagnostic Workup

**Step 1: Thorough History**
- Feeding history: latch observed by lactation consultant, feeding duration, maternal breast fullness, wet/dirty nappies
- 3-day feeding diary
- Developmental history (hypotonia pattern)
- Family heights/weights (constitutional small stature?)

**Step 2: Physical Examination**
- Dysmorphic features (chromosome disorders)
- Cardiopulmonary exam (cardiac murmur, tachypnoea)
- Abdominal exam (organomegaly → storage disorders)
- Neurological exam (hypotonia — central vs peripheral?)
- Skin (eczema → food allergy with CMPA)

**Step 3: Laboratory (directed by clinical suspicion)**

*First-tier (all FTT with no clear non-organic cause):*
- CBC, CMP (electrolytes, renal and liver function)
- TFTs (TSH, free T4) — hypothyroidism causes hypotonia and poor growth
- Urinalysis + urine culture (UTI is a common masquerader)
- Coeliac screen (TTG-IgA + total IgA) if solids have been introduced

*Second-tier (if above unrevealing + hypotonia prominent):*
- Creatine kinase (neuromuscular disorder)
- Lactate, ammonia (metabolic disease)
- Chromosomes / microarray
- Sweat chloride test (cystic fibrosis)

---

### Nutritional Rehabilitation Plan

**Goal:** Achieve **catch-up growth** — requires **1.5–2× recommended caloric intake** for age until weight returns to baseline trajectory.

**Immediate (breastfeeding support):**
1. **Lactation consultant assessment** — observe latch in clinic
2. Pre/post-feed weights to quantify milk transfer
3. If transfer inadequate: supplement with expressed breast milk (EBM) or formula after each feed
4. Target: **150–200 mL/kg/day** fluid intake (breastfed infants may need supplementation)

**Caloric targets:**
- Normal 0–6 month infant: ~100–110 kcal/kg/day
- **FTT catch-up: 150–200 kcal/kg/day** using ideal weight for height
- Use caloric fortification if needed: add MCT oil or formula powder to expressed milk

**Solid food introduction (this infant at 6 months):**
- **Now appropriate** — WHO and AAP recommend solids at **6 months** for all infants
- Start iron-rich foods first (iron-fortified cereals, pureed meat)
- Introduce one new food every 3–5 days
- Solids supplement but do not replace breast milk until 12 months
- Vitamin D 400 IU/day (all breastfed infants from birth)
- Iron supplementation: if iron deficient (check ferritin)

**Supplementation:**
- **Vitamin D** 400 IU/day — already indicated for all breastfed infants
- **Iron** — exclusive breastfeeding confers some protection, but check ferritin at 6 months; supplemental iron 1 mg/kg/day if deficient

---

### Growth Monitoring
- Recheck weight in **1–2 weeks** after intervention
- Expect **catch-up weight gain** of 2–3× normal for age
- Plot serial weights on WHO growth chart; assess trajectory, not single points
- Refer to dietitian and consider inpatient admission if no catch-up within 4–6 weeks of adequate outpatient support

> ⚕️ *The hypotonia in this infant warrants a neuromuscular and metabolic evaluation — do not attribute FTT to breastfeeding alone until organic causes are excluded.*
""",
    ),

    # 14 — Antibiotic therapy for CAP
    (
        ["pneumonia", "20 kg", "ampicillin", "amoxicillin", "azithromycin", "atypical", "step down", "antibiotic"],
        """## Antibiotic Therapy for Community-Acquired Pneumonia — 20 kg, 6-Year-Old

### Clinical Assessment
**Indication for hospitalisation confirmed:**
- SpO₂ 93% on room air → hypoxaemia
- RR 35/min → tachypnoea (normal <6yr: ≤40/min; this is borderline severe)
- Fever 39°C
- Lobar consolidation

**Severity classification (WHO/PIDS-IDSA):**
- **Severe CAP** based on hypoxaemia → IV antibiotics + supplemental O₂

---

### Antibiotic Options: Comparison

#### 1. IV Ampicillin (preferred first-line for hospitalised non-severe CAP, normal host)
- **Dose for this child (20 kg):** 50 mg/kg/dose q6h IV = **1000 mg q6h**
- Range: 150–200 mg/kg/day ÷ q6h (max 12 g/day)
- **Spectrum:** S. pneumoniae (primary pathogen in this age group), Streptococcal spp.
- Does NOT cover atypicals (Mycoplasma, Chlamydophila)
- **Indication:** Uncomplicated lobar CAP in 6-year-old where S. pneumoniae is most likely (>80% of bacterial CAP in school-age children)

#### 2. Oral Amoxicillin (outpatient or step-down after IV ampicillin)
- **Dose for this child (20 kg):** 80–90 mg/kg/day ÷ q12h PO = **800–900 mg q12h** (high-dose; max 3 g/day)
- **Use:** Not appropriate as initial therapy given hypoxaemia; appropriate for step-down or mild CAP
- Same spectrum as ampicillin; amoxicillin has better oral bioavailability (90% vs 40% for ampicillin)

#### 3. Azithromycin (atypical coverage)
- **Dose for this child (20 kg):** 10 mg/kg day 1 (200 mg), then 5 mg/kg days 2–5 (100 mg) — 5-day course
- Max: 500 mg/250 mg
- **Spectrum:** Mycoplasma pneumoniae, Chlamydophila pneumoniae, Legionella — NOT reliable S. pneumoniae coverage (25–30% macrolide resistance)
- **Indication:** Atypical CAP, or addition to beta-lactam when atypicals suspected (see below)

---

### Step-Down from IV to Oral

**Criteria for transition to oral amoxicillin:**
1. Afebrile × 12–24 hours
2. SpO₂ ≥95% on room air
3. Tolerating oral intake
4. Clinical improvement (RR normalising, improving work of breathing)

Typically achievable in **48–72 hours** with appropriate IV therapy.

**Step-down:** IV ampicillin → oral amoxicillin 80–90 mg/kg/day ÷ q12h for total course of **5–7 days** (PIDS 2011 guidelines support shorter courses of 5 days for uncomplicated CAP).

---

### When to Add Atypical Coverage

**Mycoplasma pneumoniae** becomes common after age 5 and peaks in school-age children (5–15 years). Add azithromycin if:

- Gradual onset (days of prodrome: headache, malaise, low-grade fever before respiratory symptoms)
- "Walking pneumonia" pattern — child ill but not severely toxic
- Bilateral or patchy infiltrates (vs lobar consolidation → more typical of pneumococcal)
- Extrapulmonary features: rash, haemolytic anaemia, bullous myringitis, cold agglutinins
- No response to beta-lactam after 48 hours
- Age ≥5 years with mild-moderate disease that could be atypical

**For this child** (lobar consolidation, severe enough for hospitalisation): start ampicillin alone; **reassess at 48h** — add azithromycin if no improvement.

---

### Summary Recommendation for This Child

| Phase | Antibiotic | Dose |
|-------|-----------|------|
| Hospitalised (acute) | IV Ampicillin | 1000 mg q6h IV |
| Step-down (once criteria met) | Oral Amoxicillin | 900 mg q12h PO |
| If atypicals suspected | + Azithromycin | 200 mg day 1, 100 mg days 2–5 |
| **Total course** | 5–7 days | From start of effective therapy |

> ⚕️ *No routine blood culture is required for outpatient CAP. Blood cultures for hospitalised children with severe disease or those not responding to first-line therapy. Empiric therapy covers the most common pathogens by age.*
""",
    ),

    # 15 — DKA management
    (
        ["dka", "diabetic ketoacidosis", "10-year-old", "polyuria", "polydipsia", "glucose 340", "cerebral oedema", "insulin infusion", "bicarbonate"],
        """## Diabetic Ketoacidosis (DKA) in Paediatrics — Full Management Protocol

### Diagnosis
**New-onset Type 1 DM with DKA confirmed:**
- Classic triad: polyuria, polydipsia, weight loss
- Random glucose 340 mg/dL (>200 mg/dL)
- pH 7.28 (acidosis), HCO₃⁻ 14 mEq/L (normal 22–26)
- Glucosuria, ketonuria
- **DKA defined by:** glucose >200 mg/dL + pH <7.3 or HCO₃⁻ <15 + ketonaemia/ketonuria

---

### DKA vs HHS in Paediatrics

| Feature | DKA | HHS |
|---------|-----|-----|
| Onset | Hours–days | Days–weeks |
| Type 1 DM | Yes (usually) | Type 2 DM or mixed |
| Glucose | Usually <600 mg/dL | Often >600 mg/dL |
| pH | <7.3 | Usually >7.3 |
| Bicarbonate | <15 mEq/L | Normal or mildly low |
| Ketones | Moderate–large | Minimal or absent |
| Osmolality | Mildly elevated | Markedly elevated (>320 mOsm/kg) |
| Dehydration | Moderate (~7–10%) | Severe (10–15%) |
| CNS | Less common | More common (obtundation, seizures) |
| Treatment | Insulin + fluids + electrolytes | Fluids primarily (cautious insulin) |

---

### DKA Management Protocol (ISPAD/BSPED Guidelines)

#### Step 1: Assessment and Stabilisation
- Weight, vital signs, neurological status (GCS)
- Calculate **% dehydration:** mild 3–5%, moderate 5–10% (most DKA), severe >10%
- **Severity classification:**
  - Mild: pH 7.2–7.3, HCO₃⁻ 10–15
  - Moderate: pH 7.1–7.2, HCO₃⁻ 5–10
  - Severe: pH <7.1, HCO₃⁻ <5

#### Step 2: Fluid Resuscitation
**Assume 5–7% dehydration** (deficit ~500–700 mL in this ~35 kg child, estimated by age).

**Bolus** (only if haemodynamically unstable — shock, prolonged capillary refill):
- **10 mL/kg 0.9% NaCl** over 30–60 minutes
- Repeat once if needed
- **Avoid large fluid boluses** — associated with cerebral oedema risk

**Rehydration fluid (after bolus):**
- Replace **deficit + maintenance** evenly over **48 hours** (NOT 24h — slower rehydration reduces cerebral oedema risk)
- **Fluid: 0.9% NaCl initially**, transitioning to 0.45% NaCl + 5% dextrose once glucose <300 mg/dL (to prevent hypoglycaemia while insulin continues)

**Example for 35 kg child (moderate DKA, 7% dehydrated):**
- Deficit: 0.07 × 35 kg × 1000 mL = 2,450 mL
- Maintenance (Holliday-Segar): (10×100) + (10×50) + (15×20) = 1,800 mL/24h → 3,600 mL/48h
- Total over 48h: 2,450 + 3,600 = 6,050 mL → **~126 mL/hour**
- Subtract any bolus given

#### Step 3: Insulin Infusion
**Do NOT start insulin in the first hour** — first establish IV access and begin fluids.

**Start insulin at:** **0.05–0.1 unit/kg/hour** (continuous IV infusion)
- Use 0.05 u/kg/h for severe DKA or young child (reduces cerebral oedema risk)
- For this child (35 kg): **0.05 × 35 = 1.75 units/hour**

**Prepare infusion:** 50 units regular insulin in 50 mL 0.9% NaCl = 1 unit/mL

**Target glucose fall:** 50–100 mg/dL/hour — faster drop is not better
- **When glucose reaches 250–300 mg/dL:** add dextrose to IV fluids (0.45% NaCl + 5% dextrose)
- Do NOT stop insulin until pH >7.3, HCO₃⁻ >15, ketones cleared — continue at lower rate with dextrose

#### Step 4: Potassium Replacement — CRITICAL
DKA causes total body potassium depletion despite normal/elevated serum K⁺ (insulin and acidosis correct → K⁺ shifts intracellularly → hypokalaemia if not replaced).

| Serum K⁺ | Action |
|----------|--------|
| <3.0 mEq/L | **Hold insulin**, give KCl 0.5 mEq/kg IV over 1h before starting insulin |
| 3.0–5.5 mEq/L | Add **40 mEq/L KCl** (or 20 mEq KCl + 20 mEq KPO₄) to IV fluids |
| >5.5 mEq/L | Hold potassium; recheck in 1–2h |

**Bicarbonate:** Do NOT give bicarbonate routinely — worsens CNS acidosis, risk of paradoxical CSF acidosis and hypokalaemia. Consider only for life-threatening hyperkalaemia or pH <6.9 with haemodynamic compromise.

---

### Monitoring Protocol

| Parameter | Frequency |
|-----------|-----------|
| Blood glucose | Hourly |
| Electrolytes, BUN, creatinine | Every 2–4 hours |
| Blood gas (VBG acceptable) | Every 2–4 hours until pH >7.3 |
| Neurological observations (GCS, pupils) | **Every hour** — cerebral oedema detection |
| Urine output | Hourly |
| Weight | 4–6 hourly |

---

### Cerebral Oedema — Most Feared Complication

- Occurs in 0.5–1% of paediatric DKA; mortality 20–25%; morbidity in survivors 10–25%
- Typically 4–12 hours after starting treatment
- **Risk factors:** younger age, new-onset DM, severe acidosis, high BUN, large fluid boluses, rapid glucose fall, bicarbonate use

**Warning signs (treat immediately):**
- Sudden deterioration in consciousness (drop in GCS)
- Headache, vomiting (after initial improvement)
- Bradycardia, hypertension (Cushing's triad)
- Papilloedema, pupillary changes

**Treatment of cerebral oedema:**
1. **Mannitol 0.5–1 g/kg IV** over 20 minutes — first-line
2. Or **3% hypertonic saline 2.5–5 mL/kg** over 20–30 minutes
3. Reduce IV fluid rate by 30%
4. Head elevation 30°, minimise stimulation
5. Urgent paediatric ICU + neurosurgery consultation
6. CT head after stabilisation (to assess for herniation)

---

### Transition to Subcutaneous Insulin
Once pH >7.3, HCO₃⁻ >15, ketones resolved, tolerating oral fluids:
- Give first SC dose **30 minutes before** stopping IV insulin
- Start on **multiple daily injection (MDI) regimen** or **insulin pump**
- Diabetes education team involvement essential before discharge

> ⚕️ *DKA is a PICU-level emergency. The two main causes of death are cerebral oedema and hypokalaemia — prevent both with careful fluid management and potassium monitoring.*
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
    # Require at least 2 trigger matches to use a mock (avoids spurious matches)
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
    return "⏳  Model not yet serving (training in progress — finishes ~Apr 17 2026)", "status-training"


def respond(message: str, history: list[list[str]]) -> str:
    """Gradio streaming chat function."""
    client = _get_vllm_client()
    if client is None:
        mock = _find_mock_answer(message)
        if mock:
            yield mock
        else:
            yield (
                "⏳ **PediatricianGemma is still training** (~finishes Apr 17 2026).\n\n"
                "Please try one of the example questions on the right — those have "
                "detailed preview answers available while the model trains."
            )
        return

    try:
        models = client.models.list()
        model_id = models.data[0].id
    except Exception:
        return "Error: could not retrieve model name from vLLM."

    # Build messages
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
.model-badge     { background: #1e40af; color: white; padding: 4px 12px;
                   border-radius: 999px; font-size: 0.85em; font-weight: 600; }
.specialty-tag   { background: #f0fdf4; border: 1px solid #86efac; color: #166534;
                   padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
#chatbox         { height: 520px; }
.example-btn     { text-align: left !important; font-size: 0.82em !important; }
.audio-input     { border: 1px dashed #93c5fd !important; border-radius: 8px !important;
                   background: #eff6ff !important; }
"""

DESCRIPTION = """
## PediatricianGemma
<span class="model-badge">Gemma-4-31B</span>&nbsp;
<span class="specialty-tag">Pediatrics</span>

**Base model:** `google/gemma-4-31b-it` &nbsp;|&nbsp;
**Fine-tuned on:** 50,000 pediatric Q&A cases (neonatal · vaccinations · emergencies · development · pharmacology) &nbsp;|&nbsp;
**Served via:** vLLM (OpenAI-compatible)

> ⚕️ For clinical decision support only. Always consult a licensed pediatrician for actual medical decisions.
"""

STATUS_TRAINING = """
> ⏳ **Model status:** Training in progress — fine-tuning on DGX Spark GB10 (~finishes Apr 17 2026).
> The UI is ready; connect to vLLM once the model finishes training.
> Set `VLLM_BASE_URL` in `.env` and restart.
"""


def build_interface():
    status_text, _ = _model_status()

    with gr.Blocks(title="PediatricianGemma") as demo:

        gr.Markdown(DESCRIPTION)
        gr.Markdown(f"**Status:** {status_text}")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="PediatricianGemma",
                    elem_id="chatbox",
                    render_markdown=True,
                    avatar_images=(None, "🩺"),
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a pediatric clinical question…",
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
                    "50k pediatric cases the model was trained on.*"
                )
                for q in EXAMPLE_QUESTIONS:
                    # Truncate display to 2 lines
                    display = q[0][:120] + "…" if len(q[0]) > 120 else q[0]
                    gr.Button(display, size="sm", elem_classes=["example-btn"]).click(
                        fn=lambda text=q[0]: text,
                        outputs=msg_box,
                    )

        # Wire up interactions
        def user_submit(message, history):
            return "", history + [{"role": "user", "content": message}]

        def bot_respond(history):
            if not history:
                return history
            raw = history[-1]["content"]
            # Gradio 6.x preprocess always wraps content in a list
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
