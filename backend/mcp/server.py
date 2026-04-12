"""
MCP (Model Context Protocol) server — exposes medical reference tools
that the specialist agents can call during inference.

Tools:
  - get_vaccination_schedule(age_months, sex)
  - get_pediatric_dosing(drug_name, weight_kg, age_years)
  - get_growth_percentile(weight_kg, height_cm, age_months, sex)
  - get_lab_reference_range(test_name, age_years, sex)

Run standalone:
  python -m backend.mcp.server
"""
from __future__ import annotations

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

import uvicorn
from mcp.server.fastmcp import FastMCP

from backend.config import MCP_HOST, MCP_PORT

mcp = FastMCP("DocLlms Medical Tools")


# ── Vaccination Schedule ────────────────────────────────────────────────────

VACCINATION_SCHEDULE = {
    0:  ["Hepatitis B (HepB) — dose 1"],
    2:  ["DTaP dose 1", "IPV dose 1", "Hib dose 1", "PCV15/PCV20 dose 1", "RV dose 1", "HepB dose 2"],
    4:  ["DTaP dose 2", "IPV dose 2", "Hib dose 2", "PCV15/PCV20 dose 2", "RV dose 2"],
    6:  ["DTaP dose 3", "IPV dose 3 (6–18 mo)", "Hib dose 3 (if needed)", "PCV15/PCV20 dose 3",
         "RV dose 3 (if RotaTeq)", "HepB dose 3 (6–18 mo)", "Influenza (annual, 6 mo+)"],
    12: ["MMR dose 1", "Varicella dose 1", "HepA dose 1 (12–23 mo)", "PCV15/PCV20 dose 4",
         "Hib dose 4 (12–15 mo)"],
    15: ["DTaP dose 4 (15–18 mo)"],
    18: ["HepA dose 2 (18 mo)"],
    48: ["DTaP dose 5 (4–6 yr)", "IPV dose 4 (4–6 yr)", "MMR dose 2 (4–6 yr)",
         "Varicella dose 2 (4–6 yr)"],
    132: ["Tdap (11–12 yr)", "MenACWY dose 1 (11–12 yr)", "HPV series start (11–12 yr)",
          "COVID-19 (per current guidelines)"],
}


@mcp.tool()
def get_vaccination_schedule(age_months: int) -> str:
    """
    Return the ACIP-recommended vaccines due at the given age in months.
    Provide the child's age in whole months (e.g. 2, 4, 6, 12, 15, 18, 48, 132).
    """
    # Find closest schedule entry
    keys = sorted(VACCINATION_SCHEDULE.keys())
    best = min(keys, key=lambda k: abs(k - age_months))
    vaccines = VACCINATION_SCHEDULE.get(best, [])
    age_label = f"{age_months} months" if age_months < 24 else f"{age_months // 12} years"
    if not vaccines:
        return f"No scheduled vaccines defined for age {age_label}. Influenza is recommended annually from 6 months."
    vax_list = "\n".join(f"  • {v}" for v in vaccines)
    return f"Vaccines typically due at ~{age_label} (ACIP schedule):\n{vax_list}\n\nNote: Always verify against current CDC/ACIP guidelines."


# ── Pediatric Drug Dosing ────────────────────────────────────────────────────

DOSING_GUIDE: dict[str, dict] = {
    "amoxicillin": {
        "standard":    "25–45 mg/kg/day ÷ q8–12h (max 1500 mg/day)",
        "otitis_media": "80–90 mg/kg/day ÷ q12h (high-dose for resistant S. pneumo)",
        "form": "250 mg/5 mL or 400 mg/5 mL suspension",
    },
    "ibuprofen": {
        "standard": "5–10 mg/kg/dose q6–8h PRN (max 40 mg/kg/day, max single dose 400 mg)",
        "note": "Use ≥6 months of age only",
        "form": "100 mg/5 mL suspension",
    },
    "acetaminophen": {
        "standard": "10–15 mg/kg/dose q4–6h PRN (max 75 mg/kg/day, max 5 doses/24h)",
        "form": "160 mg/5 mL suspension",
    },
    "azithromycin": {
        "standard": "10 mg/kg day 1, then 5 mg/kg/day × 4 days (max 500 mg/250 mg)",
        "community_pneumonia": "10 mg/kg/day × 5 days",
        "form": "200 mg/5 mL suspension",
    },
    "prednisolone": {
        "croup": "0.6 mg/kg × 1 dose (max 16 mg)",
        "asthma_burst": "1–2 mg/kg/day ÷ q12–24h × 3–5 days (max 60 mg/day)",
        "form": "15 mg/5 mL or 5 mg/5 mL solution",
    },
    "amoxicillin-clavulanate": {
        "standard": "45 mg/kg/day (amoxicillin component) ÷ q12h (max 1750 mg amox/day)",
        "form": "600 mg/5 mL (ES-600) or 400 mg/5 mL suspension",
    },
}


@mcp.tool()
def get_pediatric_dosing(drug_name: str, weight_kg: float, age_years: float) -> str:
    """
    Return weight-based pediatric dosing for a common drug.
    drug_name: e.g. 'amoxicillin', 'ibuprofen', 'acetaminophen', 'azithromycin', 'prednisolone'
    weight_kg: child's weight in kilograms
    age_years: child's age in years
    """
    key = drug_name.lower().strip()
    info = DOSING_GUIDE.get(key)
    if not info:
        available = ", ".join(DOSING_GUIDE.keys())
        return f"Drug '{drug_name}' not in quick-reference database. Available: {available}. Consult a current formulary (e.g. Harriet Lane, BNF for Children)."

    lines = [f"Pediatric dosing — {drug_name.title()} | Weight: {weight_kg} kg | Age: {age_years} yr\n"]
    for label, val in info.items():
        lines.append(f"  {label.replace('_', ' ').title()}: {val}")

    # Calculate example dose for standard dosing if parseable
    std = info.get("standard", "")
    lines.append(f"\n⚠️  Always verify against a current formulary. Doses are general guidelines only.")
    return "\n".join(lines)


# ── Growth Percentile (simplified CDC approximation) ─────────────────────────

@mcp.tool()
def get_growth_info(age_months: int, weight_kg: float, height_cm: float, sex: str) -> str:
    """
    Provide growth assessment context for a child.
    sex: 'male' or 'female'
    Returns approximate WHO/CDC percentile guidance and flags for concern.
    """
    # Simplified expected ranges (not a full CDC lookup — use as clinical context prompt)
    expected = {
        # age_months: (weight_p50_kg, height_p50_cm)
        2:   (5.1, 58.0),
        4:   (6.4, 63.9),
        6:   (7.3, 67.6),
        9:   (8.2, 71.5),
        12:  (9.2, 75.7),
        18:  (10.9, 82.3),
        24:  (12.2, 87.1),
        36:  (14.3, 95.0),
        48:  (16.3, 102.0),
        60:  (18.3, 109.0),
    }
    keys = sorted(expected.keys())
    best_key = min(keys, key=lambda k: abs(k - age_months))
    p50_w, p50_h = expected[best_key]

    w_ratio = weight_kg / p50_w
    h_ratio = height_cm / p50_h

    def classify(ratio):
        if ratio < 0.85:   return "well below average (concern for undernutrition/pathology)"
        if ratio < 0.95:   return "slightly below average"
        if ratio < 1.05:   return "near average (≈50th percentile)"
        if ratio < 1.15:   return "slightly above average"
        return "well above average"

    w_class = classify(w_ratio)
    h_class = classify(h_ratio)

    flags = []
    if w_ratio < 0.85:
        flags.append("⚠️  Weight significantly below median — evaluate for failure to thrive, malnutrition")
    if h_ratio < 0.85:
        flags.append("⚠️  Height significantly below median — consider growth hormone deficiency, hypothyroidism, chronic disease")
    if not flags:
        flags.append("✓  No major growth concerns based on this rough assessment")

    return (
        f"Growth Assessment — Age: {age_months} mo | Sex: {sex}\n"
        f"  Weight: {weight_kg} kg  ({w_class}, median ≈ {p50_w} kg)\n"
        f"  Height: {height_cm} cm  ({h_class}, median ≈ {p50_h} cm)\n\n"
        + "\n".join(flags) +
        "\n\n📋  Use CDC/WHO growth charts for precise percentiles. This tool provides rough guidance only."
    )


# ── Lab Reference Ranges ─────────────────────────────────────────────────────

LAB_RANGES: dict[str, dict] = {
    "hemoglobin": {
        "neonate_term": "14–20 g/dL",
        "2_months":     "9–14 g/dL  (physiologic nadir)",
        "6mo_2yr":      "10.5–13.5 g/dL",
        "2_12yr":       "11.5–15.5 g/dL",
        "adolescent_male": "13–16 g/dL",
        "adolescent_female": "12–16 g/dL",
    },
    "wbc": {
        "neonate":  "9,000–30,000 /µL",
        "1_month":  "5,000–19,500 /µL",
        "1_3yr":    "6,000–17,500 /µL",
        "4_8yr":    "5,000–15,000 /µL",
        "9_12yr":   "4,500–13,500 /µL",
        "adolescent": "4,500–11,000 /µL",
    },
    "creatinine": {
        "neonate":    "0.3–1.0 mg/dL  (reflects maternal)",
        "infant":     "0.2–0.4 mg/dL",
        "child_2_6yr": "0.3–0.5 mg/dL",
        "child_7_12yr": "0.5–0.8 mg/dL",
        "adolescent": "0.6–1.2 mg/dL",
    },
    "sodium": {"all_ages": "135–145 mEq/L"},
    "potassium": {
        "neonate":    "3.7–5.9 mEq/L",
        "infant":     "4.1–5.3 mEq/L",
        "child_adult": "3.5–5.0 mEq/L",
    },
    "glucose": {
        "neonate_first_24h": "40–60 mg/dL  (lower acceptable threshold in well neonates)",
        "after_24h_child": "60–100 mg/dL (fasting)",
    },
    "tsh": {
        "neonate_1_4days": "1–39 mIU/L  (elevated at birth, falls rapidly)",
        "1mo_1yr":   "0.8–6.3 mIU/L",
        "1_6yr":     "0.7–5.7 mIU/L",
        "7_11yr":    "0.6–4.8 mIU/L",
        "adolescent": "0.5–4.3 mIU/L",
    },
}


@mcp.tool()
def get_lab_reference_range(test_name: str, age_years: float, sex: str = "any") -> str:
    """
    Return pediatric reference ranges for a lab test.
    test_name: e.g. 'hemoglobin', 'wbc', 'creatinine', 'sodium', 'potassium', 'glucose', 'tsh'
    age_years: child's age in years
    sex: 'male', 'female', or 'any'
    """
    key = test_name.lower().strip()
    info = LAB_RANGES.get(key)
    if not info:
        available = ", ".join(LAB_RANGES.keys())
        return f"'{test_name}' not in quick-reference. Available: {available}."

    lines = [f"Pediatric reference ranges — {test_name.upper()} | Age: {age_years} yr | Sex: {sex}\n"]
    for age_group, val in info.items():
        lines.append(f"  {age_group.replace('_', ' ').title()}: {val}")
    lines.append("\n📋  Values from Nelson's / Harriet Lane. Confirm with your lab's own reference ranges.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(f"Starting MCP server on {MCP_HOST}:{MCP_PORT}{MCP_ENDPOINT}")
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT, path=MCP_ENDPOINT)
