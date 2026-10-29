"""Generate synthetic labeled training data for Indian PSU + NBFC sentiment."""

import random
import csv
from pathlib import Path

POSITIVE_TEMPLATES = [
    "{company} reports {adj} profit growth of {pct}% in Q{q}",
    "{company} declares {adj} dividend of Rs {num} per share",
    "{company} NPAs fall to {pct}%, lowest in {num} quarters",
    "RBI approves {company} merger with {target}",
    "{company} raises Rs {num} crore via {instrument}",
    "{company} stock surges {pct}% after {event}",
    "{company} gets RBI nod for {product} expansion",
    "{company} asset quality improves, provisions decline {pct}%",
    "{company} CASA ratio rises to {pct}%, boosting margins",
    "Government infuses Rs {num} crore into {company}",
    "{company} sanctions record {num} crore loans under {scheme}",
    "{company} net interest income jumps {pct}% YoY",
    "{company} rating upgraded by {agency} to {grade}",
    "{company} recovers Rs {num} crore from bad loans",
    "{company} expands branch network by {num} in {region}",
    "{company} launches new digital banking platform",
    "{company} garners strong response to {instrument} issue",
    "{company} capital adequacy ratio improves to {pct}%",
    "{company} slippages decline, credit cost normalizes",
    "{company} turnaround story continues with Q{q} beat",
]

NEGATIVE_TEMPLATES = [
    "{company} reports {adj} loss of Rs {num} crore in Q{q}",
    "{company} NPAs jump to {pct}%, highest in {num} quarters",
    "RBI slaps Rs {num} crore penalty on {company} for {violation}",
    "{company} stock crashes {pct}% after {event}",
    "{company} provisioning doubles to Rs {num} crore",
    "{company} fraud detected in {unit}, Rs {num} crore at risk",
    "{company} downgraded by {agency} to {grade}",
    "{company} net interest margin shrinks {pct} bps",
    "{company} slippages spike, fresh NPAs at Rs {num} crore",
    "Government delays recapitalization plan for {company}",
    "{company} under RBI PCA framework, lending restrictions imposed",
    "{company} deposit growth slows to {pct}%, weakest since {year}",
    "{company} employee union threatens strike over {issue}",
    "{company} treasury losses mount to Rs {num} crore",
    "{company} SMA book rises, stress assets climb {pct}%",
    "{company} fails to meet RBI capital norms",
    "{company} moratorium book remains elevated at {pct}%",
    "{company} credit rating outlook revised to negative",
    "{company} merger with {target} faces regulatory hurdles",
    "{company} write-offs surge to Rs {num} crore in Q{q}",
]

NEUTRAL_TEMPLATES = [
    "{company} Q{q} results in line with estimates",
    "{company} net profit flat at Rs {num} crore",
    "{company} to consider dividend at board meeting",
    "RBI maintains repo rate at {pct}%; {company} status quo",
    "{company} appoints {person} as new CEO",
    "{company} opens {num} new branches in {region}",
    "{company} files draft papers for {instrument} issue",
    "{company} extends moratorium facility for {segment}",
    "{company} AGM scheduled for {date}",
    "{company} redeems Rs {num} crore tier-II bonds",
    "{company} changes accounting policy for loan provisioning",
    "{company} receives no objection certificate for co-lending",
    "{company} signs MoU with {target} for co-lending",
    "{company} total deposits stand at Rs {num} lakh crore",
    "{company} advances book grows {pct}% sequentially",
    "{company} board to meet on {date} to discuss {agenda}",
    "{company} credit rating affirmed at {grade} by {agency}",
    "{company} launches {product} for {segment} customers",
    "{company} completes divestment of {unit} stake",
    "{company} maintains branch count at {num} across India",
]

PSU_BANKS = [
    "State Bank of India", "SBI", "PNB", "Punjab National Bank",
    "Bank of Baroda", "Canara Bank", "Union Bank of India",
    "Bank of India", "Indian Bank", "Central Bank of India",
    "Indian Overseas Bank", "Bank of Maharashtra",
    "Punjab and Sind Bank", "UCO Bank",
]

NBFC_NAMES = [
    "Bajaj Finance", "Cholamandalam Finance", "Shriram Finance",
    "Mahindra Finance", "L&T Finance", "SBI Cards",
    "Muthoot Finance", "IIFL Finance", "Tata Capital",
    "Aditya Birla Capital", "Manappuram Finance",
    "CreditAccess Grameen", "Poonawalla Fincorp",
    "Jio Financial Services", "Muthoot Microfin",
]

TARGETS = ["another PSU bank", "a regional lender", "an NBFC peer", "a fintech startup"]
INSTRUMENTS = ["QIP", "rights issue", "tier-II bonds", "AT-1 bonds", "NCDs"]
AGENCIES = ["CRISIL", "ICRA", "CARE", "India Ratings"]
VIOLATIONS = ["KYC norms", "NPA classification", "priority sector lending", "IT security"]
EVENTS_POS = ["strong earnings", "merger approval", "capital infusion", "NPA recovery"]
EVENTS_NEG = ["weak earnings", "RBI penalty", "NPA spike", "fraud disclosure"]
SCHEMES = ["PM Mudra", " Stand-Up India", "Startup India", "housing for all"]
REGIONS = ["rural India", "South India", "North East", "Western states"]
ISSUES = ["wage revision", "privatization fears", "pension reform"]
PRODUCTS = ["gold loan", "MSME loan", "affordable housing", "vehicle finance", "personal loan"]
SEGMENTS = ["retail", "MSME", "agriculture", "salaried"]
PERSONS = ["Mr. Sharma", "Ms. Gupta", "Mr. Reddy", "Ms. Patel"]
UNITS = ["retail banking", "treasury", "international division", "subsidiary"]


def _fill(template: str, is_nbfc: bool = False) -> str:
    company = random.choice(NBFC_NAMES if is_nbfc else PSU_BANKS)
    return template.format(
        company=company,
        target=random.choice(TARGETS),
        instrument=random.choice(INSTRUMENTS),
        agency=random.choice(AGENCIES),
        violation=random.choice(VIOLATIONS),
        event=random.choice(EVENTS_POS if "surge" in template or "jumps" in template else EVENTS_NEG),
        scheme=random.choice(SCHEMES),
        region=random.choice(REGIONS),
        issue=random.choice(ISSUES),
        product=random.choice(PRODUCTS),
        segment=random.choice(SEGMENTS),
        person=random.choice(PERSONS),
        unit=random.choice(UNITS),
        adj=random.choice(["strong", "robust", "healthy", "solid"]) if "profit" in template or "growth" in template else random.choice(["sharp", "heavy", "steep", "significant"]),
        pct=random.choice([2.1, 3.5, 5.2, 7.8, 9.1, 12.3, 15.6, 18.4, 21.5, 25.0]),
        num=random.choice([50, 120, 250, 500, 800, 1200, 2500, 5000, 8000, 12000]),
        q=random.choice([1, 2, 3, 4]),
        year=random.choice([2019, 2020, 2021, 2022, 2023]),
        grade=random.choice(["AAA", "AA+", "AA", "A+", "A", "BBB+"]),
        date="March 2026",
        agenda="dividend and capital plan",
    )


def generate(output_path: str | Path, n_per_class: int = 300, seed: int = 42) -> Path:
    """Generate labeled training CSV."""
    random.seed(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _ in range(n_per_class):
        for tmpl in POSITIVE_TEMPLATES:
            rows.append({"text": _fill(tmpl, is_nbfc=random.random() > 0.5), "label": "positive"})
        for tmpl in NEGATIVE_TEMPLATES:
            rows.append({"text": _fill(tmpl, is_nbfc=random.random() > 0.5), "label": "negative"})
        for tmpl in NEUTRAL_TEMPLATES:
            rows.append({"text": _fill(tmpl, is_nbfc=random.random() > 0.5), "label": "neutral"})

    # Shuffle
    random.shuffle(rows)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} labeled headlines -> {output_path}")
    return output_path


if __name__ == "__main__":
    from src.config.settings import LOCAL_DATA
    generate(LOCAL_DATA / "sentiment_training.csv", n_per_class=300)
