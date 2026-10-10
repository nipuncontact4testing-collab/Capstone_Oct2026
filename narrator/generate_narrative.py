"""Part 3: GenAI-powered insight narrator (SCR = Situation, Complication, Resolution).

Reads narrator/findings.json (written by analysis/clean_and_eda.py), builds a prompt
from it, asks Gemini for a business narrative, and verifies the result:

    findings.json --> build_prompt --> Gemini  --\
                                                  +--> accuracy check --> sample_output.txt
                      offline template (no key) --/

Nothing numeric is typed in this file: every figure in the prompt, in the offline
template and in the accuracy check comes from findings.json.

Usage (from the repo root):
    python narrator/generate_narrative.py              # Gemini if a key is set, else offline
    python narrator/generate_narrative.py --offline    # force the template path
    python narrator/generate_narrative.py --show-prompt

Environment variables:
    GEMINI_API_KEY (or GOOGLE_API_KEY)  free Google AI Studio key - never commit it
    GEMINI_MODEL                        optional override, default below
Exit code: 0 = all five key figures verified, 1 = accuracy check failed.
"""
import argparse
import calendar
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
FINDINGS_PATH = HERE / "findings.json"
DEFAULT_OUT = HERE / "sample_output.txt"

DEFAULT_MODEL = "gemini-2.5-flash"
# A factual business report should be deterministic, not creatively varied.
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2048  # >= 1024 so the narrative is never cut off
TIMEOUT_MS = 30000        # HttpOptions timeout is in MILLISECONDS (30 s)

SYSTEM_INSTRUCTION = (
    "You are a senior data analyst writing for Mamaearth's regional ops and finance heads. "
    "Write a concise narrative with exactly three labeled sections: 'Situation', "
    "'Complication', and 'Resolution'. Every number in the output must come from the supplied "
    "FINDINGS and appear with the same value; do not invent statistics, estimate, round "
    "differently, convert values, or perform new arithmetic. Copy each supplied figure exactly, "
    "including its unit and decimals. Call out that a blended average hides a concentrated "
    "problem. Give 3 to 4 concrete actions for ops and finance, each tied to a finding. Use "
    "plain business English, no markdown tables, and stay under 350 words."
)


# --------------------------------------------------------------------------------------
# Formatting helpers - the ONE place a number becomes text
# --------------------------------------------------------------------------------------
def fmt(value, kind):
    if kind == "inr":
        return f"INR {value:,.2f}"
    if kind == "pct":
        return f"{value:.1f}%"
    if kind == "int":
        return f"{int(value):,}"
    return str(value)


def load_findings(path=FINDINGS_PATH):
    path = Path(path)
    if not path.exists():
        sys.exit(f"{path} not found. Run first:  python analysis/clean_and_eda.py")
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(findings):
    """Task 2: the prompt is assembled from findings.json, never hardcoded."""
    rates = findings["return_rate_by_payment"]
    segment = findings["highest_risk_segment"]
    true_peak = findings["true_peak_month"]
    inflated = findings["outlier_inflated_month"]
    lines = [
        f"- Cleaned total revenue: {fmt(findings['cleaned_total_revenue_inr'], 'inr')}",
        f"- Raw total revenue: {fmt(findings['raw_total_revenue_inr'], 'inr')}",
        f"- Duplicate reconciliation delta: {fmt(findings['duplicate_reconciliation_delta_inr'], 'inr')}",
        f"- Return rate by payment: COD {fmt(rates['COD'], 'pct')}, "
        f"CARD {fmt(rates['CARD'], 'pct')}, UPI {fmt(rates['UPI'], 'pct')}",
        f"- Highest-risk segment: {segment['payment_method']} in tier-{segment['city_tier']} "
        f"({fmt(segment['return_rate_pct'], 'pct')} return rate)",
        f"- True peak month: {true_peak['month']} at {fmt(true_peak['revenue_inr'], 'inr')}",
        f"- Outlier-inflated month: {inflated['month']} appears to have "
        f"{fmt(inflated['apparent_revenue_inr'], 'inr')} revenue; corrected revenue is "
        f"{fmt(inflated['corrected_revenue_inr'], 'inr')}",
    ]
    return (
        "Write the SCR narrative for this analysis of Mamaearth order data.\n\n"
        "FINDINGS (verified, computed by code):\n" + "\n".join(lines) + "\n\n"
        "Answer this question: where are returns really coming from, and what is the true "
        "revenue picture once the data is cleaned?"
    )


# --------------------------------------------------------------------------------------
# Task 2 + 3: Gemini call with parameters and error handling
# --------------------------------------------------------------------------------------
def generate_scr_narrative(findings: dict, *, model=None, api_key=None) -> dict:
    """Generate an SCR narrative, falling back offline if Gemini is unavailable."""
    try:
        model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            return generate_scr_narrative_offline(findings)

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        config = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        if "2.5" in model:  # 2.5 models spend output tokens on hidden "thinking"; switch it off
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        response = client.models.generate_content(
            model=model,
            contents=build_prompt(findings),
            config=types.GenerateContentConfig(**config),
        )
        text = (response.text or "").strip()
        finish = ""
        if getattr(response, "candidates", None):
            finish = str(getattr(response.candidates[0], "finish_reason", "") or "")
        if not text:
            raise ValueError(f"empty response (finish_reason={finish or 'unknown'})")
        if "MAX_TOKENS" in finish:
            raise ValueError("response truncated (finish_reason=MAX_TOKENS)")
        usage = getattr(response, "usage_metadata", None)
        tokens = getattr(usage, "total_token_count", None)
        checks = _required_figure_checks(text, findings)
        if not all(check["found"] for check in checks):
            print("[narrator] Gemini response missed required figures; using offline fallback.")
            return generate_scr_narrative_offline(findings)
        return {"status": "success", "narrative": text, "tokens": tokens}
    except Exception as err:  # network, quota, timeout, bad model name, missing SDK ...
        print(f"[narrator] Gemini unavailable -> using offline fallback. Reason: {err}")
        return generate_scr_narrative_offline(findings)


def call_gemini(findings, model=None, api_key=None):
    """Backward-compatible alias for generate_scr_narrative."""
    return generate_scr_narrative(findings, model=model, api_key=api_key)


# --------------------------------------------------------------------------------------
# Task 4: offline fallback - needs no key and no network
# --------------------------------------------------------------------------------------
def offline_narrative(findings):
    f = findings
    rates = f["return_rate_by_payment"]
    segment = f["highest_risk_segment"]
    true_peak = f["true_peak_month"]
    inflated = f["outlier_inflated_month"]
    true_peak_month_name = calendar.month_name[int(true_peak["month"].split("-")[1])]
    seg = f"{segment['payment_method']} orders in tier-{segment['city_tier']} cities"
    return "\n".join([
        "SITUATION",
        f"After reconciliation, cleaned revenue is {fmt(f['cleaned_total_revenue_inr'], 'inr')} "
        f"versus raw revenue of {fmt(f['raw_total_revenue_inr'], 'inr')}. The "
        f"{fmt(f['duplicate_reconciliation_delta_inr'], 'inr')} difference is attributable "
        "to duplicate orders.",
        "",
        "COMPLICATION",
        f"The payment return rates vary: COD {fmt(rates['COD'], 'pct')}, CARD "
        f"{fmt(rates['CARD'], 'pct')}, and UPI {fmt(rates['UPI'], 'pct')}. The highest-risk "
        f"segment is {seg}, with a {fmt(segment['return_rate_pct'], 'pct')} return rate. "
        f"The outlier-inflated peak is {inflated['month']} at "
        f"{fmt(inflated['apparent_revenue_inr'], 'inr')}; its corrected revenue is "
        f"{fmt(inflated['corrected_revenue_inr'], 'inr')}. The true peak is "
        f"{true_peak_month_name} ({true_peak['month']}) at "
        f"{fmt(true_peak['revenue_inr'], 'inr')}.",
        "",
        "RESOLUTION",
        f"1. Regional ops: investigate {seg} and prioritize interventions there, given its "
        f"{fmt(segment['return_rate_pct'], 'pct')} return rate.",
        "2. Payments: review the COD process first, since its return rate is higher than CARD and UPI.",
        f"3. Finance: use the corrected {inflated['month']} revenue of "
        f"{fmt(inflated['corrected_revenue_inr'], 'inr')} and {true_peak_month_name} peak "
        f"revenue of {fmt(true_peak['revenue_inr'], 'inr')} when planning.",
        "4. Analytics: continue monitoring payment- and segment-level returns alongside monthly revenue.",
    ])


def generate_scr_narrative_offline(findings: dict) -> dict:
    """Return a deterministic SCR narrative without network or API configuration."""
    narrative = offline_narrative(findings)
    checks = _required_figure_checks(narrative, findings)
    assert all(check["found"] for check in checks), "Offline SCR template missed a required figure"
    return {
        "status": "success",
        "narrative": narrative,
        "tokens": 0,
    }


# --------------------------------------------------------------------------------------
# Task 5: numeric accuracy check
# --------------------------------------------------------------------------------------
def check_narrative(text, findings):
    """Print per-figure status after comma normalization; return individual checks."""
    results = _required_figure_checks(text, findings)
    for result in results:
        print(f"  [{'PASS' if result['found'] else 'FAIL'}] {result['label']}: {result['expected']}")
    assert all(result["found"] for result in results), "Narrative is missing one or more required figures"
    return results


def _required_figure_checks(text, findings):
    flat = text.replace(",", "").lower()
    cleaned = findings["cleaned_total_revenue_inr"]
    cod_rate = findings["return_rate_by_payment"]["COD"]
    segment_rate = findings["highest_risk_segment"]["return_rate_pct"]
    delta = findings["duplicate_reconciliation_delta_inr"]
    peak_month = findings["true_peak_month"]["month"]
    peak_revenue = findings["true_peak_month"]["revenue_inr"]
    month_name = calendar.month_name[int(peak_month.split("-")[1])]

    def money_forms(value):
        return (f"{value:,.2f}".replace(",", "").lower(),
                f"{value:.1f}".replace(",", "").lower())

    checks = [
        (
            "Cleaned total revenue",
            f"{cleaned:,.2f} or {cleaned:.1f}",
            any(token in flat for token in money_forms(cleaned)),
        ),
        ("COD return rate", f"{cod_rate:.1f}", f"{cod_rate:.1f}".lower() in flat),
        (
            "Highest-risk segment return rate",
            f"{segment_rate:.1f}",
            f"{segment_rate:.1f}".lower() in flat,
        ),
        (
            "Duplicate reconciliation delta",
            f"{delta:,.2f} or {delta:.1f}",
            any(token in flat for token in money_forms(delta)),
        ),
        (
            "True peak month and revenue",
            f"{month_name} and {peak_revenue:,.2f} or {peak_revenue:.1f}",
            month_name.lower() in flat and any(token in flat for token in money_forms(peak_revenue)),
        ),
    ]
    return [
        {"label": label, "expected": expected, "found": found}
        for label, expected, found in checks
    ]


def unverified_numbers(text, findings):
    """Extra safety net: numbers in the narrative that do not trace back to findings.json."""
    allowed = set()

    def collect(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, (int, float)):
            allowed.update({f"{obj:g}", f"{obj:.1f}", f"{obj:.2f}", str(round(obj)), str(int(obj))})
        elif isinstance(obj, str):
            allowed.update(re.findall(r"\d+(?:\.\d+)?", obj))

    collect(findings)
    allowed.update({"20", "1", "2", "3", "4", "2026"})  # 20% threshold, list numbers, tiers, year
    scrubbed = re.sub(r"\d{4}-\d{2}", "", text.replace(",", ""))
    found = re.findall(r"(?<![\d.])\d+(?:\.\d+)?", scrubbed)
    return sorted({n for n in found if n not in allowed and n.rstrip("0").rstrip(".") not in allowed})


def format_report(checks, extra):
    lines = ["ACCURACY CHECK (key figures vs findings.json)"]
    for check in checks:
        lines.append(f"  [{'PASS' if check['found'] else 'FAIL'}] {check['label']}: {check['expected']}")
    passed = sum(c["found"] for c in checks)
    lines.append(f"  Result: {passed}/{len(checks)} key figures found")
    if extra:
        lines.append(f"  Warning - numbers not traceable to findings.json: {', '.join(extra)}")
    return "\n".join(lines), passed == len(checks)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="skip Gemini, use the template")
    ap.add_argument("--show-prompt", action="store_true", help="print the prompt and exit")
    ap.add_argument("--findings", default=str(FINDINGS_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    findings = load_findings(args.findings)
    if args.show_prompt:
        print("SYSTEM INSTRUCTION:\n" + SYSTEM_INSTRUCTION + "\n\nPROMPT:\n" + build_prompt(findings))
        return 0

    source = "offline template"
    note = "Offline template path (no Gemini call made)."
    if args.offline:
        res = generate_scr_narrative_offline(findings)
    else:
        res = generate_scr_narrative(findings, model=args.model)
        if res["tokens"] != 0:
            source = f"Gemini ({args.model or os.environ.get('GEMINI_MODEL', DEFAULT_MODEL)})"
            note = (
                f"Generated by Gemini model {args.model or os.environ.get('GEMINI_MODEL', DEFAULT_MODEL)}, "
                f"temperature={TEMPERATURE}, max_output_tokens={MAX_OUTPUT_TOKENS}."
            )
        else:
            note = "Offline template used because Gemini was unavailable or no API key was configured."
    text = res["narrative"]

    checks = check_narrative(text, findings)
    report, ok = format_report(checks, unverified_numbers(text, findings))
    output = f"SOURCE: {source}\nNOTE: {note}\n\n{text}\n\n{report}\n"
    print(output)
    Path(args.out).write_text(output, encoding="utf-8")
    print(f"[narrator] wrote {Path(args.out).name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
