
# Mamaearth returns & growth intelligence pipeline

**Capstone - Data Analytics with AI & Gen AI, E&ICT Academy IIT Roorkee** | Author: Nipun Malik

> **The question:** Where are returns really coming from, and what is the true revenue
> picture once the data is cleaned?

This repository answers it with three connected layers. Each layer only reports numbers it
computed itself or received from the layer before it.

```
data/*.csv ──► SQL layer ──► Python layer ──► GenAI layer
(raw, never    schema, seed   clean, reconcile,   findings.json ─► Gemini (or offline
 hand-edited)  9 reports      analyse, chart      template) ─► accuracy check
```

| Layer | Folder | What it produces | Feeds the next layer by |
|---|---|---|---|
| 1. SQL | `sql/` | Relational store (SQLite) and 9 business reports on the **raw** data | Establishing the raw totals that Python must reconcile against |
| 2. Python | `analysis/` | Cleaned data, hypothesis tests, 2 charts | Writing `narrator/findings.json` from its own computed results |
| 3. GenAI | `narrator/` | SCR business narrative + automatic number check | - |

## Repository structure

```
.
├── README.md
├── requirements.txt
├── sql/
│   ├── schema.sql            Task 1  tables, types, NOT NULL, DEFAULT, PK, FK
│   ├── seed_data.sql         Task 2  INSERTs generated from the CSVs (blank -> NULL)
│   ├── reports.sql           Task 3  reports (a)-(i), each with its exact output as a comment
│   ├── generate_seed.py      rebuilds seed_data.sql from data/*.csv
│   └── run_sql.py            builds the DB and runs the reports (no sqlite3 CLI needed)
├── data/                     customers.csv  products.csv  orders.csv  (as given)
├── analysis/
│   ├── clean_and_eda.py      Part 2 tasks 1-10, prints every result, writes findings.json
│   └── visualize.py          Part 2 task 11, writes the two PNGs
├── visualizations/
│   ├── return_rate_by_payment.png
│   └── monthly_revenue_trend.png
└── narrator/
    ├── findings.json         written BY CODE (clean_and_eda.py), never hand-typed
    ├── generate_narrative.py Part 3 tasks 2-5
    └── sample_output.txt     a narrative plus its accuracy check
```

## Run order (reproduces every number)

Requires Python 3.9+ and SQLite (the Python SQL runner uses Python's built-in `sqlite3`
module). Run every command from the repository root.

```bash
# 0. Install the Python dependencies
pip install -r requirements.txt

# 1. SQL layer: regenerate seed_data.sql from data/*.csv, create the database,
#    verify row counts (45 / 16 / 180), and run reports (a)-(i).
python sql/run_sql.py
# Optional: rewrite the expected report-output comments in reports.sql.
python sql/run_sql.py --refresh-reports

# 2. Part 2: clean and analyze the source data. Task 5 reconciles raw and cleaned
#    revenue; after the Part 2 analysis completes, the script writes the verified
#    analysis figures, including the Task 5 reconciliation, to narrator/findings.json.
python analysis/clean_and_eda.py

# 3. Generate both charts (the script quietly reruns the analysis so they match its results).
python analysis/visualize.py

# 4a. Run the narrator fully offline. No API key or network access is needed.
python narrator/generate_narrative.py --offline

# 4b. Optional: put your Google AI Studio key in the root .env file as
#     GEMINI_API_KEY=your-key, then run the same script. The .env file is ignored by Git.
python narrator/generate_narrative.py
```

The narrator loads `GEMINI_API_KEY` from the repository-root `.env` file (or from an already
set environment variable, which takes precedence). Create `.env` if needed and add
`GEMINI_API_KEY=your-key`; never commit or share that file. If the key is blank or absent,
running `python narrator/generate_narrative.py` selects the offline template automatically;
`--offline` forces that path even if a key is present. Gemini errors, timeouts, truncated
replies, or responses missing required figures likewise fall back to the offline narrative.
Both paths write `narrator/sample_output.txt` and check the five required figures. The script
exits with code 0 only when all five are present. Optional: `GEMINI_MODEL` overrides the
default model (`gemini-2.5-flash`).

To run the SQL files directly instead of using `sql/run_sql.py`, use these commands from the
repository root; the seed file is first regenerated from the CSVs:

```bash
python sql/generate_seed.py
sqlite3 sql/mamaearth.db < sql/schema.sql
sqlite3 sql/mamaearth.db < sql/seed_data.sql
sqlite3 sql/mamaearth.db < sql/reports.sql
```

Everything is deterministic: re-running any step regenerates identical numbers and PNGs.

## How the layers connect

1. **SQL → Python (reconciliation).** Report (a) is computed on the raw data, so it still
   contains the double-submitted orders. `clean_and_eda.py` Task 5 recomputes that raw total
   in pandas, then shows that the cleaned total is lower by *exactly* the value of the
   dropped duplicates (asserted in code). Casing fixes and `discount = 0` imputation cannot
   change revenue: SQL already treats a NULL discount as 0% via `COALESCE`.
2. **Python → GenAI.** `clean_and_eda.py` ends by writing the verified revenue,
   payment-return, highest-risk-segment, and monthly-peak figures to `narrator/findings.json`.
   The prompt, the offline template and the accuracy checker all read that file; no figure is
   typed anywhere in `generate_narrative.py`.
3. **GenAI → verified output.** After the narrative is produced (Gemini or template) a
   checker confirms five key figures from `findings.json` appear in it, and warns about any
   number that cannot be traced back to the findings.

## Key decisions

| Topic | Decision and why |
|---|---|
| Revenue definition | `quantity × price × (1 − discount_pct/100.0)`; NULL discount = 0%. SQL divides by **100.0** because `discount_pct` is an INTEGER column and `/100` would silently return 0. |
| Blank cells in SQL | Loaded as `NULL` (not `''`); `discount_pct` and `rating` are nullable. `seed_data.sql` uses plain `INSERT`s, avoiding the `.import` header-row and empty-string traps. A verification query proves no empty strings exist. |
| Casing | `strip().upper()` then map to `COD / Card / UPI`: 7 raw spellings become 3. |
| Duplicates | A double-submitted order is the same `customer_id + product_id + order_date + quantity` under a new `order_id`. The first occurrence is kept; dropped IDs are printed. Done **before** imputation so the rating median is not skewed. |
| Imputation | Missing `discount_pct` → 0; missing `rating` → median of observed ratings (computed after de-duplication). |
| Outliers | 1.5×IQR rule on `quantity`. Rows are **flagged, never dropped**; the trend is shown with and without them. Outliers are found before the monthly trend is built. |
| Highest-risk segment | Highest return rate among payment × tier segments with at least 5 orders, so a tiny segment cannot win by chance. Order counts (`n`) are always shown. |
| Hypothesis test | COD vs prepaid return rates plus a chi-square test; verdict is labelled SUPPORTED / NOT SUPPORTED. |
| Correlation | Pearson matrix with each pair labelled negligible (<0.10), weak (<0.30), moderate (<0.50) or strong. `rating ~ returned` is noted as a *consequence* of a return, not a predictor. |
| Gemini call | System instruction + prompt built from `findings.json`; `temperature=0.0`; `max_output_tokens=2048`; `HttpOptions(timeout=30000)` in **milliseconds**; API key is read from an environment variable. |
| Offline fallback | Template-based SCR narrative using the same findings; needs no key or network and returns the same success result shape as the online path. |

## Where each marked task lives

| Part | Task | Location |
|---|---|---|
| 1 | Schema (6) | `sql/schema.sql` |
| 1 | Seed data (4) | `sql/seed_data.sql`, `sql/generate_seed.py` |
| 1 | Reports a-i (20) | `sql/reports.sql` (outputs pasted above each query) |
| 2 | Tasks 1-6 cleaning (23) | `analysis/clean_and_eda.py` (Task headers are printed) |
| 2 | Tasks 7-10 analysis (14) | `analysis/clean_and_eda.py` |
| 2 | Task 11 charts (3) | `analysis/visualize.py` → `visualizations/` |
| 3 | findings.json (4) | written at the end of `analysis/clean_and_eda.py` |
| 3 | SCR function, parameters, errors (12) | `narrator/generate_narrative.py` → `build_prompt`, `generate_scr_narrative` |
| 3 | Offline fallback (5) | `generate_scr_narrative_offline` |
| 3 | Accuracy check (5) | `check_narrative`, `unverified_numbers` |
| 3 | README (4) | this file |

## Findings

Exact figures are printed by `clean_and_eda.py` and recorded in `narrator/findings.json` and
`narrator/sample_output.txt`; they are not repeated here so this file can never drift from the
code. In summary:

- The blended return rate is a poor guide: COD returns are highest, especially in tier-2
  cities (see `visualizations/return_rate_by_payment.png`).
- The raw monthly revenue peak is created by a handful of **bulk orders**; with them
  removed, March is the true peak month (see `visualizations/monthly_revenue_trend.png`).
- True revenue is lower than the raw SQL total by exactly the value of the double-submitted orders.

## Security

No API key is stored in this repository. Keys are read from `GEMINI_API_KEY` /
`GOOGLE_API_KEY`; `.env` files are git-ignored.
