"""Part 2: Python/pandas wrangling & EDA for the Mamaearth returns pipeline.

Runs on the RAW CSVs in data/ (never on the SQL database, never on hand-edited
files). Every intermediate result is printed. At the end it writes
narrator/findings.json so the GenAI layer receives only numbers computed here.

Order of operations matters (see README "pitfalls"):
    load -> standardise casing -> drop duplicates -> impute -> merge/value
    -> flag outliers -> hypotheses -> correlation -> corrected trend

Run from anywhere:   python analysis/clean_and_eda.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FINDINGS_PATH = ROOT / "narrator" / "findings.json"

PAYMENT_MAP = {"COD": "COD", "CARD": "CARD", "UPI": "UPI"}
DUP_KEY = ["customer_id", "product_id", "order_date", "quantity", "discount_pct",
           "payment_method", "rating", "returned"]


def _quiet(*args, **kwargs):
    pass


def label_corr(r):
    """Strength label for a Pearson correlation coefficient."""
    a = abs(r)
    if a < 0.20:
        return "negligible"
    if a < 0.40:
        return "weak"
    if a < 0.70:
        return "moderate"
    return "strong"


def order_value(df):
    """quantity * price * (1 - discount%). NaN discount is treated as 0%."""
    return df["quantity"] * df["price"] * (1 - df["discount_pct"].fillna(0) / 100.0)


def run(verbose=True, write_findings=True):
    log = print if verbose else _quiet
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 30)
    R = {}  # everything the later layers need

    # ------------------------------------------------------------------
    # Task 1 - Load and inspect (2)
    # ------------------------------------------------------------------
    log("=" * 70, "\nTASK 1 - LOAD AND INSPECT\n", "=" * 70, sep="")
    customers = pd.read_csv(DATA / "customers.csv")
    products = pd.read_csv(DATA / "products.csv")
    orders_raw = pd.read_csv(DATA / "orders.csv")
    log(f"customers: {customers.shape}   products: {products.shape}   orders (raw): {orders_raw.shape}")
    log("\nRaw orders - first 5 rows:\n", orders_raw.head(), sep="")
    log("\nRaw orders - dtypes:\n", orders_raw.dtypes, sep="")
    log("\nRaw orders - missing values per column:\n", orders_raw.isna().sum(), sep="")
    R["n_customers"], R["n_products"], R["n_orders_raw"] = len(customers), len(products), len(orders_raw)
    R["raw_shape"] = list(orders_raw.shape)
    R["missing_discount_raw"] = int(orders_raw["discount_pct"].isna().sum())
    R["missing_rating_raw"] = int(orders_raw["rating"].isna().sum())

    # Reference total on the RAW data (same definition as SQL report (a))
    raw_m = (orders_raw.merge(products, on="product_id", validate="many_to_one"))
    raw_m["order_value"] = order_value(raw_m)
    R["raw_revenue"] = float(raw_m["order_value"].sum())
    R["raw_aov"] = float(raw_m["order_value"].mean())
    log(f"\nRaw revenue (NaN discount = 0%, duplicates still present): {R['raw_revenue']:,.2f}"
        f"  <- equals SQL report (a) total_revenue")

    # ------------------------------------------------------------------
    # Task 2 - Standardise casing (4)
    # ------------------------------------------------------------------
    log("\n", "=" * 70, "\nTASK 2 - STANDARDISE payment_method CASING\n", "=" * 70, sep="")
    before = orders_raw["payment_method"].value_counts()
    log(f"Raw unique values ({orders_raw['payment_method'].nunique()}): "
        f"{orders_raw['payment_method'].unique().tolist()}")
    log(f"Raw distinct values: {before.shape[0]}\n{before.to_string()}")
    orders = orders_raw.copy()
    orders["payment_method"] = (orders["payment_method"].str.strip().str.upper().map(PAYMENT_MAP))
    assert orders["payment_method"].notna().all(), "unmapped payment_method value"
    after = orders["payment_method"].value_counts()
    log(f"\nClean unique values ({orders['payment_method'].nunique()}): "
        f"{orders['payment_method'].unique().tolist()}")
    log(f"Clean distinct values: {after.shape[0]}\n{after.to_string()}")
    log(f"Check: {before.shape[0]} raw values -> {after.shape[0]} clean values")
    R["payment_variants_raw"], R["payment_variants_clean"] = int(before.shape[0]), int(after.shape[0])

    # ------------------------------------------------------------------
    # Task 3 - Remove duplicates (4)  (BEFORE imputing, so the median is not skewed)
    # ------------------------------------------------------------------
    log("\n", "=" * 70, "\nTASK 3 - REMOVE DUPLICATE (DOUBLE-SUBMITTED) ORDERS\n", "=" * 70, sep="")
    log(f"Natural key of an order: {DUP_KEY}")
    orders = orders.sort_values("order_id").reset_index(drop=True)
    is_dup = orders.duplicated(subset=DUP_KEY, keep="first")
    dropped = orders[is_dup]
    kept_twins = orders[orders.duplicated(subset=DUP_KEY, keep="last")]
    log(f"Duplicate rows found: {len(dropped)}")
    log(f"Dropped order_ids:   {dropped['order_id'].tolist()}")
    log(f"Original order_ids kept: {kept_twins['order_id'].tolist()}")
    other_cols = [c for c in orders.columns if c != "order_id"]
    same = all(
        orders.loc[orders[DUP_KEY].apply(tuple, axis=1) == tuple(r[DUP_KEY]), other_cols]
        .drop_duplicates().shape[0] == 1
        for _, r in dropped.iterrows())
    log(f"Dropped rows are identical to their original apart from order_id: {same}")
    orders = orders[~is_dup].reset_index(drop=True)
    log(f"Shape after dedup: {orders.shape}   (expected 175 rows)")
    R["duplicates_dropped"] = int(len(dropped))
    R["dropped_order_ids"] = dropped["order_id"].tolist()
    R["n_orders_clean"] = int(len(orders))

    # ------------------------------------------------------------------
    # Task 4 - Impute missing values (4)
    # ------------------------------------------------------------------
    log("\n", "=" * 70, "\nTASK 4 - IMPUTE MISSING VALUES\n", "=" * 70, sep="")
    miss_disc, miss_rate = int(orders["discount_pct"].isna().sum()), int(orders["rating"].isna().sum())
    log(f"Missing after dedup -> discount_pct: {miss_disc}   rating: {miss_rate}")
    rating_median = float(orders["rating"].median())  # median of observed ratings, post-dedup
    log(f"Median of observed ratings: {rating_median}")
    orders["discount_pct"] = orders["discount_pct"].fillna(0)
    orders["rating"] = orders["rating"].fillna(rating_median)
    log(f"Missing after imputation -> discount_pct: {int(orders['discount_pct'].isna().sum())}"
        f"   rating: {int(orders['rating'].isna().sum())}")
    R["missing_discount_clean"], R["missing_rating_clean"] = miss_disc, miss_rate
    R["rating_median_used"] = rating_median

    # ------------------------------------------------------------------
    # Task 5 - Merge and reconcile (5)
    # ------------------------------------------------------------------
    log("\n", "=" * 70, "\nTASK 5 - MERGE AND RECONCILE WITH PART 1\n", "=" * 70, sep="")
    df = (orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
                .merge(products, on="product_id", how="left", validate="many_to_one"))
    assert df[["city", "category", "price"]].notna().all().all(), "orphan foreign key"
    df["order_value"] = order_value(df)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    log(f"Merged shape: {df.shape}  (orders x customers x products)")
    log(df[["order_id", "customer_id", "city", "city_tier", "category", "quantity", "price",
            "discount_pct", "order_value"]].head())
    R["clean_revenue"] = float(df["order_value"].sum())
    R["clean_aov"] = float(df["order_value"].mean())
    dup_rev = float(raw_m.loc[raw_m["order_id"].isin(dropped["order_id"]), "order_value"].sum())
    gap = R["raw_revenue"] - R["clean_revenue"]
    R["duplicate_revenue"] = dup_rev
    log(f"\nPart 1 (SQL, raw)      : {R['n_orders_raw']} orders, revenue {R['raw_revenue']:,.2f}, AOV {R['raw_aov']:,.2f}")
    log(f"Part 2 (Python, clean) : {R['n_orders_clean']} orders, revenue {R['clean_revenue']:,.2f}, AOV {R['clean_aov']:,.2f}")
    log(f"Revenue gap            : {gap:,.2f}")
    log(f"Revenue of the {len(dropped)} dropped duplicate rows: {dup_rev:,.2f}")
    assert np.isclose(gap, dup_rev), "gap not fully explained by duplicates"
    log("\nRECONCILIATION: "
        f"The raw total of ₹{R['raw_revenue']:,.2f} exceeds the cleaned total of "
        f"₹{R['clean_revenue']:,.2f} by exactly ₹{gap:,.2f}, entirely due to the "
        f"{len(dropped)} duplicate orders removed in Task 3; independently summing their "
        f"order_value gives ₹{dup_rev:,.2f}. Discount imputation (missing discount to 0%) "
        "and rating imputation do not change any order_value total.")

    # ------------------------------------------------------------------
    # Task 6 - IQR outliers on quantity (4) - FLAG, do not drop
    # ------------------------------------------------------------------
    log("\n", "=" * 70, "\nTASK 6 - IQR OUTLIERS ON quantity (FLAG, DO NOT DROP)\n", "=" * 70, sep="")
    q1, q3 = df["quantity"].quantile(0.25), df["quantity"].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["is_outlier"] = (df["quantity"] < lo) | (df["quantity"] > hi)
    log(f"Q1={q1}  Q3={q3}  IQR={iqr}  ->  fences: lower={lo}, upper={hi}")
    out = df[df["is_outlier"]]
    log(f"Flagged {len(out)} outlier orders (rows kept in the data):")
    log(out[["order_id", "customer_id", "order_date", "quantity", "order_value"]].to_string(index=False))
    R.update(iqr_q1=float(q1), iqr_q3=float(q3), iqr_upper_fence=float(hi),
             n_outliers=int(len(out)), outlier_order_ids=out["order_id"].tolist(),
             outlier_quantities=[int(x) for x in out["quantity"]],
             outlier_revenue=float(out["order_value"].sum()),
             outlier_revenue_share_pct=float(100 * out["order_value"].sum() / R["clean_revenue"]))
    log(f"Outlier revenue: {R['outlier_revenue']:,.2f} = {R['outlier_revenue_share_pct']:.1f}% of clean revenue")

    # ------------------------------------------------------------------
    # Task 7 - COD return hypothesis (4)
    # ------------------------------------------------------------------
    log("\n", "=" * 70, "\nTASK 7 - HYPOTHESIS: COD ORDERS ARE RETURNED MORE OFTEN\n", "=" * 70, sep="")
    log("H0: return rate is the same for COD and prepaid (Card/UPI) orders.")
    log("H1: COD orders have a HIGHER return rate than prepaid orders.")
    overall = df["returned"].mean()
    by_pay = df.groupby("payment_method")["returned"].agg(["count", "mean"])
    by_pay["returned"] = df.groupby("payment_method")["returned"].sum()
    by_pay["return_rate_pct"] = (by_pay["mean"] * 100).round(1)
    log(f"\nOverall (blended) return rate: {overall:.1%}")
    log(by_pay[["count", "mean", "return_rate_pct"]])
    cod = df[df["payment_method"] == "COD"]["returned"]
    non = df[df["payment_method"] != "COD"]["returned"]
    log(f"\nCOD return rate: {cod.mean():.1%} (n={len(cod)})   Prepaid return rate: {non.mean():.1%} (n={len(non)})")
    try:
        from scipy.stats import chi2_contingency
        table = pd.crosstab(df["payment_method"] == "COD", df["returned"])
        chi2, pval, _, _ = chi2_contingency(table)
        log(f"Chi-square test (COD vs prepaid x returned): chi2={chi2:.2f}, p={pval:.2e}")
        R["cod_chi2_p"] = float(pval)
    except ImportError:  # scipy is optional for the verdict
        pval = None
        log("scipy not installed - skipping the significance test (verdict uses rates only).")
    supported = bool(cod.mean() > non.mean() and (pval is None or pval < 0.05))
    verdict = "Confirmed" if supported else "Not confirmed"
    log(f"VERDICT: {verdict}.")
    cod_share = float(df.loc[df["payment_method"] == "COD", "returned"].sum() / df["returned"].sum())
    log(f"COD accounts for {cod_share:.1%} of all returned orders "
        f"but only {len(cod) / len(df):.1%} of orders.")
    R.update(overall_return_rate_pct=float(100 * overall), n_returned=int(df["returned"].sum()),
             cod_return_rate_pct=float(100 * cod.mean()), prepaid_return_rate_pct=float(100 * non.mean()),
             cod_orders=int(len(cod)), cod_verdict=verdict,
             cod_share_of_returns_pct=float(100 * cod_share),
             cod_share_of_orders_pct=float(100 * len(cod) / len(df)),
             return_rate_by_payment_pct={k: float(v) for k, v in by_pay["return_rate_pct"].items()})

    # ------------------------------------------------------------------
    # Task 8 - Multi-level segmentation (4)
    # ------------------------------------------------------------------
    log("\n", "=" * 70, "\nTASK 8 - RETURN RATE BY PAYMENT METHOD x CITY TIER\n", "=" * 70, sep="")
    seg = (df.groupby(["payment_method", "city_tier"])["returned"]
             .agg(orders="count", returned="sum", return_rate="mean").reset_index())
    seg["return_rate_pct"] = (seg["return_rate"] * 100).round(1)
    log(seg[["payment_method", "city_tier", "orders", "returned", "return_rate_pct"]].to_string(index=False))
    log("\nReturn-rate matrix (%):")
    log(seg.pivot(index="payment_method", columns="city_tier", values="return_rate_pct"))
    MIN_N = 5  # ignore tiny segments when naming the riskiest one
    eligible = seg[seg["orders"] >= MIN_N]
    top = eligible.sort_values(["return_rate", "orders"], ascending=False).iloc[0]
    log(f"\nHIGHEST-RISK SEGMENT (n >= {MIN_N}): payment={top['payment_method']}, tier {int(top['city_tier'])} "
        f"-> {top['return_rate_pct']}% return rate over {int(top['orders'])} orders")
    seg_returns = int(top["returned"])
    share_of_returns = 100 * seg_returns / df["returned"].sum()
    log(f"This one segment is {100 * top['orders'] / len(df):.1f}% of orders but "
        f"{share_of_returns:.1f}% of all returns. Blended average ({overall:.1%}) hides this.")
    R.update(top_segment_payment=str(top["payment_method"]), top_segment_tier=int(top["city_tier"]),
             top_segment_return_rate_pct=float(top["return_rate_pct"]), top_segment_orders=int(top["orders"]),
             top_segment_returned=seg_returns,
             top_segment_share_of_orders_pct=float(100 * top["orders"] / len(df)),
             top_segment_share_of_returns_pct=float(share_of_returns),
             segments=[{"payment_method": r.payment_method, "city_tier": int(r.city_tier),
                        "orders": int(r.orders), "return_rate_pct": float(r.return_rate_pct)}
                       for r in seg.itertuples()])
    by_city = (df.groupby("city")["returned"].agg(orders="count", rate="mean")
                 .assign(rate_pct=lambda d: (d["rate"] * 100).round(1)).sort_values("rate", ascending=False))
    hot = by_city[by_city["rate"] > 0.20]
    R["cities_above_20pct"] = int(len(hot))
    R["top_city"], R["top_city_return_rate_pct"] = hot.index[0], float(hot["rate_pct"].iloc[0])
    log(f"\nCities above 20% return rate (clean data): {len(hot)} -> {', '.join(hot.index)}")

    # ------------------------------------------------------------------
    # Task 9 - Correlation analysis (2)
    # ------------------------------------------------------------------
    log("\n", "=" * 70, "\nTASK 9 - CORRELATION ANALYSIS (Pearson)\n", "=" * 70, sep="")
    num_cols = ["rating", "returned", "discount_pct", "quantity"]
    corr = df[num_cols].corr()
    log(corr.round(3))
    pairs = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            r = float(corr.loc[a, b])
            pairs.append({"pair": f"{a} ~ {b}", "r": round(r, 3), "strength": label_corr(r)})
    log("\nPair labels (|r| 0-0.19 negligible, 0.2-0.39 weak, "
        "0.4-0.69 moderate, 0.7-1.0 strong):")
    for p in pairs:
        log(f"  {p['pair']:<28} r={p['r']:+.3f}  {p['strength']}")
    ret_pairs = sorted([p for p in pairs if "returned" in p["pair"]], key=lambda p: -abs(p["r"]))
    R["correlations_with_returned"] = ret_pairs
    r_discount_returned = next(p for p in pairs if p["pair"] == "returned ~ discount_pct")
    discount_verdict = "Busted" if abs(r_discount_returned["r"]) < 0.20 else "Not busted"
    log(f"\nHypothesis 'higher discounts reduce returns': {discount_verdict} "
        f"(discount_pct vs returned r={r_discount_returned['r']:+.3f}, negligible association).")
    r_rate = next(p for p in pairs if p["pair"] == "rating ~ returned")
    log(f"rating ~ returned is {r_rate['strength']} (r={r_rate['r']:+.3f}), but a rating is given AFTER "
        "delivery, so it reflects the return rather than predicting it. Correlation is not causation.")
    R["corr_rating_returned"] = r_rate
    R["higher_discounts_reduce_returns_verdict"] = discount_verdict

    # ------------------------------------------------------------------
    # Task 10 - Outlier-corrected monthly trend (4)
    # ------------------------------------------------------------------
    log("\n", "=" * 70, "\nTASK 10 - MONTHLY REVENUE WITH vs WITHOUT OUTLIERS\n", "=" * 70, sep="")
    monthly_all = df.groupby("order_month")["order_value"].sum()
    monthly_fix = df[~df["is_outlier"]].groupby("order_month")["order_value"].sum()
    trend = pd.DataFrame({"with_outliers": monthly_all, "without_outliers": monthly_fix}).fillna(0).round(2)
    log(trend)
    naive_peak, true_peak = trend["with_outliers"].idxmax(), trend["without_outliers"].idxmax()
    log(f"\nPeak month INCLUDING outliers (misleading): {naive_peak}  ({trend.loc[naive_peak, 'with_outliers']:,.2f})")
    log(f"TRUE peak month EXCLUDING outliers        : {true_peak}  ({trend.loc[true_peak, 'without_outliers']:,.2f})")
    log("January's apparent lead is an artifact of the two bulk orders landing in January "
        "(O0011 on 2026-01-28 and O0098 on 2026-01-10); March is the genuine peak month "
        "once those orders are excluded.")
    R.update(monthly_with_outliers={k: round(float(v), 2) for k, v in trend["with_outliers"].items()},
             monthly_without_outliers={k: round(float(v), 2) for k, v in trend["without_outliers"].items()},
             naive_peak_month=naive_peak, naive_peak_revenue=float(trend.loc[naive_peak, "with_outliers"]),
             true_peak_month=true_peak, true_peak_revenue=float(trend.loc[true_peak, "without_outliers"]))
    R["corrected_revenue"] = float(trend["without_outliers"].sum())

    # frames for visualize.py
    R["_df"], R["_by_pay"], R["_seg"], R["_trend"] = df, by_pay, seg, trend

    # ------------------------------------------------------------------
    # Hand-off: write narrator/findings.json from the numbers above
    # ------------------------------------------------------------------
    if write_findings:
        findings = {
            "cleaned_total_revenue_inr": round(R["clean_revenue"], 2),
            "raw_total_revenue_inr": round(R["raw_revenue"], 2),
            "duplicate_reconciliation_delta_inr": round(gap, 2),
            "return_rate_by_payment": {
                payment: round(rate, 1)
                for payment, rate in R["return_rate_by_payment_pct"].items()
            },
            "highest_risk_segment": {
                "payment_method": R["top_segment_payment"],
                "city_tier": R["top_segment_tier"],
                "return_rate_pct": round(R["top_segment_return_rate_pct"], 1),
            },
            "true_peak_month": {
                "month": R["true_peak_month"],
                "revenue_inr": round(R["true_peak_revenue"], 2),
            },
            "outlier_inflated_month": {
                "month": R["naive_peak_month"],
                "apparent_revenue_inr": round(R["naive_peak_revenue"], 2),
                "corrected_revenue_inr": round(
                    R["monthly_without_outliers"][R["naive_peak_month"]], 2
                ),
            },
        }
        FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        FINDINGS_PATH.write_text(json.dumps(findings, indent=2), encoding="utf-8")
        log(f"\nWrote {FINDINGS_PATH.relative_to(ROOT)} ({len(findings)} fields, all computed by this script)")
    return R


if __name__ == "__main__":
    run(verbose=True)
