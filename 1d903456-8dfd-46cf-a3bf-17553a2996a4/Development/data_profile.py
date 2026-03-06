
import pandas as pd
import numpy as np

# Load the parquet file
df = pd.read_parquet("user_retention.parquet")

# ─────────────────────────────────────────────
# 1. SHAPE & BASIC INFO
# ─────────────────────────────────────────────
print("=" * 70)
print("USER RETENTION DATASET — FULL DATA PROFILE")
print("=" * 70)
print(f"\n📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# 2. DTYPES
# ─────────────────────────────────────────────
print("─" * 70)
print("COLUMN DTYPES")
print("─" * 70)
print(df.dtypes.to_string())

# ─────────────────────────────────────────────
# 3. MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("MISSING VALUES")
print("─" * 70)
null_counts  = df.isnull().sum()
null_pct     = (null_counts / len(df) * 100).round(2)
null_summary = pd.DataFrame({
    "missing_count": null_counts,
    "missing_%":     null_pct
}).sort_values("missing_count", ascending=False)
null_with_any = null_summary[null_summary["missing_count"] > 0]
if null_with_any.empty:
    print("✅  No missing values found in any column.")
else:
    print(null_with_any.to_string())
    print(f"\nTotal columns with nulls: {len(null_with_any)} / {df.shape[1]}")

# ─────────────────────────────────────────────
# 4. UNIQUE VALUES PER COLUMN
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("UNIQUE VALUE COUNTS")
print("─" * 70)
unique_counts = df.nunique().sort_values(ascending=False)
print(unique_counts.to_string())

# ─────────────────────────────────────────────
# 5. NUMERIC DISTRIBUTIONS
# ─────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    print("\n" + "─" * 70)
    print(f"NUMERIC COLUMN DISTRIBUTIONS ({len(numeric_cols)} columns)")
    print("─" * 70)
    desc = df[numeric_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc["skew"] = df[numeric_cols].skew().round(3)
    print(desc.to_string())

# ─────────────────────────────────────────────
# 6. CATEGORICAL / LOW-CARDINALITY COLUMNS
# ─────────────────────────────────────────────
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
low_card_num = [c for c in numeric_cols if df[c].nunique() <= 20]
candidate_cat = list(set(cat_cols + low_card_num))

if candidate_cat:
    print("\n" + "─" * 70)
    print("CATEGORICAL / LOW-CARDINALITY COLUMN VALUE COUNTS")
    print("─" * 70)
    for col in sorted(candidate_cat):
        vc = df[col].value_counts(dropna=False).head(15)
        print(f"\n  [{col}]  (unique={df[col].nunique()})")
        for val, cnt in vc.items():
            pct = cnt / len(df) * 100
            print(f"    {str(val):<30} {cnt:>8,}  ({pct:5.1f}%)")

# ─────────────────────────────────────────────
# 7. DATETIME COLUMNS (range check)
# ─────────────────────────────────────────────
date_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
# Also sniff object cols that look like dates
for col in cat_cols:
    sample = df[col].dropna().head(100)
    converted = pd.to_datetime(sample, errors="coerce")
    if converted.notna().mean() > 0.8:
        date_cols.append(col)
date_cols = list(set(date_cols))

if date_cols:
    print("\n" + "─" * 70)
    print("DATE COLUMN RANGES")
    print("─" * 70)
    for col in date_cols:
        try:
            s = pd.to_datetime(df[col], errors="coerce")
            print(f"  {col:<40} min={s.min()}  max={s.max()}")
        except Exception:
            pass

# ─────────────────────────────────────────────
# 8. SAMPLE ROWS
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("SAMPLE ROWS (first 5)")
print("─" * 70)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)
print(df.head(5).to_string())

print("\n" + "─" * 70)
print("SAMPLE ROWS (random 5)")
print("─" * 70)
print(df.sample(5, random_state=42).to_string())

print("\n" + "=" * 70)
print("END OF PROFILE")
print("=" * 70)
