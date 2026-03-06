"""
ChurnShield Data Loader
Loads all 6 data files from disk and exposes them as public canvas variables
for the Streamlit deployment to reference via: from zerve import variable
"""

import pandas as pd
import json
import joblib
from pathlib import Path
import os

BASE_DIR = Path(os.getcwd())

print("=" * 70)
print("CHURNSHIELD DATA LOADER")
print("=" * 70)
print(f"\n📂 Working directory: {BASE_DIR}\n")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL 6 DATA FILES
# ─────────────────────────────────────────────────────────────────────────────

# 1. day1_features
day1_features = pd.read_parquet(BASE_DIR / "data/day1_features.parquet")
print(f"✅ day1_features")
print(f"   Shape: {day1_features.shape[0]:,} rows × {day1_features.shape[1]} columns")
print(f"   Columns: {list(day1_features.columns[:5])}...")

# 2. temporal_targets
temporal_targets = pd.read_parquet(BASE_DIR / "data/temporal_targets.parquet")
print(f"\n✅ temporal_targets")
print(f"   Shape: {temporal_targets.shape[0]:,} rows × {temporal_targets.shape[1]} columns")
print(f"   Columns: {list(temporal_targets.columns)}")

# 3. shap_importance
shap_importance = pd.read_csv(BASE_DIR / "models/shap_feature_importance.csv")
print(f"\n✅ shap_importance")
print(f"   Shape: {shap_importance.shape[0]} features × {shap_importance.shape[1]} columns")
print(f"   Columns: {list(shap_importance.columns)}")

# 4. lift_analysis
lift_analysis = pd.read_csv(BASE_DIR / "models/lift_analysis.csv")
print(f"\n✅ lift_analysis")
print(f"   Shape: {lift_analysis.shape[0]} behaviors × {lift_analysis.shape[1]} columns")
print(f"   Columns: {list(lift_analysis.columns)}")

# 5. temporal_results (JSON string)
with open(BASE_DIR / "models/temporal_results.json", "r") as f:
    temporal_results_dict = json.load(f)
temporal_results = json.dumps(temporal_results_dict)
print(f"\n✅ temporal_results")
print(f"   Type: JSON string ({len(temporal_results)} chars)")
print(f"   Keys: {list(temporal_results_dict.keys())}")

# 6. model_bundle
model_bundle = joblib.load(BASE_DIR / "models/temporal_best_model.joblib")
print(f"\n✅ model_bundle")
print(f"   Type: {type(model_bundle).__name__}")
if isinstance(model_bundle, dict):
    print(f"   Keys: {list(model_bundle.keys())}")
    print(f"   Model: {model_bundle.get('model_name', 'Unknown')}")
    print(f"   Features: {len(model_bundle.get('feature_cols', []))}")

print("\n" + "=" * 70)
print("✅ ALL 6 VARIABLES LOADED SUCCESSFULLY")
print("=" * 70)
print("\nAvailable as canvas variables for Streamlit deployment:")
print("  • day1_features (DataFrame)")
print("  • temporal_targets (DataFrame)")
print("  • shap_importance (DataFrame)")
print("  • lift_analysis (DataFrame)")
print("  • temporal_results (JSON string)")
print("  • model_bundle (dict)")