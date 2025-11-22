#!/usr/bin/env python3
"""
Evaluate the Trained Colorant Model
Check if it's learning real patterns or just overfitting/memorizing
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from google.cloud import storage

# Load the trained model
checkpoint_path = Path('./checkpoints_colorant_v2_regression/model_best.pt')
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=" * 70)
print("🔍 MODEL EVALUATION")
print("=" * 70)
print(f"\nBest validation loss from training: {checkpoint['best_val_loss']:.4f}")
print(f"Training stopped at epoch: {checkpoint['epoch']}")

# Load the data
data_path = Path('./colorant_data_expanded/combined_all_data.csv')
df = pd.read_csv(data_path)

print(f"\n📊 Dataset Statistics:")
print(f"  Total samples: {len(df)}")
print(f"  Data sources: {df['dataset_source'].value_counts().to_dict()}")

# Check data quality
print(f"\n🔍 Data Quality Check:")

# Real vs Synthetic breakdown
real_data = df[df['dataset_source'].isin(['pubchem_extended', 'literature_extended'])]
synthetic_data = df[~df['dataset_source'].isin(['pubchem_extended', 'literature_extended'])]

print(f"  Real data: {len(real_data)} samples ({len(real_data)/len(df)*100:.1f}%)")
print(f"  Synthetic data: {len(synthetic_data)} samples ({len(synthetic_data)/len(df)*100:.1f}%)")

# Check target properties
target_features = [
    'pH_stability_3', 'pH_stability_5', 'pH_stability_7',
    'k_4C', 'light_stability', 'oxygen_sensitivity',
    'MolecularWeight', 'Complexity'
]

print(f"\n📈 Target Property Statistics (Real Data Only):")
for feat in target_features:
    if feat in real_data.columns:
        values = real_data[feat].dropna()
        if len(values) > 0:
            print(f"  {feat:25s}: mean={values.mean():.3f}, std={values.std():.3f}, range=[{values.min():.3f}, {values.max():.3f}]")

print(f"\n📈 Target Property Statistics (Synthetic Data):")
for feat in target_features:
    if feat in synthetic_data.columns:
        values = synthetic_data[feat].dropna()
        if len(values) > 0:
            print(f"  {feat:25s}: mean={values.mean():.3f}, std={values.std():.3f}, range=[{values.min():.3f}, {values.max():.3f}]")

# Reality check
print("\n" + "=" * 70)
print("🎯 REALITY CHECK")
print("=" * 70)

issues = []
recommendations = []

# 1. Check real data percentage
real_pct = len(real_data) / len(df) * 100
if real_pct < 30:
    issues.append(f"⚠️  Only {real_pct:.1f}% real data - model may not generalize well")
    recommendations.append("Collect more real colorant data from PubChem or experiments")

# 2. Check if synthetic data is too similar
print("\nChecking synthetic data diversity...")
if 'MolecularWeight' in synthetic_data.columns:
    mw_std = synthetic_data['MolecularWeight'].std()
    mw_real_std = real_data['MolecularWeight'].std() if 'MolecularWeight' in real_data.columns else 0
    
    if mw_std > 0 and mw_real_std > 0 and mw_std < mw_real_std * 0.5:
        issues.append(f"⚠️  Synthetic data has low diversity (MW std: {mw_std:.1f} vs real: {mw_real_std:.1f})")
        recommendations.append("Increase perturbation in synthetic data generation")

# 3. Check target correlation (are all targets too correlated?)
available_targets = [f for f in target_features if f in df.columns]
if len(available_targets) >= 3:
    target_data = df[available_targets].dropna()
    if len(target_data) > 10:
        corr_matrix = target_data.corr()
        high_corr = (corr_matrix.abs() > 0.95).sum().sum() - len(available_targets)  # exclude diagonal
        if high_corr > len(available_targets):
            issues.append(f"⚠️  Many targets highly correlated ({high_corr} pairs > 0.95)")
            recommendations.append("Targets may be too similar - model might be learning trivial patterns")

# 4. Check validation loss magnitude
val_loss = checkpoint['best_val_loss']
if val_loss < 0.1:
    issues.append(f"⚠️  Very low validation loss ({val_loss:.4f}) might indicate overfitting")
    recommendations.append("Validate on completely held-out real colorants")
elif val_loss > 1.0:
    issues.append(f"⚠️  High validation loss ({val_loss:.4f}) - model may not be learning")
    recommendations.append("Check data quality and increase model capacity")

# Print issues
print("\n🚨 Issues Found:")
if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  ✅ No major issues detected")

print("\n💡 Recommendations:")
if recommendations:
    for rec in recommendations:
        print(f"  • {rec}")
else:
    print("  ✅ Model looks reasonable for the current dataset")

# Final verdict
print("\n" + "=" * 70)
print("📋 FINAL ASSESSMENT")
print("=" * 70)

score = 100
verdict_parts = []

if real_pct < 20:
    score -= 40
    verdict_parts.append("⚠️  Critical: Too little real data")
elif real_pct < 30:
    score -= 20
    verdict_parts.append("⚠️  Warning: Limited real data")

if val_loss < 0.1:
    score -= 20
    verdict_parts.append("⚠️  Suspiciously low loss - likely overfitting")
elif val_loss > 0.5:
    score -= 10
    verdict_parts.append("⚠️  Moderate loss - adequate but not great")

if len(issues) > 3:
    score -= 15
    verdict_parts.append("⚠️  Multiple data quality issues")

print(f"\n🎯 Model Quality Score: {score}/100")
print(f"\nVerdict:")
if score >= 80:
    print("  ✅ GOOD - Model is production-ready")
    print("  The model should generalize reasonably well")
elif score >= 60:
    print("  ⚠️  ACCEPTABLE - Model works but has limitations")
    print("  Useful for prototyping, but collect more real data for production")
elif score >= 40:
    print("  ⚠️  WEAK - Model may not generalize well")
    print("  Needs more real data and validation")
else:
    print("  ❌ POOR - Model is not reliable")
    print("  Significant improvements needed before use")

if verdict_parts:
    print(f"\nKey concerns:")
    for vp in verdict_parts:
        print(f"  {vp}")

# Next steps
print("\n" + "=" * 70)
print("🚀 NEXT STEPS")
print("=" * 70)
print("""
1. IMMEDIATE:
   • Test on 5-10 completely new colorants (not in training data)
   • Compare predictions with known experimental values
   • Calculate R² score on test set

2. SHORT-TERM:
   • Collect more real colorant data from PubChem/literature
   • Target: 100+ real samples, reduce synthetic to <50%
   • Add experimental validation data from Kraft Heinz

3. LONG-TERM:
   • Run wet lab experiments to validate predictions
   • Iterate: collect data → retrain → validate → repeat
   • Build confidence before production deployment

4. ALTERNATIVE APPROACH:
   • Consider transfer learning from larger chemistry models
   • Look into GNN (Graph Neural Networks) for molecular properties
   • Explore active learning to select most informative samples
""")

print("=" * 70)
print("📊 Check training curves: outputs_colorant_v2_regression/training_curves.png")
print("💾 Model saved at: gs://eurofins/models/colorant_model_v2_regression_best.pt")
print("=" * 70)