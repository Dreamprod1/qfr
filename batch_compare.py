#!/usr/bin/env python3
"""
Batch Colorant Comparison and Ranking
Compare multiple colorants and rank by stability

Usage: python batch_compare.py
"""

import subprocess
import json
import pandas as pd

# List of colorants to compare
COLORANTS = [
    # Reds
    "beta-carotene",
    "lycopene",
    "astaxanthin",
    
    # Yellows
    "curcumin",
    "lutein",
    "riboflavin",
    
    # Greens
    "chlorophyll",
    
    # Blues/Purples  
    "anthocyanin",
]

print("=" * 70)
print("🎨 BATCH COLORANT COMPARISON")
print("=" * 70)
print(f"\nComparing {len(COLORANTS)} colorants for Kraft Heinz Jello project")
print("This will take ~30 seconds (PubChem API calls)...\n")

results = []

for i, colorant in enumerate(COLORANTS, 1):
    print(f"[{i}/{len(COLORANTS)}] Predicting: {colorant}...")
    
    try:
        # Run prediction (capture output)
        result = subprocess.run(
            ['python', 'predict_colorant.py', '--colorant', colorant],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse output for predictions
        output = result.stdout
        
        if "pH Stability (pH 3):" in output:
            # Extract values (normalized)
            lines = output.split('\n')
            pred_data = {'colorant': colorant}
            
            for line in lines:
                if "pH Stability (pH 3):" in line:
                    pred_data['pH_3'] = float(line.split(':')[1].strip())
                elif "pH Stability (pH 5):" in line:
                    pred_data['pH_5'] = float(line.split(':')[1].strip())
                elif "pH Stability (pH 7):" in line:
                    pred_data['pH_7'] = float(line.split(':')[1].strip())
                elif "Degradation Rate (4°C):" in line:
                    pred_data['temp_sensitivity'] = float(line.split(':')[1].strip())
                elif "Light Stability:" in line:
                    pred_data['light_stability'] = float(line.split(':')[1].strip())
                elif "Oxygen Sensitivity:" in line:
                    pred_data['oxygen_sensitivity'] = float(line.split(':')[1].strip())
            
            # Calculate overall stability score (higher = better)
            if all(k in pred_data for k in ['pH_3', 'pH_5', 'pH_7', 'light_stability']):
                pred_data['overall_stability'] = (
                    pred_data['pH_3'] + 
                    pred_data['pH_5'] + 
                    pred_data['pH_7'] + 
                    pred_data['light_stability']
                ) / 4
            
            results.append(pred_data)
            print(f"  ✅ Complete (overall score: {pred_data.get('overall_stability', 0):.3f})")
        else:
            print(f"  ❌ Failed to parse predictions")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Create DataFrame
df = pd.DataFrame(results)

if len(df) == 0:
    print("\n❌ No predictions collected. Check if predict_colorant.py is working.")
    exit(1)

# Sort by overall stability (descending)
df_sorted = df.sort_values('overall_stability', ascending=False).reset_index(drop=True)

# Display results
print("\n" + "=" * 70)
print("📊 RANKED RESULTS (Best to Worst)")
print("=" * 70)

print(f"\n{'Rank':<6}{'Colorant':<20}{'Overall':<12}{'pH Avg':<12}{'Light':<12}")
print("-" * 70)

for idx, row in df_sorted.iterrows():
    rank = idx + 1
    colorant = row['colorant']
    overall = row['overall_stability']
    ph_avg = (row['pH_3'] + row['pH_5'] + row['pH_7']) / 3
    light = row['light_stability']
    
    print(f"{rank:<6}{colorant:<20}{overall:>10.3f}  {ph_avg:>10.3f}  {light:>10.3f}")

# Top 3 recommendations
print("\n" + "=" * 70)
print("🏆 TOP 3 RECOMMENDED FOR TESTING")
print("=" * 70)

for idx in range(min(3, len(df_sorted))):
    row = df_sorted.iloc[idx]
    print(f"\n{idx+1}. {row['colorant'].upper()}")
    print(f"   Overall Stability: {row['overall_stability']:.3f}")
    print(f"   pH Stability: {(row['pH_3'] + row['pH_5'] + row['pH_7'])/3:.3f}")
    print(f"   Light Stability: {row['light_stability']:.3f}")
    
    # Recommendations
    recommendations = []
    
    if row['temp_sensitivity'] > 0.5:
        recommendations.append("⚠️ Store cold (temperature sensitive)")
    
    if row['light_stability'] < -0.5:
        recommendations.append("⚠️ Use dark packaging (light sensitive)")
    
    if row['oxygen_sensitivity'] > 0.5:
        recommendations.append("⚠️ Use antioxidants (oxygen sensitive)")
    
    if recommendations:
        print("   " + " | ".join(recommendations))
    else:
        print("   ✅ Stable across all conditions")

# Save to CSV
output_file = 'colorant_comparison_results.csv'
df_sorted.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("💾 SAVED RESULTS")
print("=" * 70)
print(f"CSV file: {output_file}")
print(f"Total colorants compared: {len(df)}")
print(f"\nUse these rankings to prioritize lab testing!")
print("Test top 3-5 first, then expand based on results.")
print("=" * 70)

# ROI Calculation
print("\n💰 ESTIMATED ROI")
print("=" * 70)
total_candidates = len(COLORANTS)
top_n = min(5, len(df_sorted))

print(f"Without model: Test all {total_candidates} colorants")
print(f"  Cost: {total_candidates} × $500 = ${total_candidates * 500:,}")
print(f"  Time: {total_candidates} weeks")

print(f"\nWith model: Test top {top_n} colorants")
print(f"  Cost: {top_n} × $500 = ${top_n * 500:,}")
print(f"  Time: {top_n} weeks")

savings_cost = (total_candidates - top_n) * 500
savings_time = total_candidates - top_n

print(f"\n💰 SAVINGS:")
print(f"  Cost: ${savings_cost:,} ({savings_cost/(total_candidates*500)*100:.0f}%)")
print(f"  Time: {savings_time} weeks ({savings_time/total_candidates*100:.0f}%)")
print("=" * 70)