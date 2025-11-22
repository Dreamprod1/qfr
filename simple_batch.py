#!/usr/bin/env python3
"""
Simple Batch Colorant Comparison
Directly calls model without subprocess parsing

Usage: python simple_batch.py
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
import time
from pathlib import Path

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class ColorantRegressionModel(nn.Module):
    """Neural network for colorant property prediction"""
    
    def __init__(self, input_dim=256, hidden_dims=[128, 64, 32], output_dim=8, dropout=0.4):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ==============================================================================
# PREDICTOR
# ==============================================================================

class BatchPredictor:
    """Batch prediction without subprocess"""
    
    def __init__(self, model_path='./checkpoints_colorant_v2_regression/model_best.pt'):
        self.model_path = Path(model_path)
        self.model = None
        
        self.input_features = [
            'MolecularWeight', 'XLogP', 'TPSA', 'Complexity',
            'HBondDonorCount', 'HBondAcceptorCount',
            'RotatableBondCount', 'HeavyAtomCount'
        ]
        
        self.output_features = [
            'pH_stability_3', 'pH_stability_5', 'pH_stability_7',
            'k_4C', 'light_stability', 'oxygen_sensitivity',
            'MolecularWeight_pred', 'Complexity_pred'
        ]
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            print(f"❌ Model not found at {self.model_path}")
            sys.exit(1)
        
        self.model = ColorantRegressionModel()
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def fetch_from_pubchem(self, colorant_name):
        """Fetch properties from PubChem"""
        try:
            # Get CID
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(colorant_name)}/cids/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            cid = response.json()['IdentifierList']['CID'][0]
            time.sleep(0.5)
            
            # Get properties
            props_list = ','.join(self.input_features)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{props_list}/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()['PropertyTable']['Properties'][0]
            return data
            
        except Exception:
            return None
    
    def prepare_features(self, properties):
        """Convert properties to model input"""
        features = []
        
        for feat in self.input_features:
            val = properties.get(feat, 0)
            try:
                features.append(float(val))
            except (ValueError, TypeError):
                features.append(0.0)
        
        # Convert and normalize
        features = np.array(features).reshape(1, -1)
        mean = np.mean(features)
        std = np.std(features) + 1e-8
        features = (features - mean) / std
        
        # Pad to 256
        padded = np.zeros((1, 256))
        padded[0, :len(features[0])] = features[0]
        
        return torch.FloatTensor(padded)
    
    def predict(self, colorant_name):
        """Predict properties for a colorant"""
        # Fetch from PubChem
        properties = self.fetch_from_pubchem(colorant_name)
        
        if properties is None:
            return None
        
        # Prepare features
        features = self.prepare_features(properties)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(features).numpy()[0]
        
        # Package results
        result = {'colorant': colorant_name}
        for i, name in enumerate(self.output_features[:len(predictions)]):
            result[name] = float(predictions[i])
        
        # Calculate overall stability
        result['overall_stability'] = np.mean([
            result['pH_stability_3'],
            result['pH_stability_5'],
            result['pH_stability_7'],
            result['light_stability']
        ])
        
        return result

# ==============================================================================
# MAIN
# ==============================================================================

def main():
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
    
    # Initialize predictor
    print("📥 Loading model...")
    predictor = BatchPredictor()
    print("✅ Model loaded\n")
    
    # Predict for each colorant
    results = []
    
    for i, colorant in enumerate(COLORANTS, 1):
        print(f"[{i}/{len(COLORANTS)}] Predicting: {colorant}...", end=" ")
        
        result = predictor.predict(colorant)
        
        if result:
            results.append(result)
            print(f"✅ (score: {result['overall_stability']:.3f})")
        else:
            print(f"❌ Failed")
        
        time.sleep(1)  # Be nice to PubChem
    
    if len(results) == 0:
        print("\n❌ No predictions collected.")
        return
    
    # Create DataFrame and sort
    df = pd.DataFrame(results)
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
        ph_avg = (row['pH_stability_3'] + row['pH_stability_5'] + row['pH_stability_7']) / 3
        light = row['light_stability']
        
        print(f"{rank:<6}{colorant:<20}{overall:>10.3f}  {ph_avg:>10.3f}  {light:>10.3f}")
    
    # Top 3 recommendations
    print("\n" + "=" * 70)
    print("🏆 TOP 3 RECOMMENDED FOR TESTING")
    print("=" * 70)
    
    for idx in range(min(3, len(df_sorted))):
        row = df_sorted.iloc[idx]
        print(f"\n{idx+1}. {row['colorant'].upper()}")
        print(f"   Overall Stability: {row['overall_stability']:.3f} (normalized)")
        print(f"   pH Stability: {(row['pH_stability_3'] + row['pH_stability_5'] + row['pH_stability_7'])/3:.3f}")
        print(f"   Light Stability: {row['light_stability']:.3f}")
        print(f"   Temperature Sensitivity: {row['k_4C']:.3f}")
        
        # Recommendations
        recommendations = []
        
        if row['k_4C'] > 0.5:
            recommendations.append("⚠️ Store cold")
        
        if row['light_stability'] < -0.5:
            recommendations.append("⚠️ Use dark packaging")
        
        if row['oxygen_sensitivity'] > 0.5:
            recommendations.append("⚠️ Use antioxidants")
        
        if recommendations:
            print("   " + " | ".join(recommendations))
        else:
            print("   ✅ Good stability profile")
    
    # Save results
    output_file = 'colorant_rankings.csv'
    df_sorted.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("💾 RESULTS SAVED")
    print("=" * 70)
    print(f"File: {output_file}")
    print(f"Colorants compared: {len(df)}")
    
    # ROI calculation
    print("\n💰 ESTIMATED ROI")
    print("=" * 70)
    
    total = len(COLORANTS)
    top_n = min(5, len(df_sorted))
    
    print(f"Without model: Test all {total} → ${total * 500:,} | {total} weeks")
    print(f"With model: Test top {top_n} → ${top_n * 500:,} | {top_n} weeks")
    print(f"\n💰 SAVINGS: ${(total - top_n) * 500:,} ({(total-top_n)/total*100:.0f}%) | {total - top_n} weeks ({(total-top_n)/total*100:.0f}%)")
    
    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review rankings in colorant_rankings.csv")
    print("2. Test top 3-5 colorants in lab")
    print("3. Compare predictions vs actual results")
    print("4. Calculate R² score to validate model")
    print("=" * 70)

if __name__ == "__main__":
    main()