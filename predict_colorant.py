#!/usr/bin/env python3
"""
Colorant Property Predictor - Production Tool
Use this to predict stability properties for new colorants

Usage:
    python predict_colorant.py --colorant "beta-carotene"
    python predict_colorant.py --batch colorants.csv
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List
import requests
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class PredictorConfig:
    MODEL_PATH = Path('./checkpoints_colorant_v2_regression/model_best.pt')
    
    INPUT_FEATURES = [
        'MolecularWeight', 'XLogP', 'TPSA', 'Complexity',
        'HBondDonorCount', 'HBondAcceptorCount',
        'RotatableBondCount', 'HeavyAtomCount'
    ]
    
    OUTPUT_FEATURES = [
        'pH_stability_3', 'pH_stability_5', 'pH_stability_7',
        'k_4C', 'light_stability', 'oxygen_sensitivity',
        'MolecularWeight', 'Complexity'
    ]
    
    # Feature scaling parameters (from training)
    # These would be saved during training - using placeholders
    FEATURE_MEAN = None  # Load from processor.pkl if available
    FEATURE_STD = None

# ==============================================================================
# PUBCHEM FETCHER
# ==============================================================================

class PubChemFetcher:
    """Fetch chemical properties from PubChem"""
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    @staticmethod
    def fetch_properties(colorant_name: str) -> Dict:
        """Fetch properties for a colorant by name"""
        
        print(f"\n🔍 Searching PubChem for: {colorant_name}")
        
        # 1. Get CID
        try:
            url = f"{PubChemFetcher.BASE_URL}/compound/name/{requests.utils.quote(colorant_name)}/cids/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"  ❌ Not found in PubChem")
                return None
            
            cid = response.json()['IdentifierList']['CID'][0]
            print(f"  ✅ Found CID: {cid}")
            
            time.sleep(0.5)  # Be nice to PubChem
            
            # 2. Get properties
            props = [
                'MolecularFormula', 'MolecularWeight', 'CanonicalSMILES',
                'IUPACName', 'XLogP', 'TPSA', 'Complexity',
                'HBondDonorCount', 'HBondAcceptorCount',
                'RotatableBondCount', 'HeavyAtomCount'
            ]
            
            props_str = ','.join(props)
            url = f"{PubChemFetcher.BASE_URL}/compound/cid/{cid}/property/{props_str}/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"  ❌ Failed to get properties")
                return None
            
            data = response.json()['PropertyTable']['Properties'][0]
            data['colorant_name'] = colorant_name
            data['pubchem_cid'] = cid
            data['pubchem_url'] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
            
            print(f"  ✅ Retrieved properties")
            return data
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None

# ==============================================================================
# PREDICTOR
# ==============================================================================

class ColorantPredictor:
    """Predict colorant properties using trained model"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.config = PredictorConfig()
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        print(f"\n📥 Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Rebuild model architecture
        from train_colorant_model import ColorantRegressionModel, ColorantModelConfig
        
        config = ColorantModelConfig()
        self.model = ColorantRegressionModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"  ✅ Model loaded (trained for {checkpoint['epoch']} epochs)")
        print(f"  ✅ Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    def prepare_features(self, properties: Dict) -> np.ndarray:
        """Prepare features for model input"""
        
        # Extract relevant features
        features = []
        for feat in self.config.INPUT_FEATURES:
            if feat in properties:
                # Convert to float, handle None/missing values
                try:
                    val = float(properties[feat])
                except (ValueError, TypeError):
                    val = 0.0
                features.append(val)
            else:
                features.append(0.0)  # Default if missing
        
        # Convert to numpy
        features = np.array(features).reshape(1, -1)
        
        # Normalize (using simple standardization)
        # In production, load actual scaler from training
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            features = (features - mean) / std
        
        # Pad to 256 dimensions
        padded = np.zeros((1, 256))
        padded[0, :len(features[0])] = features[0]
        
        return padded
    
    def predict(self, properties: Dict) -> Dict:
        """Make prediction for a colorant"""
        
        # Prepare input
        features = self.prepare_features(properties)
        features_tensor = torch.FloatTensor(features)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(features_tensor)
            predictions = predictions.numpy()[0]
        
        # Format output
        results = {}
        for i, feat in enumerate(self.config.OUTPUT_FEATURES[:len(predictions)]):
            results[feat] = float(predictions[i])
        
        return results
    
    def interpret_predictions(self, predictions: Dict) -> Dict:
        """Interpret predictions in human-readable format"""
        
        interpretation = {
            'overall_stability': 'Unknown',
            'best_ph_range': 'Unknown',
            'temperature_sensitivity': 'Unknown',
            'light_sensitivity': 'Unknown',
            'oxygen_sensitivity_level': 'Unknown',
            'recommendations': []
        }
        
        # Overall stability (average of pH stabilities)
        if all(f'pH_stability_{ph}' in predictions for ph in [3, 5, 7]):
            avg_stability = np.mean([
                predictions['pH_stability_3'],
                predictions['pH_stability_5'],
                predictions['pH_stability_7']
            ])
            
            if avg_stability > 0.7:
                interpretation['overall_stability'] = 'HIGH'
            elif avg_stability > 0.4:
                interpretation['overall_stability'] = 'MEDIUM'
            else:
                interpretation['overall_stability'] = 'LOW'
        
        # Best pH range
        if all(f'pH_stability_{ph}' in predictions for ph in [3, 5, 7]):
            ph_stabilities = {
                'acidic (pH 3)': predictions['pH_stability_3'],
                'neutral (pH 5)': predictions['pH_stability_5'],
                'alkaline (pH 7)': predictions['pH_stability_7']
            }
            best_ph = max(ph_stabilities, key=ph_stabilities.get)
            interpretation['best_ph_range'] = best_ph
            
            if ph_stabilities[best_ph] < 0.5:
                interpretation['recommendations'].append(
                    "⚠️ Low stability across all pH ranges - consider buffering"
                )
        
        # Temperature sensitivity
        if 'k_4C' in predictions:
            k = predictions['k_4C']
            if k < 0.001:
                interpretation['temperature_sensitivity'] = 'LOW (stable)'
            elif k < 0.01:
                interpretation['temperature_sensitivity'] = 'MEDIUM'
            else:
                interpretation['temperature_sensitivity'] = 'HIGH (use refrigeration)'
                interpretation['recommendations'].append(
                    "⚠️ High temperature sensitivity - store cold"
                )
        
        # Light sensitivity
        if 'light_stability' in predictions:
            ls = predictions['light_stability']
            if ls > 0.7:
                interpretation['light_sensitivity'] = 'LOW (light-stable)'
            elif ls > 0.4:
                interpretation['light_sensitivity'] = 'MEDIUM'
                interpretation['recommendations'].append(
                    "💡 Moderate light sensitivity - use opaque packaging"
                )
            else:
                interpretation['light_sensitivity'] = 'HIGH (protect from light)'
                interpretation['recommendations'].append(
                    "⚠️ High light sensitivity - use dark containers"
                )
        
        # Oxygen sensitivity
        if 'oxygen_sensitivity' in predictions:
            os = predictions['oxygen_sensitivity']
            if os < 0.3:
                interpretation['oxygen_sensitivity_level'] = 'LOW'
            elif os < 0.6:
                interpretation['oxygen_sensitivity_level'] = 'MEDIUM'
                interpretation['recommendations'].append(
                    "💨 Consider nitrogen flushing for longer shelf life"
                )
            else:
                interpretation['oxygen_sensitivity_level'] = 'HIGH (use antioxidants)'
                interpretation['recommendations'].append(
                    "⚠️ High oxygen sensitivity - use antioxidants and inert atmosphere"
                )
        
        return interpretation

# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

def predict_single_colorant(colorant_name: str):
    """Predict properties for a single colorant"""
    
    print("=" * 70)
    print(f"🎨 COLORANT PROPERTY PREDICTION")
    print("=" * 70)
    
    # 1. Fetch from PubChem
    fetcher = PubChemFetcher()
    properties = fetcher.fetch_properties(colorant_name)
    
    if properties is None:
        print("\n❌ Could not fetch properties. Try a different name or provide properties manually.")
        return
    
    # 2. Display chemical properties
    print(f"\n📊 Chemical Properties:")
    print(f"  Formula: {properties.get('MolecularFormula', 'N/A')}")
    
    # Safely format numeric values
    mw = properties.get('MolecularWeight', 'N/A')
    print(f"  Molecular Weight: {float(mw):.2f}" if mw != 'N/A' else "  Molecular Weight: N/A")
    
    xlogp = properties.get('XLogP', 'N/A')
    print(f"  XLogP: {float(xlogp):.2f}" if xlogp != 'N/A' else "  XLogP: N/A")
    
    tpsa = properties.get('TPSA', 'N/A')
    print(f"  TPSA: {float(tpsa):.2f}" if tpsa != 'N/A' else "  TPSA: N/A")
    
    complexity = properties.get('Complexity', 'N/A')
    print(f"  Complexity: {float(complexity):.1f}" if complexity != 'N/A' else "  Complexity: N/A")
    
    print(f"  PubChem: {properties.get('pubchem_url', 'N/A')}")
    
    # 3. Load model and predict
    predictor = ColorantPredictor(PredictorConfig.MODEL_PATH)
    predictions = predictor.predict(properties)
    
    # 4. Display predictions
    print(f"\n🔮 Predicted Properties:")
    print(f"  pH Stability (pH 3): {predictions.get('pH_stability_3', 0):.3f}")
    print(f"  pH Stability (pH 5): {predictions.get('pH_stability_5', 0):.3f}")
    print(f"  pH Stability (pH 7): {predictions.get('pH_stability_7', 0):.3f}")
    print(f"  Degradation Rate (4°C): {predictions.get('k_4C', 0):.4f}")
    print(f"  Light Stability: {predictions.get('light_stability', 0):.3f}")
    print(f"  Oxygen Sensitivity: {predictions.get('oxygen_sensitivity', 0):.3f}")
    
    # 5. Interpret results
    interpretation = predictor.interpret_predictions(predictions)
    
    print(f"\n💡 Interpretation:")
    print(f"  Overall Stability: {interpretation['overall_stability']}")
    print(f"  Best pH Range: {interpretation['best_ph_range']}")
    print(f"  Temperature Sensitivity: {interpretation['temperature_sensitivity']}")
    print(f"  Light Sensitivity: {interpretation['light_sensitivity']}")
    print(f"  Oxygen Sensitivity: {interpretation['oxygen_sensitivity_level']}")
    
    if interpretation['recommendations']:
        print(f"\n📋 Recommendations:")
        for rec in interpretation['recommendations']:
            print(f"  {rec}")
    
    print("\n" + "=" * 70)
    print("✅ Prediction complete!")
    print("=" * 70)

def predict_batch(csv_path: str):
    """Predict properties for multiple colorants from CSV"""
    
    print("=" * 70)
    print(f"🎨 BATCH COLORANT PREDICTION")
    print("=" * 70)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\n📂 Loaded {len(df)} colorants from {csv_path}")
    
    # Initialize predictor
    predictor = ColorantPredictor(PredictorConfig.MODEL_PATH)
    fetcher = PubChemFetcher()
    
    # Process each colorant
    results = []
    for idx, row in df.iterrows():
        colorant_name = row['colorant_name']
        print(f"\n[{idx+1}/{len(df)}] Processing: {colorant_name}")
        
        # Fetch properties
        properties = fetcher.fetch_properties(colorant_name)
        if properties is None:
            continue
        
        # Predict
        predictions = predictor.predict(properties)
        
        # Combine
        result = {**properties, **predictions}
        results.append(result)
        
        time.sleep(1)  # Be nice to PubChem
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = csv_path.replace('.csv', '_predictions.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved predictions to: {output_path}")
    print("=" * 70)

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Predict colorant properties')
    parser.add_argument('--colorant', type=str, help='Single colorant name')
    parser.add_argument('--batch', type=str, help='CSV file with colorants')
    
    args = parser.parse_args()
    
    if args.colorant:
        predict_single_colorant(args.colorant)
    elif args.batch:
        predict_batch(args.batch)
    else:
        print("Usage:")
        print("  python predict_colorant.py --colorant 'beta-carotene'")
        print("  python predict_colorant.py --batch colorants.csv")

if __name__ == "__main__":
    main()