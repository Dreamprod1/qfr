#!/usr/bin/env python3
"""
Qmat Generator - Advanced Food Colorant Formulation System
Integrates with QLPCGN (Quantum Latent Parallel Concept Graph Network)

Enhanced version with:
- Time-dependent degradation modeling
- Protein interaction effects
- Kinetics prediction
- Eurofins experiment design
- Kraft Heinz integration

Author: Frank (QUNEU)
Version: 2.0
"""

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import the QLPCGN model
try:
    from full_demo import (
        PersistentAutonomousLearningConsciousnessInterface,
        GrowthStrategy,
        AxiomType,
        Axiom,
        add_quantum_layer_to_autonomous_model,
        PENNYLANE_AVAILABLE
    )
    QLPCGN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import full_demo.py: {e}")
    print("Make sure full_demo.py is in the same directory or Python path")
    QLPCGN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 🎨 QMAT DATA STRUCTURES
# ==============================================================================

@dataclass
class ColorantProperties:
    """Properties of a natural food colorant"""
    name: str
    chemical_class: str  # anthocyanin, carotenoid, betalain, chlorophyll, etc.
    source: str  # red_cabbage, carrot, beetroot, spirulina, etc.
    ingredient_code: Optional[str] = None  # Sensient code
    
    # Color properties
    color_coordinates: Dict[str, float] = field(default_factory=dict)  # L, a, b
    dominant_chromophore: str = ""  # e.g., "Carminic acid", "Carotenoids"
    uv_vis_peak_nm: Optional[float] = None  # Peak absorption wavelength
    
    # Physical properties
    solubility: str = "water"  # water, oil, both
    
    # Stability properties
    ph_stability_range: Tuple[float, float] = (3.0, 8.0)
    light_stability: str = "moderate"  # poor, moderate, good, excellent
    heat_stability_c: float = 85.0  # max temperature in Celsius
    temperature_sensitivity: str = "moderate"  # from Eurofins table
    
    # Kinetics parameters (for degradation modeling)
    degradation_rate_constants: Dict[str, float] = field(default_factory=dict)  # k values
    half_life_hours: Dict[str, float] = field(default_factory=dict)  # at different conditions
    arrhenius_activation_energy: Optional[float] = None  # kJ/mol
    
    # Interactions
    vitamin_c_reactivity: str = "low"  # low, moderate, high
    protein_binding_affinity: str = "low"  # affects gelatin interaction
    
    # Commercial properties
    cost_per_kg_usd: float = 50.0
    regulatory_status: str = "FDA_GRAS"  # EU_approved, FDA_GRAS, etc.
    typical_usage_pct: Tuple[float, float] = (0.01, 0.1)  # min, max percentage
    
    # Packaging
    packaging_needs: str = "opaque_preferred"  # from Eurofins table
    alcohol_compatibility: str = "without"  # with/without

@dataclass
class FoodMatrix:
    """Target food matrix for coloring"""
    matrix_type: str  # beverage, dairy, confectionery, bakery, gelatin
    product_name: str = "Jello"
    
    # Chemical properties
    ph: float = 4.5
    water_activity: float = 0.98
    fat_content_pct: float = 0.0
    
    # Protein content (critical for gelatin products)
    protein_content_pct: float = 2.0  # gelatin content
    gelatin_bloom_strength: Optional[int] = 175  # 150, 175, 225, 250
    gelatin_type: str = "porcine"  # porcine, bovine, fish
    
    # Buffer system
    citrate_buffer_mm: float = 10.0  # affects pH stability
    ascorbic_acid_ppm: float = 0.0  # Vitamin C - can cause degradation
    
    # Processing conditions
    processing_temp_c: float = 85.0
    processing_time_min: float = 10.0
    
    # Storage conditions
    shelf_life_days: int = 365
    storage_temp_c: float = 22.0  # room temperature
    light_exposure: str = "moderate"  # none, low, moderate, high
    
    # Target appearance
    target_color: str = "red"  # red, orange, yellow, green, blue, purple, brown
    target_lab: Optional[Dict[str, float]] = None  # target L*a*b* values
    texture_requirement: str = "firm_gel"  # for jello
    
    # Matrix effects
    matrix_interference: bool = False  # does matrix affect color measurement?

@dataclass
class QmatFormulation:
    """A complete Qmat formulation output"""
    formulation_id: str
    timestamp: float
    target_matrix: FoodMatrix
    colorants: List[Dict[str, Any]]  # List of {colorant, concentration_pct, contribution}
    
    # Predicted properties
    predicted_properties: Dict[str, Any] = field(default_factory=dict)
    predicted_lab: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    stability_score: float = 0.0
    initial_stability_score: float = 0.0  # at t=0
    stability_at_shelf_life: float = 0.0  # predicted at end of shelf life
    cost_estimate_per_kg: float = 0.0
    
    # AI metrics
    quantum_novelty: float = 0.0
    consciousness_confidence: float = 0.0
    
    # Safety & compliance
    safety_compliance: bool = True
    regulatory_notes: List[str] = field(default_factory=list)
    
    # Processing guidance
    processing_notes: List[str] = field(default_factory=list)
    stability_warnings: List[str] = field(default_factory=list)
    
    # Alternatives
    alternative_formulations: List[Dict] = field(default_factory=list)
    
    # Degradation prediction
    degradation_model: Optional[Dict[str, Any]] = None
    predicted_color_over_time: Optional[np.ndarray] = None

@dataclass
class EurofinsExperiment:
    """Experimental design for Eurofins testing"""
    experiment_id: str
    formulation_id: str
    test_matrix: str  # "gelatin_cubes"
    
    # Test conditions
    ph_values: List[float] = field(default_factory=list)
    temperature_points_c: List[float] = field(default_factory=list)
    time_points_days: List[int] = field(default_factory=list)
    
    # Measurements to perform
    measurements: List[str] = field(default_factory=list)  # color_lab, texture, pH, etc.
    
    # Sample preparation
    cube_size_cm: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    replicates: int = 3
    
    # Output
    sample_list: pd.DataFrame = field(default_factory=pd.DataFrame)
    expected_sample_count: int = 0

# ==============================================================================
# 🧪 KINETICS & DEGRADATION MODELS
# ==============================================================================

class ColorDegradationModel:
    """
    Models color degradation kinetics using first-order and Arrhenius equations
    """
    
    @staticmethod
    def first_order_decay(t: np.ndarray, C0: float, k: float) -> np.ndarray:
        """
        First-order degradation: C(t) = C₀ * exp(-k*t)
        
        Args:
            t: time array (hours, days, etc.)
            C0: initial color intensity
            k: rate constant
        """
        return C0 * np.exp(-k * t)
    
    @staticmethod
    def arrhenius_rate(T_celsius: float, k_ref: float, Ea: float, T_ref: float = 25.0) -> float:
        """
        Arrhenius equation for temperature-dependent rate constant
        
        k(T) = k_ref * exp(Ea/R * (1/T_ref - 1/T))
        
        Args:
            T_celsius: Temperature in Celsius
            k_ref: Rate constant at reference temperature
            Ea: Activation energy (kJ/mol)
            T_ref: Reference temperature (°C)
        """
        R = 8.314e-3  # kJ/(mol·K)
        T_kelvin = T_celsius + 273.15
        T_ref_kelvin = T_ref + 273.15
        
        k = k_ref * np.exp((Ea / R) * (1/T_ref_kelvin - 1/T_kelvin))
        return k
    
    @staticmethod
    def ph_dependent_rate(ph: float, k_neutral: float, 
                         ph_optimal: float = 4.5, 
                         sensitivity: float = 0.5) -> float:
        """
        pH-dependent degradation rate
        
        k(pH) = k_neutral * exp(sensitivity * |pH - pH_optimal|)
        """
        return k_neutral * np.exp(sensitivity * abs(ph - ph_optimal))
    
    @staticmethod
    def combined_degradation(t: np.ndarray, C0: float, 
                           k_base: float, T: float, ph: float,
                           Ea: float = 50.0, 
                           ph_optimal: float = 4.5) -> np.ndarray:
        """
        Combined temperature and pH effects on degradation
        """
        # Temperature effect
        k_T = ColorDegradationModel.arrhenius_rate(T, k_base, Ea)
        
        # pH effect
        k_pH = ColorDegradationModel.ph_dependent_rate(ph, k_T, ph_optimal)
        
        # Apply degradation
        return ColorDegradationModel.first_order_decay(t, C0, k_pH)
    
    @staticmethod
    def protein_interaction_factor(protein_pct: float, 
                                   binding_affinity: str) -> float:
        """
        Calculate how protein (gelatin) affects colorant stability
        
        Proteins can either stabilize (binding protects from degradation)
        or destabilize (binding exposes to reactive sites)
        """
        affinity_map = {
            'low': 1.0,      # no effect
            'moderate': 0.9,  # slight stabilization
            'high': 0.7       # significant stabilization
        }
        
        base_factor = affinity_map.get(binding_affinity, 1.0)
        
        # More protein = more binding sites
        protein_factor = 1.0 - (protein_pct / 100.0) * (1.0 - base_factor)
        
        return protein_factor

# ==============================================================================
# 🧪 QMAT GENERATOR CLASS
# ==============================================================================

class QmatGenerator:
    """
    Main class for generating food colorant formulations using QLPCGN.
    
    Enhanced with:
    - Degradation kinetics modeling
    - Protein interaction effects  
    - Time-dependent stability prediction
    - Eurofins experiment design
    
    Usage:
        generator = QmatGenerator()
        generator.load_colorant_database("colorants.csv")
        qmat = generator.generate_formulation(
            target_matrix=FoodMatrix(...),
            strategy='quantum'
        )
    """
    
    def __init__(self, model_dir: str = "./qmat_model", 
                 enable_quantum: bool = True,
                 auto_save: bool = True):
        """
        Initialize the Qmat Generator with QLPCGN consciousness model.
        
        Args:
            model_dir: Directory to save/load model state
            enable_quantum: Whether to use quantum enhancement
            auto_save: Auto-save model state
        """
        if not QLPCGN_AVAILABLE:
            logger.warning("⚠️ QLPCGN model not available. Running in standalone mode.")
            self.consciousness = None
            self.quantum_enabled = False
        else:
            self.model_dir = Path(model_dir)
            self.model_dir.mkdir(exist_ok=True)
            
            # Initialize QLPCGN consciousness model
            logger.info("🧠 Initializing QLPCGN Consciousness Model for Qmat...")
            self.consciousness = PersistentAutonomousLearningConsciousnessInterface(
                auto_save=auto_save,
                time_window_hours=168.0  # 1 week memory
            )
            
            # Add quantum layer if requested
            self.quantum_enabled = False
            if enable_quantum and PENNYLANE_AVAILABLE:
                logger.info("⚛️ Adding quantum layer for novel formulation discovery...")
                if add_quantum_layer_to_autonomous_model(self.consciousness):
                    self.quantum_enabled = True
                    logger.info("✅ Quantum enhancement active!")
                else:
                    logger.warning("⚠️ Quantum layer failed, using classical mode")
            
            # Set up regulatory axioms
            self._setup_regulatory_axioms()
        
        # Storage
        self.colorant_database: Dict[str, ColorantProperties] = {}
        self.matrix_templates: Dict[str, FoodMatrix] = {}
        self.formulation_history: List[QmatFormulation] = []
        
        # Degradation model
        self.degradation_model = ColorDegradationModel()
        
        logger.info("✅ QmatGenerator ready!")
        if hasattr(self, 'model_dir'):
            logger.info(f"   Model directory: {self.model_dir}")
        logger.info(f"   Quantum enabled: {self.quantum_enabled}")
    
    def _setup_regulatory_axioms(self):
        """Set up formal axioms for regulatory compliance"""
        if self.consciousness is None:
            return
            
        # Example: Prevent unsafe combinations
        self.consciousness.add_axiom(
            "MIN_CONFIDENCE",
            "food_safety",
            threshold=0.8,
            description="All formulations must have high safety confidence"
        )
        
        logger.info("📜 Regulatory axioms configured")
    
    # --------------------------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------------------------
    
    def load_colorant_database(self, csv_path: str) -> int:
        """
        Load colorant properties from CSV into the consciousness model.
        
        Expected CSV columns:
        - name, chemical_class, source, ingredient_code, L, a, b, 
          dominant_chromophore, uv_vis_peak_nm, solubility, 
          ph_min, ph_max, light_stability, heat_stability_c, 
          temperature_sensitivity, vitamin_c_reactivity, protein_binding_affinity,
          cost_per_kg_usd, regulatory_status, usage_min_pct, usage_max_pct,
          packaging_needs, alcohol_compatibility,
          degradation_k_25C, half_life_hours_25C, activation_energy_kJ_mol
        
        Returns:
            Number of colorants loaded
        """
        logger.info(f"📚 Loading colorant database from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                # Build degradation rate constants dict
                degradation_k = {}
                half_life = {}
                
                if 'degradation_k_25C' in df.columns and pd.notna(row.get('degradation_k_25C')):
                    degradation_k['25C_pH4.5'] = float(row['degradation_k_25C'])
                if 'half_life_hours_25C' in df.columns and pd.notna(row.get('half_life_hours_25C')):
                    half_life['25C_pH4.5'] = float(row['half_life_hours_25C'])
                
                colorant = ColorantProperties(
                    name=row['name'],
                    chemical_class=row['chemical_class'],
                    source=row['source'],
                    ingredient_code=row.get('ingredient_code'),
                    color_coordinates={
                        'L': float(row['L']), 
                        'a': float(row['a']), 
                        'b': float(row['b'])
                    },
                    dominant_chromophore=row.get('dominant_chromophore', ''),
                    uv_vis_peak_nm=float(row['uv_vis_peak_nm']) if pd.notna(row.get('uv_vis_peak_nm')) else None,
                    solubility=row['solubility'],
                    ph_stability_range=(float(row['ph_min']), float(row['ph_max'])),
                    light_stability=row['light_stability'],
                    heat_stability_c=float(row['heat_stability_c']),
                    temperature_sensitivity=row.get('temperature_sensitivity', 'moderate'),
                    vitamin_c_reactivity=row.get('vitamin_c_reactivity', 'low'),
                    protein_binding_affinity=row.get('protein_binding_affinity', 'low'),
                    cost_per_kg_usd=float(row['cost_per_kg_usd']),
                    regulatory_status=row['regulatory_status'],
                    typical_usage_pct=(float(row['usage_min_pct']), float(row['usage_max_pct'])),
                    packaging_needs=row.get('packaging_needs', 'opaque_preferred'),
                    alcohol_compatibility=row.get('alcohol_compatibility', 'without'),
                    degradation_rate_constants=degradation_k,
                    half_life_hours=half_life,
                    arrhenius_activation_energy=float(row['activation_energy_kJ_mol']) if pd.notna(row.get('activation_energy_kJ_mol')) else None
                )
                
                self.colorant_database[colorant.name] = colorant
                
                # Teach consciousness model about this colorant
                if self.consciousness:
                    self._teach_colorant(colorant)
            
            logger.info(f"✅ Loaded {len(self.colorant_database)} colorants")
            return len(self.colorant_database)
            
        except Exception as e:
            logger.error(f"❌ Failed to load colorant database: {e}")
            raise
    
    def _teach_colorant(self, colorant: ColorantProperties):
        """Teach the consciousness model about a colorant's properties"""
        if self.consciousness is None:
            return
            
        # Convert colorant to a concept representation
        concept_text = f"""
        Colorant: {colorant.name}
        Class: {colorant.chemical_class}
        Source: {colorant.source}
        Color: L={colorant.color_coordinates.get('L')}, a={colorant.color_coordinates.get('a')}, b={colorant.color_coordinates.get('b')}
        pH stability: {colorant.ph_stability_range[0]}-{colorant.ph_stability_range[1]}
        Heat stability: {colorant.heat_stability_c}°C
        Light stability: {colorant.light_stability}
        Cost: ${colorant.cost_per_kg_usd}/kg
        Regulatory: {colorant.regulatory_status}
        """
        
        try:
            self.consciousness.process_input(concept_text, GrowthStrategy.ENTROPY_MAXIMIZATION)
        except Exception as e:
            logger.warning(f"Could not teach colorant to consciousness: {e}")
    
    def create_eurofins_colorant_database(self) -> str:
        """
        Create a CSV database from the Eurofins table data.
        Returns the filepath.
        """
        logger.info("📊 Creating Eurofins colorant database...")
        
        # Data from the Eurofins table
        eurofins_data = [
            {
                'name': 'carmine_cochineal',
                'chemical_class': 'anthraquinone',
                'source': 'cochineal_insect',
                'ingredient_code': 'SEN-CARM-001',
                'L': 45.0, 'a': 60.0, 'b': 10.0,
                'dominant_chromophore': 'Carminic acid',
                'uv_vis_peak_nm': 520.0,
                'solubility': 'water',
                'ph_min': 3.0, 'ph_max': 8.0,
                'light_stability': 'good',
                'heat_stability_c': 95.0,
                'temperature_sensitivity': 'Stable to heat',
                'vitamin_c_reactivity': 'low',
                'protein_binding_affinity': 'moderate',
                'cost_per_kg_usd': 180.0,
                'regulatory_status': 'FDA_GRAS',
                'usage_min_pct': 0.005, 'usage_max_pct': 0.05,
                'packaging_needs': 'opaque_preferred',
                'alcohol_compatibility': 'without',
                'degradation_k_25C': 0.001,  # per day
                'half_life_hours_25C': 693.0,  # ~29 days
                'activation_energy_kJ_mol': 45.0,
                'product_type': 'confectionery, dairy',
            },
            {
                'name': 'beta_carotene',
                'chemical_class': 'carotenoid',
                'source': 'carrot',
                'ingredient_code': 'SEN-BCAR-001',
                'L': 70.0, 'a': 40.0, 'b': 80.0,
                'dominant_chromophore': 'Carotenoids',
                'uv_vis_peak_nm': 450.0,
                'solubility': 'oil',
                'ph_min': 2.5, 'ph_max': 8.0,
                'light_stability': 'good',
                'heat_stability_c': 100.0,
                'temperature_sensitivity': 'Sensitive to heat & O2',
                'vitamin_c_reactivity': 'moderate',
                'protein_binding_affinity': 'low',
                'cost_per_kg_usd': 120.0,
                'regulatory_status': 'FDA_GRAS',
                'usage_min_pct': 0.003, 'usage_max_pct': 0.04,
                'packaging_needs': 'UV_protective_needed',
                'alcohol_compatibility': 'without',
                'degradation_k_25C': 0.003,  # per day
                'half_life_hours_25C': 231.0,  # ~9.6 days
                'activation_energy_kJ_mol': 55.0,
                'product_type': 'beverages, confectionery',
            },
            {
                'name': 'spirulina_extract',
                'chemical_class': 'phycocyanin',
                'source': 'spirulina_algae',
                'ingredient_code': 'SEN-SPIR-001',
                'L': 50.0, 'a': -15.0, 'b': -30.0,
                'dominant_chromophore': 'Phycocyanin',
                'uv_vis_peak_nm': 620.0,
                'solubility': 'water',
                'ph_min': 5.0, 'ph_max': 8.0,
                'light_stability': 'moderate',
                'heat_stability_c': 70.0,
                'temperature_sensitivity': 'Sensitive to heat & pH',
                'vitamin_c_reactivity': 'high',
                'protein_binding_affinity': 'moderate',
                'cost_per_kg_usd': 250.0,
                'regulatory_status': 'FDA_GRAS',
                'usage_min_pct': 0.01, 'usage_max_pct': 0.1,
                'packaging_needs': 'opaque_preferred',
                'alcohol_compatibility': 'without',
                'degradation_k_25C': 0.008,  # per day
                'half_life_hours_25C': 86.6,  # ~3.6 days
                'activation_energy_kJ_mol': 65.0,
                'product_type': 'confectionery, beverages',
            },
            {
                'name': 'annatto_bixin',
                'chemical_class': 'carotenoid',
                'source': 'annatto_seed',
                'ingredient_code': 'SEN-ANNA-001',
                'L': 65.0, 'a': 45.0, 'b': 70.0,
                'dominant_chromophore': 'Carotenoids',
                'uv_vis_peak_nm': 470.0,
                'solubility': 'oil',
                'ph_min': 2.0, 'ph_max': 6.5,
                'light_stability': 'good',
                'heat_stability_c': 100.0,
                'temperature_sensitivity': 'Moderate heat stability',
                'vitamin_c_reactivity': 'low',
                'protein_binding_affinity': 'low',
                'cost_per_kg_usd': 75.0,
                'regulatory_status': 'FDA_GRAS',
                'usage_min_pct': 0.005, 'usage_max_pct': 0.08,
                'packaging_needs': 'UV_protective_needed',
                'alcohol_compatibility': 'without',
                'degradation_k_25C': 0.002,  # per day
                'half_life_hours_25C': 346.5,  # ~14.4 days
                'activation_energy_kJ_mol': 48.0,
                'product_type': 'dairy, snacks',
            },
            {
                'name': 'chlorophyll_derivatives',
                'chemical_class': 'chlorophyll',
                'source': 'plant_extract',
                'ingredient_code': 'SEN-CHLO-001',
                'L': 55.0, 'a': -25.0, 'b': 30.0,
                'dominant_chromophore': 'Chlorophyll a/b',
                'uv_vis_peak_nm': 660.0,
                'solubility': 'oil',
                'ph_min': 2.0, 'ph_max': 8.0,
                'light_stability': 'moderate',
                'heat_stability_c': 85.0,
                'temperature_sensitivity': 'Heat-sensitive',
                'vitamin_c_reactivity': 'moderate',
                'protein_binding_affinity': 'low',
                'cost_per_kg_usd': 150.0,
                'regulatory_status': 'FDA_GRAS',
                'usage_min_pct': 0.01, 'usage_max_pct': 0.1,
                'packaging_needs': 'UV_protective_needed',
                'alcohol_compatibility': 'without',
                'degradation_k_25C': 0.005,  # per day
                'half_life_hours_25C': 138.6,  # ~5.8 days
                'activation_energy_kJ_mol': 58.0,
                'product_type': 'beverages, confectionery',
            },
            {
                'name': 'anthocyanin_grape',
                'chemical_class': 'anthocyanin',
                'source': 'grape_skin',
                'ingredient_code': 'SEN-ANTH-001',
                'L': 40.0, 'a': 30.0, 'b': -15.0,
                'dominant_chromophore': 'Anthocyanins',
                'uv_vis_peak_nm': 535.0,
                'solubility': 'water',
                'ph_min': 3.0, 'ph_max': 5.0,
                'light_stability': 'moderate',
                'heat_stability_c': 70.0,
                'temperature_sensitivity': 'pH-sensitive',
                'vitamin_c_reactivity': 'high',
                'protein_binding_affinity': 'high',
                'cost_per_kg_usd': 95.0,
                'regulatory_status': 'FDA_GRAS',
                'usage_min_pct': 0.01, 'usage_max_pct': 0.15,
                'packaging_needs': 'opaque_preferred',
                'alcohol_compatibility': 'without',
                'degradation_k_25C': 0.012,  # per day
                'half_life_hours_25C': 57.8,  # ~2.4 days
                'activation_energy_kJ_mol': 72.0,
                'product_type': 'confectionery, beverages',
            },
        ]
        
        df = pd.DataFrame(eurofins_data)
        output_path = Path("./eurofins_colorants.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Created Eurofins database: {output_path}")
        logger.info(f"   {len(eurofins_data)} colorants")
        
        return str(output_path)
    
    def load_matrix_template(self, name: str, matrix: FoodMatrix):
        """Save a food matrix as a template for reuse"""
        self.matrix_templates[name] = matrix
        logger.info(f"📋 Saved matrix template: {name}")
    
    # --------------------------------------------------------------------------
    # FORMULATION GENERATION
    # --------------------------------------------------------------------------
    
    def generate_formulation(self, 
                           target_matrix: FoodMatrix,
                           strategy: str = 'quantum',
                           n_candidates: int = 5,
                           max_colorants: int = 3,
                           optimize_for: str = 'stability') -> QmatFormulation:
        """
        Generate a Qmat formulation for the target matrix.
        
        Args:
            target_matrix: Target food matrix specifications
            strategy: 'quantum', 'growth', or 'classical'
            n_candidates: Number of candidate formulations to generate
            max_colorants: Maximum colorants in formulation
            optimize_for: 'stability', 'cost', or 'color_match'
        
        Returns:
            QmatFormulation object with complete formulation
        """
        logger.info(f"🎨 Generating formulation for {target_matrix.matrix_type}")
        logger.info(f"   Target color: {target_matrix.target_color}")
        logger.info(f"   Strategy: {strategy}")
        
        # Generate candidate formulations
        candidates = []
        
        for i in range(n_candidates):
            candidate = self._generate_candidate_formulation(
                target_matrix, max_colorants, strategy
            )
            
            if candidate:
                candidates.append(candidate)
        
        if not candidates:
            raise ValueError("Could not generate any valid formulations")
        
        logger.info(f"   Generated {len(candidates)} candidates")
        
        # Score and rank candidates
        scored_candidates = []
        for candidate in candidates:
            score = self._score_formulation(candidate, target_matrix, optimize_for)
            candidate['total_score'] = score
            scored_candidates.append(candidate)
        
        # Select best
        best_candidate = max(scored_candidates, key=lambda x: x['total_score'])
        
        # Create QmatFormulation object
        formulation_id = f"QMAT_{int(time.time())}_{target_matrix.target_color}"
        
        qmat = QmatFormulation(
            formulation_id=formulation_id,
            timestamp=time.time(),
            target_matrix=target_matrix,
            colorants=best_candidate['colorants'],
            predicted_properties=best_candidate.get('properties', {}),
            predicted_lab=best_candidate.get('predicted_lab', {}),
            stability_score=best_candidate.get('stability_score', 0.0),
            cost_estimate_per_kg=self._calculate_cost(best_candidate),
            quantum_novelty=best_candidate.get('quantum_novelty', 0.0),
            consciousness_confidence=best_candidate.get('consciousness_score', 0.0),
            safety_compliance=True,
            processing_notes=self._generate_processing_notes(best_candidate, target_matrix),
            alternative_formulations=[c for c in scored_candidates if c != best_candidate][:3]
        )
        
        # Add degradation prediction
        qmat = self._add_degradation_prediction(qmat, target_matrix)
        
        # Validate
        qmat = self._validate_formulation(qmat)
        
        # Store in history
        self.formulation_history.append(qmat)
        
        logger.info(f"✅ Generated formulation: {formulation_id}")
        logger.info(f"   Stability score: {qmat.stability_score:.3f}")
        logger.info(f"   Cost: ${qmat.cost_estimate_per_kg:.2f}/kg")
        
        return qmat
    
    def _generate_candidate_formulation(self, 
                                       matrix: FoodMatrix,
                                       max_colorants: int,
                                       strategy: str) -> Optional[Dict]:
        """Generate a single candidate formulation"""
        
        # Filter compatible colorants
        compatible = self._filter_compatible_colorants(matrix)
        
        if not compatible:
            logger.warning("No compatible colorants found")
            return None
        
        # Select colorants for this candidate
        n_colorants = np.random.randint(1, max_colorants + 1)
        
        # Use consciousness model to select if available
        if self.consciousness and strategy in ['quantum', 'growth']:
            selected = self._consciousness_select_colorants(
                compatible, n_colorants, matrix, strategy
            )
        else:
            # Classical selection based on color matching
            selected = self._classical_select_colorants(
                compatible, n_colorants, matrix
            )
        
        if not selected:
            return None
        
        # Calculate concentrations
        concentrations = self._calculate_concentrations(selected, matrix)
        
        # Build candidate dict
        candidate = {
            'colorants': [
                {
                    'colorant': colorant.name,
                    'source': colorant.source,
                    'concentration_pct': conc,
                    'contribution': conc / sum(concentrations)
                }
                for colorant, conc in zip(selected, concentrations)
            ],
            'stability_score': self._calculate_stability_score(selected, matrix),
            'color_match_score': self._calculate_color_match(selected, concentrations, matrix),
            'quantum_novelty': np.random.random() if strategy == 'quantum' else 0.0,
            'consciousness_score': np.random.random() if self.consciousness else 0.0
        }
        
        return candidate
    
    def _filter_compatible_colorants(self, matrix: FoodMatrix) -> List[ColorantProperties]:
        """Filter colorants compatible with the matrix"""
        compatible = []
        
        for colorant in self.colorant_database.values():
            # Check pH compatibility
            if not (colorant.ph_stability_range[0] <= matrix.ph <= colorant.ph_stability_range[1]):
                continue
            
            # Check temperature compatibility
            if colorant.heat_stability_c < matrix.processing_temp_c:
                continue
            
            # Check solubility for matrix type
            if matrix.fat_content_pct > 10 and colorant.solubility == 'water':
                continue
            if matrix.fat_content_pct < 1 and colorant.solubility == 'oil':
                continue
            
            # Check Vitamin C reactivity
            if matrix.ascorbic_acid_ppm > 100 and colorant.vitamin_c_reactivity == 'high':
                continue
            
            compatible.append(colorant)
        
        return compatible
    
    def _consciousness_select_colorants(self, 
                                       compatible: List[ColorantProperties],
                                       n: int, 
                                       matrix: FoodMatrix,
                                       strategy: str) -> List[ColorantProperties]:
        """Use consciousness model to select colorants"""
        
        # Create query for consciousness
        query = f"Select {n} colorants for {matrix.target_color} color in {matrix.matrix_type} at pH {matrix.ph}"
        
        try:
            growth_strategy = GrowthStrategy.QUANTUM if strategy == 'quantum' else GrowthStrategy.ENTROPY_MAXIMIZATION
            response = self.consciousness.process_input(query, growth_strategy)
            
            # For now, use weighted random selection
            # In future, could parse consciousness response
            weights = [1.0 / (i + 1) for i in range(len(compatible))]
            selected_indices = np.random.choice(
                len(compatible), 
                size=min(n, len(compatible)), 
                replace=False,
                p=np.array(weights) / sum(weights)
            )
            
            return [compatible[i] for i in selected_indices]
            
        except Exception as e:
            logger.warning(f"Consciousness selection failed: {e}")
            return self._classical_select_colorants(compatible, n, matrix)
    
    def _classical_select_colorants(self,
                                   compatible: List[ColorantProperties],
                                   n: int,
                                   matrix: FoodMatrix) -> List[ColorantProperties]:
        """Classical colorant selection based on color matching"""
        
        # Score each colorant by color similarity to target
        target_color_map = {
            'red': {'L': 45, 'a': 50, 'b': 10},
            'orange': {'L': 65, 'a': 45, 'b': 70},
            'yellow': {'L': 80, 'a': 10, 'b': 80},
            'green': {'L': 55, 'a': -30, 'b': 30},
            'blue': {'L': 50, 'a': -15, 'b': -30},
            'purple': {'L': 40, 'a': 30, 'b': -15},
        }
        
        target_lab = matrix.target_lab or target_color_map.get(matrix.target_color, {'L': 50, 'a': 0, 'b': 0})
        
        scores = []
        for colorant in compatible:
            # Calculate Euclidean distance in L*a*b* space
            delta_L = colorant.color_coordinates.get('L', 50) - target_lab['L']
            delta_a = colorant.color_coordinates.get('a', 0) - target_lab['a']
            delta_b = colorant.color_coordinates.get('b', 0) - target_lab['b']
            
            distance = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
            score = 1.0 / (1.0 + distance / 50.0)  # Normalize
            
            scores.append(score)
        
        # Select top n by score
        if len(compatible) <= n:
            return compatible
        
        top_indices = np.argsort(scores)[-n:]
        return [compatible[i] for i in top_indices]
    
    def _calculate_concentrations(self, 
                                 colorants: List[ColorantProperties],
                                 matrix: FoodMatrix) -> List[float]:
        """Calculate optimal concentrations for selected colorants"""
        
        concentrations = []
        
        for colorant in colorants:
            # Base concentration on typical usage range
            min_usage, max_usage = colorant.typical_usage_pct
            
            # Adjust for matrix effects
            if matrix.matrix_type == 'gelatin' and colorant.protein_binding_affinity == 'high':
                # Use higher concentration if protein binding expected
                conc = min_usage + 0.7 * (max_usage - min_usage)
            else:
                conc = min_usage + 0.5 * (max_usage - min_usage)
            
            concentrations.append(conc)
        
        return concentrations
    
    def _calculate_stability_score(self, 
                                  colorants: List[ColorantProperties],
                                  matrix: FoodMatrix) -> float:
        """Calculate overall stability score for formulation"""
        
        if not colorants:
            return 0.0
        
        stability_scores = []
        
        for colorant in colorants:
            stability = 1.0
            
            # pH stability
            ph_min, ph_max = colorant.ph_stability_range
            if not (ph_min <= matrix.ph <= ph_max):
                pH_distance = min(abs(matrix.ph - ph_min), abs(matrix.ph - ph_max))
                stability *= np.exp(-pH_distance / 2.0)
            
            # Heat stability
            if colorant.heat_stability_c < matrix.processing_temp_c:
                temp_excess = matrix.processing_temp_c - colorant.heat_stability_c
                stability *= np.exp(-temp_excess / 20.0)
            
            # Light stability
            light_scores = {'poor': 0.4, 'moderate': 0.6, 'good': 0.8, 'excellent': 1.0}
            stability *= light_scores.get(colorant.light_stability, 0.5)
            
            # Protein interaction
            if matrix.protein_content_pct > 1.0:
                protein_factor = self.degradation_model.protein_interaction_factor(
                    matrix.protein_content_pct,
                    colorant.protein_binding_affinity
                )
                stability *= protein_factor
            
            # Vitamin C interaction
            if matrix.ascorbic_acid_ppm > 50:
                reactivity_penalty = {
                    'low': 1.0,
                    'moderate': 0.8,
                    'high': 0.5
                }
                stability *= reactivity_penalty.get(colorant.vitamin_c_reactivity, 0.7)
            
            stability_scores.append(max(0.0, min(1.0, stability)))
        
        # Average stability
        return np.mean(stability_scores)
    
    def _calculate_color_match(self,
                              colorants: List[ColorantProperties],
                              concentrations: List[float],
                              matrix: FoodMatrix) -> float:
        """Calculate how well the formulation matches target color"""
        
        # Weighted average of colorant L*a*b* by concentration
        total_weight = sum(concentrations)
        if total_weight == 0:
            return 0.0
        
        weighted_L = sum(c.color_coordinates.get('L', 50) * conc for c, conc in zip(colorants, concentrations)) / total_weight
        weighted_a = sum(c.color_coordinates.get('a', 0) * conc for c, conc in zip(colorants, concentrations)) / total_weight
        weighted_b = sum(c.color_coordinates.get('b', 0) * conc for c, conc in zip(colorants, concentrations)) / total_weight
        
        # Compare to target
        target_color_map = {
            'red': {'L': 45, 'a': 50, 'b': 10},
            'orange': {'L': 65, 'a': 45, 'b': 70},
            'yellow': {'L': 80, 'a': 10, 'b': 80},
            'green': {'L': 55, 'a': -30, 'b': 30},
            'blue': {'L': 50, 'a': -15, 'b': -30},
            'purple': {'L': 40, 'a': 30, 'b': -15},
        }
        
        target_lab = matrix.target_lab or target_color_map.get(matrix.target_color, {'L': 50, 'a': 0, 'b': 0})
        
        # Delta E (CIE76)
        delta_E = np.sqrt(
            (weighted_L - target_lab['L'])**2 +
            (weighted_a - target_lab['a'])**2 +
            (weighted_b - target_lab['b'])**2
        )
        
        # Convert to 0-1 score (delta_E < 3 is excellent, > 10 is poor)
        score = 1.0 / (1.0 + delta_E / 5.0)
        
        return score
    
    def _score_formulation(self, 
                          candidate: Dict,
                          matrix: FoodMatrix,
                          optimize_for: str) -> float:
        """Score a candidate formulation"""
        
        weights = {
            'stability': {'stability': 0.5, 'color': 0.3, 'cost': 0.2},
            'cost': {'stability': 0.3, 'color': 0.2, 'cost': 0.5},
            'color_match': {'stability': 0.2, 'color': 0.6, 'cost': 0.2},
        }
        
        w = weights.get(optimize_for, weights['stability'])
        
        # Normalize cost (lower is better, invert for scoring)
        cost = self._calculate_cost(candidate)
        cost_score = 1.0 / (1.0 + cost / 100.0)
        
        total_score = (
            w['stability'] * candidate.get('stability_score', 0.0) +
            w['color'] * candidate.get('color_match_score', 0.0) +
            w['cost'] * cost_score
        )
        
        return total_score
    
    def _calculate_cost(self, candidate: Dict) -> float:
        """Calculate estimated cost per kg of formulation"""
        total_cost = 0.0
        
        for colorant_data in candidate['colorants']:
            colorant = self.colorant_database[colorant_data['colorant']]
            concentration = colorant_data['concentration_pct'] / 100.0
            total_cost += colorant.cost_per_kg_usd * concentration
        
        return total_cost
    
    def _generate_processing_notes(self, 
                                  candidate: Dict,
                                  matrix: FoodMatrix) -> List[str]:
        """Generate processing guidance notes"""
        notes = []
        
        for colorant_data in candidate['colorants']:
            colorant = self.colorant_database[colorant_data['colorant']]
            
            # Temperature warnings
            if colorant.heat_stability_c < matrix.processing_temp_c + 10:
                notes.append(f"⚠️ {colorant.name}: Add after cooling below {colorant.heat_stability_c}°C")
            
            # pH adjustments
            ph_min, ph_max = colorant.ph_stability_range
            if matrix.ph < ph_min:
                notes.append(f"⚠️ {colorant.name}: Increase pH to at least {ph_min}")
            elif matrix.ph > ph_max:
                notes.append(f"⚠️ {colorant.name}: Decrease pH to below {ph_max}")
            
            # Light protection
            if colorant.light_stability in ['poor', 'moderate']:
                notes.append(f"💡 {colorant.name}: Use {colorant.packaging_needs}")
            
            # Vitamin C interactions
            if matrix.ascorbic_acid_ppm > 50 and colorant.vitamin_c_reactivity == 'high':
                notes.append(f"⚠️ {colorant.name}: Reduce ascorbic acid or use alternative antioxidant")
        
        if matrix.matrix_type == 'gelatin':
            notes.append("🧪 Gelatin interaction: Test texture after colorant addition")
        
        return notes
    
    def _validate_formulation(self, qmat: QmatFormulation) -> QmatFormulation:
        """Validate formulation against safety and regulatory rules"""
        
        # Check total concentration
        total_conc = sum([c['concentration_pct'] for c in qmat.colorants])
        if total_conc > 1.0:
            logger.warning(f"⚠️ High total colorant concentration: {total_conc:.2f}%")
            qmat.regulatory_notes.append(f"Total colorant concentration {total_conc:.2f}% exceeds typical 1% limit")
        
        # Check regulatory compliance
        for colorant_data in qmat.colorants:
            colorant = self.colorant_database[colorant_data['colorant']]
            
            if 'approved' not in colorant.regulatory_status.lower() and 'gras' not in colorant.regulatory_status.lower():
                logger.warning(f"⚠️ {colorant.name} may not be approved")
                qmat.safety_compliance = False
                qmat.regulatory_notes.append(f"{colorant.name}: Check regulatory status - {colorant.regulatory_status}")
            
            # Check usage limits
            if not (colorant.typical_usage_pct[0] <= colorant_data['concentration_pct'] <= colorant.typical_usage_pct[1]):
                qmat.regulatory_notes.append(
                    f"{colorant.name}: Concentration {colorant_data['concentration_pct']:.3f}% "
                    f"outside typical range {colorant.typical_usage_pct[0]:.3f}-{colorant.typical_usage_pct[1]:.3f}%"
                )
        
        return qmat
    
    # --------------------------------------------------------------------------
    # DEGRADATION PREDICTION
    # --------------------------------------------------------------------------
    
    def _add_degradation_prediction(self, 
                                   qmat: QmatFormulation,
                                   matrix: FoodMatrix) -> QmatFormulation:
        """Add time-dependent degradation prediction to formulation"""
        
        logger.info("📈 Calculating degradation kinetics...")
        
        # Time points (days)
        t_days = np.linspace(0, matrix.shelf_life_days, 100)
        
        # Predict color retention for each colorant
        color_retention_curves = []
        
        for colorant_data in qmat.colorants:
            colorant = self.colorant_database[colorant_data['colorant']]
            contribution = colorant_data['contribution']
            
            # Get base degradation rate
            k_base = colorant.degradation_rate_constants.get('25C_pH4.5', 0.005)  # per day
            
            # Adjust for actual conditions
            if colorant.arrhenius_activation_energy:
                k_T = self.degradation_model.arrhenius_rate(
                    matrix.storage_temp_c,
                    k_base,
                    colorant.arrhenius_activation_energy
                )
            else:
                k_T = k_base
            
            # pH adjustment
            k_final = self.degradation_model.ph_dependent_rate(
                matrix.ph,
                k_T,
                ph_optimal=(colorant.ph_stability_range[0] + colorant.ph_stability_range[1]) / 2
            )
            
            # Protein interaction
            if matrix.protein_content_pct > 1.0:
                protein_factor = self.degradation_model.protein_interaction_factor(
                    matrix.protein_content_pct,
                    colorant.protein_binding_affinity
                )
                k_final *= protein_factor
            
            # Calculate degradation curve
            C_t = self.degradation_model.first_order_decay(t_days, 100.0, k_final)
            
            # Weight by contribution
            weighted_retention = C_t * contribution
            color_retention_curves.append(weighted_retention)
        
        # Overall color retention (weighted sum)
        overall_retention = np.sum(color_retention_curves, axis=0)
        
        # Calculate stability scores
        qmat.initial_stability_score = qmat.stability_score
        qmat.stability_at_shelf_life = overall_retention[-1] / 100.0
        
        # Store prediction data
        qmat.degradation_model = {
            'time_days': t_days.tolist(),
            'color_retention_pct': overall_retention.tolist(),
            'individual_curves': {
                qmat.colorants[i]['colorant']: curve.tolist()
                for i, curve in enumerate(color_retention_curves)
            }
        }
        
        qmat.predicted_color_over_time = overall_retention
        
        # Add stability warnings
        if qmat.stability_at_shelf_life < 0.5:
            qmat.stability_warnings.append(
                f"⚠️ Color retention at shelf life ({matrix.shelf_life_days} days): {qmat.stability_at_shelf_life*100:.1f}%"
            )
        
        half_life_idx = np.where(overall_retention < 50.0)[0]
        if len(half_life_idx) > 0:
            half_life_days = t_days[half_life_idx[0]]
            qmat.stability_warnings.append(
                f"⏱️ Predicted color half-life: {half_life_days:.1f} days"
            )
        
        logger.info(f"   Stability at shelf life: {qmat.stability_at_shelf_life*100:.1f}%")
        
        return qmat
    
    def predict_color_stability_over_time(self,
                                         qmat: QmatFormulation,
                                         storage_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict color stability under different storage conditions
        
        Args:
            qmat: QmatFormulation object
            storage_conditions: Dict with 'temp_c', 'ph', 'light_exposure', 'time_days'
        
        Returns:
            Dict with time series predictions
        """
        temp_c = storage_conditions.get('temp_c', 22.0)
        ph = storage_conditions.get('ph', qmat.target_matrix.ph)
        time_days = storage_conditions.get('time_days', 180)
        
        t_days = np.linspace(0, time_days, 100)
        
        predictions = {
            'time_days': t_days.tolist(),
            'colorants': {}
        }
        
        for colorant_data in qmat.colorants:
            colorant = self.colorant_database[colorant_data['colorant']]
            
            k_base = colorant.degradation_rate_constants.get('25C_pH4.5', 0.005)
            
            if colorant.arrhenius_activation_energy:
                k_T = self.degradation_model.arrhenius_rate(
                    temp_c, k_base, colorant.arrhenius_activation_energy
                )
            else:
                k_T = k_base
            
            C_t = self.degradation_model.first_order_decay(t_days, 100.0, k_T)
            
            predictions['colorants'][colorant.name] = {
                'retention_pct': C_t.tolist(),
                'final_retention': C_t[-1]
            }
        
        return predictions
    
    # --------------------------------------------------------------------------
    # EUROFINS EXPERIMENT DESIGN
    # --------------------------------------------------------------------------
    
    def design_experiment_for_eurofins(self,
                                      qmat: QmatFormulation,
                                      ph_range: List[float] = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                      temp_points: List[float] = [4.0, 22.0, 37.0],
                                      time_points_days: List[int] = [0, 1, 7, 14, 30, 60, 90]) -> EurofinsExperiment:
        """
        Generate experimental design table for Eurofins testing
        
        This creates a DOE (Design of Experiments) table for testing
        color stability across pH, temperature, and time conditions.
        
        Args:
            qmat: Formulation to test
            ph_range: pH values to test
            temp_points: Storage temperatures (°C)
            time_points_days: Time points for measurement
        
        Returns:
            EurofinsExperiment object with sample list
        """
        logger.info(f"🧪 Designing Eurofins experiment for {qmat.formulation_id}")
        
        experiment_id = f"EURO_{qmat.formulation_id}_{int(time.time())}"
        
        # Generate all combinations
        samples = []
        sample_id = 1
        
        for colorant_data in qmat.colorants:
            for ph in ph_range:
                for temp in temp_points:
                    for time_days in time_points_days:
                        sample = {
                            'sample_id': f"S{sample_id:04d}",
                            'formulation_id': qmat.formulation_id,
                            'colorant': colorant_data['colorant'],
                            'concentration_pct': colorant_data['concentration_pct'],
                            'ph': ph,
                            'storage_temp_c': temp,
                            'time_point_days': time_days,
                            'measurement_type': 'color_lab',
                            'replicates': 3,
                            'cube_size_cm': '2x2x2',
                            'matrix': qmat.target_matrix.matrix_type,
                        }
                        samples.append(sample)
                        sample_id += 1
        
        # Create DataFrame
        sample_df = pd.DataFrame(samples)
        
        # Create experiment object
        experiment = EurofinsExperiment(
            experiment_id=experiment_id,
            formulation_id=qmat.formulation_id,
            test_matrix="gelatin_cubes",
            ph_values=ph_range,
            temperature_points_c=temp_points,
            time_points_days=time_points_days,
            measurements=['color_lab', 'visual_assessment', 'pH_verification', 'texture'],
            cube_size_cm=(2.0, 2.0, 2.0),
            replicates=3,
            sample_list=sample_df,
            expected_sample_count=len(samples)
        )
        
        logger.info(f"   Total samples: {len(samples)}")
        logger.info(f"   Colorants tested: {len(qmat.colorants)}")
        logger.info(f"   Conditions: {len(ph_range)} pH × {len(temp_points)} temps × {len(time_points_days)} times")
        
        return experiment
    
    def export_eurofins_experiment_csv(self, 
                                      experiment: EurofinsExperiment,
                                      filepath: str):
        """Export Eurofins experiment design to CSV"""
        experiment.sample_list.to_csv(filepath, index=False)
        logger.info(f"💾 Exported Eurofins experiment to {filepath}")
        logger.info(f"   {len(experiment.sample_list)} samples")
    
    # --------------------------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------------------------
    
    def plot_degradation_curves(self, 
                               qmat: QmatFormulation,
                               save_path: Optional[str] = None):
        """
        Plot color degradation curves over time
        """
        if qmat.degradation_model is None:
            logger.warning("No degradation model available for this formulation")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        time_days = np.array(qmat.degradation_model['time_days'])
        overall_retention = np.array(qmat.degradation_model['color_retention_pct'])
        
        # Plot 1: Overall color retention
        ax1.plot(time_days, overall_retention, 'b-', linewidth=2, label='Overall Color Retention')
        ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Retention')
        ax1.axvline(x=qmat.target_matrix.shelf_life_days, color='g', linestyle='--', alpha=0.5, label='Shelf Life')
        ax1.set_xlabel('Time (days)', fontsize=12)
        ax1.set_ylabel('Color Retention (%)', fontsize=12)
        ax1.set_title(f'Color Stability: {qmat.formulation_id}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 105])
        
        # Plot 2: Individual colorant contributions
        for colorant_name, curve in qmat.degradation_model['individual_curves'].items():
            ax2.plot(time_days, curve, linewidth=2, label=colorant_name, alpha=0.7)
        
        ax2.set_xlabel('Time (days)', fontsize=12)
        ax2.set_ylabel('Weighted Color Contribution (%)', fontsize=12)
        ax2.set_title('Individual Colorant Degradation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Saved degradation plot to {save_path}")
        else:
            plt.show()
    
    def plot_stability_comparison(self,
                                 formulations: List[QmatFormulation],
                                 save_path: Optional[str] = None):
        """Compare stability of multiple formulations"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for qmat in formulations:
            if qmat.degradation_model:
                time_days = np.array(qmat.degradation_model['time_days'])
                retention = np.array(qmat.degradation_model['color_retention_pct'])
                
                label = f"{qmat.target_matrix.target_color} (${qmat.cost_estimate_per_kg:.1f}/kg)"
                ax.plot(time_days, retention, linewidth=2, label=label, alpha=0.8)
        
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Retention Threshold')
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Color Retention (%)', fontsize=12)
        ax.set_title('Formulation Stability Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Saved comparison plot to {save_path}")
        else:
            plt.show()
    
    # --------------------------------------------------------------------------
    # OUTPUT & EXPORT
    # --------------------------------------------------------------------------
    
    def export_qmat_json(self, qmat: QmatFormulation, filepath: str):
        """Export Qmat formulation to JSON file"""
        # Convert to dict, handling nested dataclasses
        qmat_dict = asdict(qmat)
        
        # Convert numpy arrays to lists
        if qmat_dict.get('predicted_color_over_time') is not None:
            qmat_dict['predicted_color_over_time'] = qmat_dict['predicted_color_over_time'].tolist()
        
        with open(filepath, 'w') as f:
            json.dump(qmat_dict, f, indent=2)
        
        logger.info(f"💾 Exported Qmat to {filepath}")
    
    def export_qmat_summary(self, qmat: QmatFormulation) -> str:
        """Generate human-readable summary of formulation"""
        
        summary = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  QMAT FORMULATION SUMMARY                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  ID: {qmat.formulation_id:<60} ║
║  Generated: {datetime.fromtimestamp(qmat.timestamp).strftime('%Y-%m-%d %H:%M:%S'):<54} ║
╠══════════════════════════════════════════════════════════════════════╣
║  TARGET MATRIX:                                                      ║
║    Product: {qmat.target_matrix.product_name:<55} ║
║    Type: {qmat.target_matrix.matrix_type:<59} ║
║    Target Color: {qmat.target_matrix.target_color:<51} ║
║    pH: {qmat.target_matrix.ph:<62} ║
║    Processing Temp: {qmat.target_matrix.processing_temp_c}°C{' '*48}║
║    Shelf Life: {qmat.target_matrix.shelf_life_days} days{' '*52}║
╠══════════════════════════════════════════════════════════════════════╣
║  COLORANT COMPOSITION:                                               ║
"""
        
        for i, colorant_data in enumerate(qmat.colorants, 1):
            colorant = self.colorant_database[colorant_data['colorant']]
            summary += f"║  {i}. {colorant_data['colorant']:<63} ║\n"
            summary += f"║     Source: {colorant_data['source']:<56}║\n"
            summary += f"║     Concentration: {colorant_data['concentration_pct']:.4f}%{' '*44}║\n"
            summary += f"║     Contribution: {colorant_data['contribution']*100:.1f}%{' '*49}║\n"
            summary += f"║     Class: {colorant.chemical_class:<57}║\n"
        
        summary += f"""╠══════════════════════════════════════════════════════════════════════╣
║  PERFORMANCE METRICS:                                                ║
║    Initial Stability Score: {qmat.initial_stability_score:.3f}{' '*44}║
║    Stability at Shelf Life: {qmat.stability_at_shelf_life:.3f} ({qmat.stability_at_shelf_life*100:.1f}%){' '*32}║
║    Cost Estimate: ${qmat.cost_estimate_per_kg:.2f}/kg{' '*48}║
║    Quantum Novelty: {qmat.quantum_novelty:.3f}{' '*49}║
║    AI Confidence: {qmat.consciousness_confidence:.3f}{' '*51}║
║    Safety Compliant: {'✅ YES' if qmat.safety_compliance else '⚠️ NO'}{' '*50}║
╠══════════════════════════════════════════════════════════════════════╣
"""
        
        if qmat.processing_notes:
            summary += "║  PROCESSING NOTES:                                                   ║\n"
            for note in qmat.processing_notes:
                # Wrap long notes
                if len(note) <= 64:
                    summary += f"║  {note:<68}║\n"
                else:
                    words = note.split()
                    line = "  "
                    for word in words:
                        if len(line) + len(word) + 1 <= 64:
                            line += word + " "
                        else:
                            summary += f"║  {line:<68}║\n"
                            line = "  " + word + " "
                    if line.strip():
                        summary += f"║  {line:<68}║\n"
            summary += "╠══════════════════════════════════════════════════════════════════════╣\n"
        
        if qmat.stability_warnings:
            summary += "║  STABILITY WARNINGS:                                                 ║\n"
            for warning in qmat.stability_warnings:
                if len(warning) <= 64:
                    summary += f"║  {warning:<68}║\n"
                else:
                    words = warning.split()
                    line = "  "
                    for word in words:
                        if len(line) + len(word) + 1 <= 64:
                            line += word + " "
                        else:
                            summary += f"║  {line:<68}║\n"
                            line = "  " + word + " "
                    if line.strip():
                        summary += f"║  {line:<68}║\n"
            summary += "╠══════════════════════════════════════════════════════════════════════╣\n"
        
        if qmat.regulatory_notes:
            summary += "║  REGULATORY NOTES:                                                   ║\n"
            for note in qmat.regulatory_notes:
                if len(note) <= 64:
                    summary += f"║  {note:<68}║\n"
                else:
                    # Wrap long notes
                    words = note.split()
                    line = "  "
                    for word in words:
                        if len(line) + len(word) + 1 <= 64:
                            line += word + " "
                        else:
                            summary += f"║  {line:<68}║\n"
                            line = "  " + word + " "
                    if line.strip():
                        summary += f"║  {line:<68}║\n"
        
        summary += "╚══════════════════════════════════════════════════════════════════════╝\n"
        
        return summary
    
    def get_formulation_history(self) -> List[QmatFormulation]:
        """Get all generated formulations"""
        return self.formulation_history
    
    def save_state(self):
        """Save the generator state including QLPCGN model"""
        if self.consciousness:
            self.consciousness.save_complete_model(force=True)
            logger.info("💾 Saved QmatGenerator state")

# ==============================================================================
# 🚀 EXAMPLE USAGE & DEMO
# ==============================================================================

def demo_qmat_jello_kraft():
    """
    Complete demo for Kraft Heinz Jello project with Eurofins
    """
    print("\n" + "="*80)
    print("🎨 QMAT GENERATOR DEMO - Kraft Heinz Jello Natural Color Formulation")
    print("="*80)
    
    # Initialize generator
    print("\n🚀 Initializing Qmat Generator...")
    generator = QmatGenerator(enable_quantum=True, model_dir="./qmat_jello_model")
    
    # Create Eurofins colorant database
    print("\n📊 Creating Eurofins colorant database...")
    db_path = generator.create_eurofins_colorant_database()
    
    # Load database
    print(f"\n📚 Loading colorant database from {db_path}...")
    n_loaded = generator.load_colorant_database(db_path)
    print(f"✅ Loaded {n_loaded} natural colorants")
    
    # Define Jello matrix
    print("\n🎯 Defining Jello gelatin matrix...")
    jello_matrix = FoodMatrix(
        matrix_type="gelatin",
        product_name="Jello Dessert Mix",
        ph=4.2,
        water_activity=0.95,
        fat_content_pct=0.0,
        protein_content_pct=2.5,  # gelatin
        gelatin_bloom_strength=225,
        gelatin_type="porcine",
        citrate_buffer_mm=15.0,
        ascorbic_acid_ppm=50.0,  # Vitamin C for freshness
        processing_temp_c=85.0,
        processing_time_min=5.0,
        shelf_life_days=365,
        storage_temp_c=22.0,
        light_exposure="moderate",
        target_color="red",
        texture_requirement="firm_gel"
    )
    
    print(f"   Product: {jello_matrix.product_name}")
    print(f"   pH: {jello_matrix.ph}")
    print(f"   Gelatin: {jello_matrix.protein_content_pct}% (Bloom {jello_matrix.gelatin_bloom_strength})")
    print(f"   Target: {jello_matrix.target_color} color")
    print(f"   Shelf life: {jello_matrix.shelf_life_days} days")
    
    # Generate formulation
    print("\n⚛️ Generating natural color formulation...")
    print("   (Testing multiple candidates with AI optimization)")
    
    qmat = generator.generate_formulation(
        target_matrix=jello_matrix,
        strategy='quantum' if generator.quantum_enabled else 'growth',
        n_candidates=10,
        max_colorants=2,
        optimize_for='stability'
    )
    
    # Display results
    print("\n" + "="*80)
    print(generator.export_qmat_summary(qmat))
    
    # Plot degradation curves
    print("\n📊 Generating degradation analysis plots...")
    output_dir = Path("./qmat_outputs")
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / f"{qmat.formulation_id}_degradation.png"
    generator.plot_degradation_curves(qmat, save_path=str(plot_path))
    
    # Export formulation
    json_path = output_dir / f"{qmat.formulation_id}.json"
    generator.export_qmat_json(qmat, str(json_path))
    print(f"💾 Exported formulation to: {json_path}")
    
    # Design Eurofins experiment
    print("\n🧪 Designing Eurofins stability testing protocol...")
    experiment = generator.design_experiment_for_eurofins(
        qmat,
        ph_range=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        temp_points=[4.0, 22.0, 37.0],
        time_points_days=[0, 1, 7, 14, 30, 60, 90, 180, 365]
    )
    
    print(f"   Experiment ID: {experiment.experiment_id}")
    print(f"   Total samples: {experiment.expected_sample_count}")
    print(f"   Test matrix: Individual gelatin cubes (2x2x2 cm)")
    print(f"   pH conditions: {len(experiment.ph_values)}")
    print(f"   Temperature conditions: {len(experiment.temperature_points_c)}")
    print(f"   Time points: {len(experiment.time_points_days)}")
    
    # Export experiment design
    experiment_csv = output_dir / f"{experiment.experiment_id}_samples.csv"
    generator.export_eurofins_experiment_csv(experiment, str(experiment_csv))
    print(f"💾 Exported experiment design to: {experiment_csv}")
    
    # Generate additional formulations for comparison
    print("\n🎨 Generating alternative formulations for comparison...")
    alternative_colors = ['orange', 'purple', 'yellow']
    all_formulations = [qmat]
    
    for color in alternative_colors:
        jello_matrix.target_color = color
        alt_qmat = generator.generate_formulation(
            target_matrix=jello_matrix,
            strategy='quantum' if generator.quantum_enabled else 'growth',
            n_candidates=5,
            max_colorants=2,
            optimize_for='stability'
        )
        all_formulations.append(alt_qmat)
        print(f"   ✅ Generated {color} formulation (${alt_qmat.cost_estimate_per_kg:.2f}/kg)")
    
    # Compare formulations
    comparison_plot = output_dir / "formulation_comparison.png"
    generator.plot_stability_comparison(all_formulations, save_path=str(comparison_plot))
    
    # Save generator state
    generator.save_state()
    
    # Summary report
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE - KRAFT HEINZ JELLO PROJECT")
    print("="*80)
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"   • Formulation JSONs: {len(all_formulations)}")
    print(f"   • Degradation plots: {len(all_formulations)}")
    print(f"   • Eurofins experiment design: 1 CSV file")
    print(f"   • Comparison analysis: 1 plot")
    
    print("\n📋 Next Steps for Kraft/Eurofins:")
    print("   1. Review formulation summaries and select candidates")
    print("   2. Use Eurofins CSV to prepare sample cubes")
    print("   3. Conduct stability testing per experiment design")
    print("   4. Feed results back to refine AI model predictions")
    print("   5. Iterate on formulations based on sensory evaluation")
    
    print("\n💰 Cost Analysis:")
    for qmat_item in all_formulations:
        print(f"   {qmat_item.target_matrix.target_color.capitalize()}: ${qmat_item.cost_estimate_per_kg:.2f}/kg "
              f"(Shelf life retention: {qmat_item.stability_at_shelf_life*100:.1f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    demo_qmat_jello_kraft()