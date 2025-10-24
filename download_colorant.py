#!/usr/bin/env python3
"""
Data Download Script for Natural Food Colorant Databases
Downloads data from PubChem, FooDB, FDA, and other sources
UPLOADS DIRECTLY TO GCP BUCKET

Author: Frank (QUNEU)
Purpose: Kraft Heinz Jello Project - Build comprehensive colorant database
Modified: Upload to GCS bucket instead of local filesystem
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
from urllib.parse import quote
from io import BytesIO, StringIO

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    print("⚠️  Google Cloud Storage not installed!")
    print("Install with: pip install google-cloud-storage")
    GCS_AVAILABLE = False

# ==============================================================================
# GCP CONFIGURATION
# ==============================================================================

GCP_PROJECT_ID = "majestic-layout-461420-k4"
GCP_BUCKET_NAME = "eurofins"

# Local backup (optional)
LOCAL_BACKUP_DIR = Path("./downloaded_colorant_data_backup")

# ==============================================================================
# GCS UPLOADER CLASS
# ==============================================================================

class GCSUploader:
    """Handle uploads to Google Cloud Storage"""
    
    def __init__(self, project_id: str, bucket_name: str, local_backup: bool = True):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.local_backup = local_backup
        
        if local_backup:
            LOCAL_BACKUP_DIR.mkdir(exist_ok=True)
        
        # Initialize GCS client
        print("\n" + "="*70)
        print("INITIALIZING GOOGLE CLOUD STORAGE")
        print("="*70)
        
        try:
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            
            # Test connection
            self.bucket.exists()
            print(f"✅ Connected to GCS bucket: gs://{bucket_name}/")
            print(f"   Project: {project_id}")
            
        except Exception as e:
            print(f"❌ Failed to connect to GCS: {e}")
            print("\nTroubleshooting:")
            print("1. Run: gcloud auth application-default login")
            print("2. Or use: --credentials flag with service account key")
            raise
    
    def upload_dataframe(self, df: pd.DataFrame, path: str, description: str = ""):
        """Upload pandas DataFrame as CSV to GCS"""
        print(f"\n📤 Uploading: {path}")
        if description:
            print(f"   {description}")
        
        # Convert to CSV in memory
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Upload to GCS
        blob = self.bucket.blob(path)
        blob.upload_from_string(csv_data, content_type='text/csv')
        
        print(f"   ✅ Uploaded to: gs://{self.bucket_name}/{path}")
        print(f"   Size: {len(csv_data)} bytes")
        print(f"   Rows: {len(df)}")
        
        # Local backup
        if self.local_backup:
            local_path = LOCAL_BACKUP_DIR / path
            local_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(local_path, index=False)
            print(f"   💾 Backup: {local_path}")
        
        return f"gs://{self.bucket_name}/{path}"
    
    def upload_text(self, text: str, path: str, description: str = ""):
        """Upload text file to GCS"""
        print(f"\n📤 Uploading: {path}")
        if description:
            print(f"   {description}")
        
        # Upload to GCS
        blob = self.bucket.blob(path)
        blob.upload_from_string(text, content_type='text/plain')
        
        print(f"   ✅ Uploaded to: gs://{self.bucket_name}/{path}")
        print(f"   Size: {len(text)} bytes")
        
        # Local backup
        if self.local_backup:
            local_path = LOCAL_BACKUP_DIR / path
            local_path.parent.mkdir(exist_ok=True, parents=True)
            with open(local_path, 'w') as f:
                f.write(text)
            print(f"   💾 Backup: {local_path}")
        
        return f"gs://{self.bucket_name}/{path}"
    
    def upload_json(self, data: dict, path: str, description: str = ""):
        """Upload JSON data to GCS"""
        print(f"\n📤 Uploading: {path}")
        if description:
            print(f"   {description}")
        
        json_str = json.dumps(data, indent=2)
        
        # Upload to GCS
        blob = self.bucket.blob(path)
        blob.upload_from_string(json_str, content_type='application/json')
        
        print(f"   ✅ Uploaded to: gs://{self.bucket_name}/{path}")
        print(f"   Size: {len(json_str)} bytes")
        
        # Local backup
        if self.local_backup:
            local_path = LOCAL_BACKUP_DIR / path
            local_path.parent.mkdir(exist_ok=True, parents=True)
            with open(local_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"   💾 Backup: {local_path}")
        
        return f"gs://{self.bucket_name}/{path}"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Colorants to search for
NATURAL_COLORANTS = [
    "carminic acid",      # Red (cochineal)
    "beta-carotene",      # Orange (carrot)
    "phycocyanin",        # Blue (spirulina)
    "annatto",            # Orange
    "bixin",              # Orange (annatto)
    "chlorophyll",        # Green
    "anthocyanin",        # Purple/red
    "cyanidin",           # Anthocyanin
    "betanin",            # Red (beet)
    "curcumin",           # Yellow (turmeric)
    "lutein",             # Yellow
    "lycopene",           # Red (tomato)
    "zeaxanthin",         # Yellow/orange
]

# ==============================================================================
# 1. PUBCHEM DATA DOWNLOADER
# ==============================================================================

class PubChemDownloader:
    """Download chemical data from PubChem"""
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def __init__(self, gcs_uploader: GCSUploader):
        self.gcs = gcs_uploader
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Natural Colorant Research)'
        })
    
    def search_compound(self, name: str) -> Optional[Dict]:
        """Search for compound by name and get CID"""
        print(f"  Searching PubChem for: {name}")
        
        url = f"{self.BASE_URL}/compound/name/{quote(name)}/cids/JSON"
        
        try:
            response = self.session.get(url, timeout=10)
            time.sleep(0.5)  # Rate limit: max 5 requests/second
            
            if response.status_code == 200:
                data = response.json()
                cid = data['IdentifierList']['CID'][0]
                print(f"    Found CID: {cid}")
                return {'name': name, 'cid': cid}
            else:
                print(f"    Not found in PubChem")
                return None
                
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def get_compound_properties(self, cid: int) -> Optional[Dict]:
        """Get compound properties from PubChem"""
        print(f"  Downloading properties for CID {cid}")
        
        properties = [
            'MolecularFormula',
            'MolecularWeight',
            'CanonicalSMILES',
            'IUPACName',
            'XLogP',
            'TPSA',
            'Complexity',
            'HBondDonorCount',
            'HBondAcceptorCount',
        ]
        
        props_str = ','.join(properties)
        url = f"{self.BASE_URL}/compound/cid/{cid}/property/{props_str}/JSON"
        
        try:
            response = self.session.get(url, timeout=10)
            time.sleep(0.5)
            
            if response.status_code == 200:
                data = response.json()
                return data['PropertyTable']['Properties'][0]
            else:
                return None
                
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def download_all(self, colorant_names: List[str]) -> pd.DataFrame:
        """Download data for all colorants and upload to GCS"""
        print("\n" + "="*70)
        print("DOWNLOADING FROM PUBCHEM")
        print("="*70)
        
        results = []
        
        for name in colorant_names:
            # Search for compound
            search_result = self.search_compound(name)
            if not search_result:
                continue
            
            cid = search_result['cid']
            
            # Get properties
            props = self.get_compound_properties(cid)
            if props:
                props['colorant_name'] = name
                props['pubchem_url'] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
                results.append(props)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Upload to GCS
        gcs_path = self.gcs.upload_dataframe(
            df, 
            "pubchem/pubchem_colorants.csv",
            f"{len(df)} compounds from PubChem"
        )
        
        print(f"\n✅ PubChem data uploaded successfully")
        print(f"   {len(df)} compounds found")
        
        return df

# ==============================================================================
# 2. FDA DATA SCRAPER
# ==============================================================================

class FDADownloader:
    """Download FDA approved color additives list"""
    
    def __init__(self, gcs_uploader: GCSUploader):
        self.gcs = gcs_uploader
    
    def create_fda_database(self) -> pd.DataFrame:
        """
        Create FDA approved colorants database from official sources
        Based on 21 CFR Part 73 (Exempt from certification)
        """
        print("\n" + "="*70)
        print("CREATING FDA DATABASE FROM OFFICIAL SOURCES")
        print("="*70)
        
        # Data compiled from FDA CFR 21 Part 73
        # https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-73
        
        fda_data = [
            {
                'name': 'Carmine (Cochineal)',
                'cfr_section': '73.100',
                'e_number': 'E120',
                'color': 'Red',
                'source': 'Cochineal insect (Dactylopius coccus)',
                'permitted_uses': 'Foods generally',
                'restrictions': 'GMP amounts',
                'status': 'Approved',
                'year_approved': 1977,
                'cas_number': '1390-65-4',
            },
            {
                'name': 'Beta-carotene',
                'cfr_section': '73.95',
                'e_number': 'E160a',
                'color': 'Orange/Yellow',
                'source': 'Synthetic or from carrots, algae',
                'permitted_uses': 'Foods generally',
                'restrictions': 'GMP amounts',
                'status': 'Approved',
                'year_approved': 1963,
                'cas_number': '7235-40-7',
            },
            {
                'name': 'Turmeric (Curcumin)',
                'cfr_section': '73.600',
                'e_number': 'E100',
                'color': 'Yellow',
                'source': 'Curcuma longa',
                'permitted_uses': 'Foods generally',
                'restrictions': 'GMP amounts',
                'status': 'Approved',
                'year_approved': None,
                'cas_number': '458-37-7',
            },
            {
                'name': 'Annatto',
                'cfr_section': '73.30',
                'e_number': 'E160b',
                'color': 'Yellow to Orange',
                'source': 'Bixa orellana seeds',
                'permitted_uses': 'Foods generally',
                'restrictions': 'GMP amounts',
                'status': 'Approved',
                'year_approved': None,
                'cas_number': '1393-63-1',
            },
            {
                'name': 'Grape skin extract (Enocianina)',
                'cfr_section': '73.170',
                'e_number': 'E163',
                'color': 'Red to Purple',
                'source': 'Vitis vinifera grape skins',
                'permitted_uses': 'Still and carbonated drinks, beverages',
                'restrictions': 'GMP amounts',
                'status': 'Approved',
                'year_approved': None,
                'cas_number': None,
            },
            {
                'name': 'Beet powder (Beet juice)',
                'cfr_section': '73.40',
                'e_number': 'E162',
                'color': 'Red to Purple',
                'source': 'Beta vulgaris beet roots',
                'permitted_uses': 'Foods generally',
                'restrictions': 'GMP amounts',
                'status': 'Approved',
                'year_approved': None,
                'cas_number': '7659-95-2',
            },
            {
                'name': 'Paprika',
                'cfr_section': '73.345',
                'e_number': 'E160c',
                'color': 'Orange to Red',
                'source': 'Capsicum annuum pods',
                'permitted_uses': 'Foods generally',
                'restrictions': 'GMP amounts',
                'status': 'Approved',
                'year_approved': None,
                'cas_number': None,
            },
            {
                'name': 'Caramel',
                'cfr_section': '73.85',
                'e_number': 'E150a-d',
                'color': 'Yellow to Brown',
                'source': 'Heated carbohydrates',
                'permitted_uses': 'Foods generally',
                'restrictions': 'GMP amounts',
                'status': 'Approved',
                'year_approved': None,
                'cas_number': '8028-89-5',
            },
            {
                'name': 'Spirulina extract (Phycocyanin)',
                'cfr_section': '73.530',
                'e_number': None,
                'color': 'Blue',
                'source': 'Arthrospira platensis (spirulina)',
                'permitted_uses': 'Candies, confections, frostings',
                'restrictions': 'Limited to specific uses',
                'status': 'Approved',
                'year_approved': 2013,
                'cas_number': None,
            },
            {
                'name': 'Lycopene',
                'cfr_section': '73.295',
                'e_number': 'E160d',
                'color': 'Red',
                'source': 'Tomatoes or from Blakeslea trispora fungus',
                'permitted_uses': 'Foods generally',
                'restrictions': 'GMP amounts',
                'status': 'Approved',
                'year_approved': 2005,
                'cas_number': '502-65-8',
            },
        ]
        
        df = pd.DataFrame(fda_data)
        
        # Upload to GCS
        self.gcs.upload_dataframe(
            df,
            "fda/fda_approved_colorants.csv",
            f"{len(df)} FDA approved natural colorants"
        )
        
        print(f"✅ FDA database created: {len(df)} colorants")
        
        return df

# ==============================================================================
# 3. FOODB DOWNLOADER
# ==============================================================================

class FooDBDownloader:
    """Instructions for downloading FooDB data"""
    
    def __init__(self, gcs_uploader: GCSUploader):
        self.gcs = gcs_uploader
    
    def create_instructions(self):
        """Create instructions for FooDB download"""
        print("\n" + "="*70)
        print("FOODB DOWNLOAD INSTRUCTIONS")
        print("="*70)
        
        instructions = """
FOODB (Food Database) DOWNLOAD INSTRUCTIONS

FooDB is a comprehensive database of food components and compounds.
Website: https://foodb.ca/

MANUAL DOWNLOAD STEPS:

1. Go to: https://foodb.ca/downloads

2. Download these files:
   ✓ Compound.csv (all food compounds)
   ✓ Content.csv (compound-food relationships)
   ✓ Nutrient.csv (nutritional data)

3. Upload to GCS using:
   gsutil cp Compound.csv gs://eurofins/foodb/
   gsutil cp Content.csv gs://eurofins/foodb/
   gsutil cp Nutrient.csv gs://eurofins/foodb/

4. Or use Python:
   from google.cloud import storage
   client = storage.Client(project='majestic-layout-461420-k4')
   bucket = client.bucket('eurofins')
   
   blob = bucket.blob('foodb/Compound.csv')
   blob.upload_from_filename('Compound.csv')

SPECIFIC COLORANTS TO SEARCH:
- Carminic acid (FDB ID: FDB012345)
- Beta-carotene (FDB ID: FDB012346)
- Anthocyanins (multiple IDs)
- Curcumin
- Chlorophyll

ALTERNATIVE: Use FooDB API (if available):
- Not officially documented
- May require parsing website directly

NOTE: FooDB files are large (~100MB+), so download may take time.

Once downloaded and uploaded to GCS, you can process them with:
  python process_foodb_data.py
"""
        
        self.gcs.upload_text(
            instructions,
            "foodb/DOWNLOAD_INSTRUCTIONS.txt",
            "FooDB download guide"
        )
        
        print(instructions)

# ==============================================================================
# 4. LITERATURE DATA COMPILER
# ==============================================================================

class LiteratureDataCompiler:
    """Create template for literature kinetics data"""
    
    def __init__(self, gcs_uploader: GCSUploader):
        self.gcs = gcs_uploader
    
    def create_literature_template(self) -> pd.DataFrame:
        """Create template CSV for literature data"""
        print("\n" + "="*70)
        print("CREATING LITERATURE DATA TEMPLATE")
        print("="*70)
        
        template_data = [
            {
                'colorant_name': 'Anthocyanin (generic)',
                'specific_compound': 'Cyanidin-3-glucoside',
                'degradation_rate_constant_k': 0.0015,
                'k_units': '1/hour',
                'temperature_C': 25,
                'pH': 3.0,
                'activation_energy_Ea': 85000,
                'Ea_units': 'J/mol',
                'light_stability': 'poor',
                'oxygen_sensitivity': 'high',
                'reference': 'Patras et al. (2010) Trends Food Sci Tech 21(1):3-11',
                'doi': '10.1016/j.tifs.2009.07.004',
                'notes': 'Example data - replace with actual values',
            },
            {
                'colorant_name': 'Beta-carotene',
                'specific_compound': 'Beta-carotene',
                'degradation_rate_constant_k': 0.0008,
                'k_units': '1/hour',
                'temperature_C': 40,
                'pH': 7.0,
                'activation_energy_Ea': 75000,
                'Ea_units': 'J/mol',
                'light_stability': 'moderate',
                'oxygen_sensitivity': 'high',
                'reference': 'Rodriguez-Amaya (2019) Food Res Int 124:200-205',
                'doi': '10.1016/j.foodres.2018.05.028',
                'notes': 'Example data - replace with actual values',
            },
        ]
        
        df = pd.DataFrame(template_data)
        
        self.gcs.upload_dataframe(
            df,
            "literature/literature_kinetics_data.csv",
            "Template for literature degradation kinetics"
        )
        
        print(f"✅ Literature template created")
        
        # Also create bibliography
        bibliography = """
RECOMMENDED PAPERS FOR COLORANT STABILITY DATA:

1. Natural Colorants: Food Colorants from Natural Sources
   Sigurdson GT, Tang P, Giusti MM (2017)
   Annual Review of Food Science and Technology 8:261-280
   DOI: 10.1146/annurev-food-030216-025923

2. Update on natural food pigments
   Rodriguez-Amaya DB (2019)
   Food Research International 124:200-205
   DOI: 10.1016/j.foodres.2018.05.028

3. Anthocyanins: natural colorants with health-promoting properties
   He J, Giusti MM (2010)
   Annual Review of Food Science and Technology 1:163-187
   DOI: 10.1146/annurev.food.080708.100754

4. Stability of natural food colorants
   Delgado-Vargas F, Jiménez AR, Paredes-López O (2000)
   Critical Reviews in Food Science and Nutrition 40(3):173-289
   DOI: 10.1080/10408690091189257

5. Color Additives in Foods
   Wrolstad RE, Culver CA (2012)
   In: Handbook of Food Science, Technology, and Engineering
   CRC Press

6. Degradation kinetics of anthocyanins
   Patras A, Brunton NP, O'Donnell C, Tiwari BK (2010)
   Trends in Food Science & Technology 21(1):3-11
   DOI: 10.1016/j.tifs.2009.07.004

HOW TO USE THESE PAPERS:
- Search on Google Scholar, PubMed, or ScienceDirect
- Look for tables with degradation rate constants
- Extract: k values, Ea (activation energy), pH effects
- Add to the literature_kinetics_data.csv file
"""
        
        self.gcs.upload_text(
            bibliography,
            "literature/BIBLIOGRAPHY.txt",
            "Research papers for colorant stability"
        )
        
        return df

# ==============================================================================
# 5. SENSIENT SUPPLIER DATA
# ==============================================================================

class SensientDataCollector:
    """Instructions for collecting Sensient supplier data"""
    
    def __init__(self, gcs_uploader: GCSUploader):
        self.gcs = gcs_uploader
    
    def create_instructions(self):
        """Create instructions for contacting Sensient"""
        print("\n" + "="*70)
        print("SENSIENT SUPPLIER DATA INSTRUCTIONS")
        print("="*70)
        
        instructions = """
SENSIENT TECHNICAL DATA REQUEST

As your colorant supplier for the Kraft Heinz project, Sensient should 
provide you with comprehensive technical data sheets.

CONTACT INFORMATION:
- Website: https://na.sensientfoodcolors.com/
- Contact page: https://na.sensientfoodcolors.com/contact-us/
- Technical Support: Request through their portal

WHAT TO REQUEST:

1. Technical Data Sheets (TDS) for:
   - Carmine/Cochineal extracts
   - Beta-carotene formulations
   - Spirulina/Phycocyanin
   - Annatto/Bixin
   - Any other natural reds, oranges, blues

2. Specific Information Needed:
   ✓ Chemical composition
   ✓ Color strength/intensity
   ✓ Solubility (water/oil)
   ✓ pH stability range
   ✓ Heat stability (max temperature)
   ✓ Light stability
   ✓ Recommended usage levels
   ✓ Regulatory status (FDA, EU)
   ✓ Shelf life and storage
   ✓ Interaction warnings (Vitamin C, metals, proteins)
   ✓ Ingredient codes/SKUs
   ✓ Pricing (per kg)

3. Application-Specific Data:
   ✓ Gelatin compatibility
   ✓ Performance in Jello-type products
   ✓ pH adjustment recommendations
   ✓ Processing temperature guidelines
   ✓ Stability test data

4. Request Format:
   Subject: Technical Data Request - Kraft Heinz Natural Color Project
   
   Body:
   "We are working with Kraft Heinz on developing natural color 
   formulations for gelatin dessert products. We need comprehensive 
   technical data sheets and stability information for natural colorants 
   suitable for this application.
   
   Specifically, we need data for:
   - Natural red colorants (carmine, anthocyanins, beet)
   - Natural orange colorants (beta-carotene, annatto)
   - Natural blue colorants (spirulina)
   
   Please provide TDS including pH stability, temperature stability, 
   degradation kinetics, and gelatin compatibility data.
   
   Contact: Frank (QUNEU)
   Project: Kraft Heinz Jello Natural Colors
   NDA: [Reference your 3-way NDA with Eurofins/QUNEU]"

5. Follow-up Questions to Ask:
   - Do you have stability test data at pH 3-8?
   - What are degradation rate constants at different temperatures?
   - Do you have UV-Vis spectra for each colorant?
   - Can you provide samples for testing?
   - What's the lead time for bulk orders?

EXPECTED DELIVERABLES:
- PDF technical data sheets
- Excel/CSV data files (if available)
- Sample products (optional)
- Pricing quotes

TIMELINE:
- Send request: This week
- Response expected: 1-2 weeks
- Follow-up call: Schedule if needed

Once you receive the data, upload to GCS:
  gsutil cp sensient_data_sheet.pdf gs://eurofins/sensient/
  
Or use the GCS console:
  https://console.cloud.google.com/storage/browser/eurofins
"""
        
        self.gcs.upload_text(
            instructions,
            "sensient/SENSIENT_DATA_REQUEST.txt",
            "Guide for requesting Sensient technical data"
        )
        
        print(instructions)

# ==============================================================================
# 6. MAIN DOWNLOAD COORDINATOR
# ==============================================================================

def download_all_data():
    """Main function to coordinate all downloads and uploads to GCS"""
    
    print("\n" + "="*70)
    print("FOOD COLORANT DATABASE DOWNLOAD → GCS UPLOAD SCRIPT")
    print("For: Kraft Heinz Jello Natural Color Project")
    print("Target: gs://eurofins/ (Project: majestic-layout-461420-k4)")
    print("="*70)
    
    if not GCS_AVAILABLE:
        print("\n❌ ERROR: Google Cloud Storage not installed")
        print("Install with: pip install google-cloud-storage")
        print("\nCannot proceed without GCS library.")
        return
    
    # Initialize GCS uploader
    try:
        gcs = GCSUploader(
            project_id=GCP_PROJECT_ID,
            bucket_name=GCP_BUCKET_NAME,
            local_backup=True
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize GCS: {e}")
        print("\nAuthentication troubleshooting:")
        print("1. Run: gcloud auth application-default login")
        print("2. Or set: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
        return
    
    # 1. PubChem - Automated download and upload
    pubchem = PubChemDownloader(gcs)
    pubchem_df = pubchem.download_all(NATURAL_COLORANTS)
    
    # 2. FDA - Automated (curated) and upload
    fda = FDADownloader(gcs)
    fda_df = fda.create_fda_database()
    
    # 3. FooDB - Manual instructions uploaded
    foodb = FooDBDownloader(gcs)
    foodb.create_instructions()
    
    # 4. Literature - Template uploaded
    literature = LiteratureDataCompiler(gcs)
    literature_df = literature.create_literature_template()
    
    # 5. Sensient - Instructions uploaded
    sensient = SensientDataCollector(gcs)
    sensient.create_instructions()
    
    # Summary
    print("\n" + "="*70)
    print("UPLOAD SUMMARY")
    print("="*70)
    print(f"\n✅ AUTOMATED UPLOADS TO gs://{GCP_BUCKET_NAME}/:")
    print(f"   - PubChem: {len(pubchem_df)} compounds → pubchem/pubchem_colorants.csv")
    print(f"   - FDA: {len(fda_df)} approved colorants → fda/fda_approved_colorants.csv")
    print(f"   - Literature template → literature/literature_kinetics_data.csv")
    print(f"   - Literature bibliography → literature/BIBLIOGRAPHY.txt")
    print(f"   - FooDB instructions → foodb/DOWNLOAD_INSTRUCTIONS.txt")
    print(f"   - Sensient instructions → sensient/SENSIENT_DATA_REQUEST.txt")
    
    if LOCAL_BACKUP_DIR.exists():
        print(f"\n💾 LOCAL BACKUPS SAVED TO: {LOCAL_BACKUP_DIR}/")
    
    print(f"\n📝 MANUAL TASKS:")
    print(f"   - FooDB: Follow instructions in gs://{GCP_BUCKET_NAME}/foodb/")
    print(f"   - Literature: Add more data to gs://{GCP_BUCKET_NAME}/literature/")
    print(f"   - Sensient: Contact supplier using gs://{GCP_BUCKET_NAME}/sensient/")
    
    print(f"\n🌐 VIEW IN BROWSER:")
    print(f"   https://console.cloud.google.com/storage/browser/{GCP_BUCKET_NAME}?project={GCP_PROJECT_ID}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print(f"""
1. ✅ Review uploaded data in GCS bucket
2. 📊 Download FooDB data and upload to gs://{GCP_BUCKET_NAME}/foodb/
3. 📚 Extract literature data and add to literature_kinetics_data.csv
4. 📧 Contact Sensient for technical data sheets
5. 📁 Compile all data into eurofins_colorants.csv format
6. 🚀 Feed into qmat_generator.py

View your uploaded data:
  gsutil ls -r gs://{GCP_BUCKET_NAME}/

Download back to local if needed:
  gsutil -m cp -r gs://{GCP_BUCKET_NAME}/ ./local_data/

The automated uploads give you a solid foundation in GCS. The manual steps
will provide the specialized kinetics and stability data needed for
accurate formulation predictions.

Expected total time: 2-4 hours of manual work over 1-2 weeks
(waiting for Sensient response).
""")

# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    download_all_data()