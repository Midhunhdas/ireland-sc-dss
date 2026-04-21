"""
consolidate_medtech_new.py
Ireland SC-DSS -- MedTech Comtrade consolidation
Reads: data/raw/medtech_comtrade_imports.csv
       data/raw/medtech_comtrade_exports.csv
Writes: data/processed/medtech_new_comtrade.csv

HS Code hierarchy (NO double counting):
  Parent codes (overview totals): 9018, 9021, 9022
  Sub-codes  (product analysis):  901839, 901890, 902190
"""
import os
import pandas as pd
import numpy as np
from fx_rates import USD_TO_EUR, FALLBACK_RATE

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(THIS_DIR, "data", "raw")
PROC_DIR = os.path.join(THIS_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ── HS code metadata ──────────────────────────────────────────────────────────
PARENT_CODES  = ['9018','9021','9022']
SUB_CODES     = ['901839','901890','902190']
PARENT_LABELS = {
    '9018': 'Medical Instruments',
    '9021': 'Implants and Stents',
    '9022': 'X-Ray Apparatus',
}
SUB_LABELS = {
    '901839': 'Catheters and Cannulae',
    '901890': 'Other Medical Instruments',
    '902190': 'Cardiovascular Implants and Stents',
}
ALL_LABELS = {**PARENT_LABELS, **SUB_LABELS}

# ── Geography ─────────────────────────────────────────────────────────────────
CONTINENT_MAP = {
    # Americas
    'United States':'Americas','USA':'Americas','Canada':'Americas','Mexico':'Americas',
    'Brazil':'Americas','Argentina':'Americas','Colombia':'Americas',
    'Chile':'Americas','Venezuela':'Americas','Trinidad and Tobago':'Americas',
    'Costa Rica':'Americas','Dominican Rep.':'Americas','Dominican Republic':'Americas',
    'Panama':'Americas','Peru':'Americas','Ecuador':'Americas','Uruguay':'Americas',
    'Guatemala':'Americas','Honduras':'Americas','El Salvador':'Americas',
    'Jamaica':'Americas','Cuba':'Americas','Puerto Rico':'Americas',
    'Bahamas':'Americas','Barbados':'Americas','Bolivia (Plurinational State of)':'Americas',
    # Europe
    'United Kingdom':'Europe','Germany':'Europe','Netherlands':'Europe',
    'France':'Europe','Belgium':'Europe','Switzerland':'Europe',
    'Italy':'Europe','Spain':'Europe','Denmark':'Europe','Sweden':'Europe',
    'Norway':'Europe','Austria':'Europe','Finland':'Europe','Poland':'Europe',
    'Ireland':'Europe','Czech Republic':'Europe','Czechia':'Europe','Portugal':'Europe',
    'Hungary':'Europe','Romania':'Europe','Greece':'Europe','Slovakia':'Europe',
    'Slovenia':'Europe','Iceland':'Europe','Luxembourg':'Europe','Malta':'Europe',
    'Estonia':'Europe','Latvia':'Europe','Lithuania':'Europe','Bulgaria':'Europe',
    'Croatia':'Europe','Cyprus':'Europe','Serbia':'Europe','Albania':'Europe',
    'Bosnia Herzegovina':'Europe','North Macedonia':'Europe','Montenegro':'Europe',
    'Moldova, Republic of':'Europe','Rep. of Moldova':'Europe','Belarus':'Europe',
    'Ukraine':'Europe','Türkiye':'Europe','Turkey':'Europe',
    # Asia
    'Japan':'Asia','China':'Asia','Rep. of Korea':'Asia','India':'Asia',
    'Singapore':'Asia','Malaysia':'Asia','Taiwan':'Asia','Other Asia, nes':'Asia',
    'Thailand':'Asia','Indonesia':'Asia','Viet Nam':'Asia','Vietnam':'Asia',
    'Philippines':'Asia','Bangladesh':'Asia','Pakistan':'Asia','Sri Lanka':'Asia',
    'Myanmar':'Asia','Cambodia':'Asia','Lao People\'s Dem. Rep.':'Asia',
    'Nepal':'Asia','Mongolia':'Asia','China, Hong Kong SAR':'Asia',
    'China, Macao SAR':'Asia','Kazakhstan':'Asia','Uzbekistan':'Asia',
    'Kyrgyzstan':'Asia','Turkmenistan':'Asia','Tajikistan':'Asia',
    "Dem. People's Rep. of Korea":'Asia',
    # Oceania
    'Australia':'Oceania','New Zealand':'Oceania','Fiji':'Oceania',
    'Papua New Guinea':'Oceania','French Polynesia':'Oceania',
    'FS Micronesia':'Oceania','Samoa':'Oceania',
    # Middle East
    'Israel':'Middle East','Saudi Arabia':'Middle East','State of Palestine':'Middle East',
    'United Arab Emirates':'Middle East','Kuwait':'Middle East',
    'Qatar':'Middle East','Bahrain':'Middle East','Oman':'Middle East',
    'Jordan':'Middle East','Lebanon':'Middle East','Iraq':'Middle East',
    'Iran':'Middle East','Yemen':'Middle East','Syria':'Middle East',
    # Africa
    'South Africa':'Africa','Egypt':'Africa','Nigeria':'Africa','Morocco':'Africa',
    'Tunisia':'Africa','Algeria':'Africa','Kenya':'Africa','Ghana':'Africa',
    'Ethiopia':'Africa','Tanzania, United Rep. of':'Africa','Uganda':'Africa',
    'Angola':'Africa','Mozambique':'Africa','Senegal':'Africa',
    'Côte d\'Ivoire':'Africa','Cameroon':'Africa','Zambia':'Africa',
    'Zimbabwe':'Africa','Botswana':'Africa','Namibia':'Africa','Mauritius':'Africa',
    'Libya':'Africa','Sudan':'Africa','Gabon':'Africa','Rwanda':'Africa',
    'Malawi':'Africa','Togo':'Africa','Benin':'Africa','Mali':'Africa',
    'Dem. Rep. of the Congo':'Africa','Congo':'Africa','Madagascar':'Africa',
    'Burkina Faso':'Africa','Guinea':'Africa','Eritrea':'Africa',
}
COORDS = {
    'United States':(39.5,-98.4),'USA':(39.5,-98.4),'United Kingdom':(55.4,-3.4),
    'Germany':(51.2,10.5),'Netherlands':(52.1,5.3),'Belgium':(50.5,4.5),
    'France':(46.2,2.2),'Switzerland':(46.8,8.2),'Italy':(41.9,12.6),
    'Spain':(40.0,-4.0),'Denmark':(56.0,10.0),'Sweden':(63.0,16.0),
    'Norway':(60.5,8.5),'Austria':(47.5,14.6),'Poland':(51.9,19.1),
    'Japan':(36.2,138.3),'China':(35.9,104.2),'Rep. of Korea':(35.9,127.8),
    'India':(20.6,78.9),'Singapore':(1.3,103.8),'Malaysia':(4.2,108.0),
    'Australia':(-25.3,133.8),'Canada':(56.1,-106.3),
    'Mexico':(23.6,-102.6),'Brazil':(-14.2,-51.9),
    'Israel':(31.0,34.9),'South Africa':(-30.6,22.9),
    'Czech Republic':(49.8,15.5),'Czechia':(49.8,15.5),'Portugal':(39.4,-8.2),
    'Hungary':(47.2,19.5),'Romania':(45.9,24.9),
    'Saudi Arabia':(23.9,45.1),'United Arab Emirates':(23.4,53.8),
    'Viet Nam':(14.1,108.3),'Vietnam':(14.1,108.3),'Thailand':(15.9,100.9),
    'New Zealand':(-40.9,174.9),'Taiwan':(23.7,120.9),
    'Indonesia':(-0.8,113.9),'Egypt':(26.8,30.8),
    'Slovakia':(48.7,19.7),'Slovenia':(46.1,14.8),'Türkiye':(38.9,35.2),'Turkey':(38.9,35.2),
    'Iceland':(64.9,-19.0),'Costa Rica':(9.7,-83.8),'Dominican Rep.':(18.7,-70.2),
    'Dominican Republic':(18.7,-70.2),'China, Hong Kong SAR':(22.3,114.2),
    'China, Macao SAR':(22.2,113.5),'Finland':(61.9,25.7),
    'Luxembourg':(49.8,6.1),'Malta':(35.9,14.5),'Ireland':(53.4,-8.0),
    'Estonia':(58.6,25.0),'Latvia':(56.9,24.6),'Lithuania':(55.2,23.9),
    'Bulgaria':(42.7,25.5),'Croatia':(45.1,15.2),'Cyprus':(35.1,33.4),
    'Greece':(39.1,21.8),'Ukraine':(48.4,31.2),
    'Kuwait':(29.3,47.5),'Qatar':(25.4,51.2),'Bahrain':(26.0,50.6),
    'Oman':(21.5,55.9),'Jordan':(30.6,36.2),'Lebanon':(33.9,35.9),
    'Philippines':(12.9,121.8),'Bangladesh':(23.7,90.4),'Pakistan':(30.4,69.3),
    'Sri Lanka':(7.9,80.8),'Myanmar':(21.9,95.9),
    'Morocco':(31.8,-7.1),'Tunisia':(33.9,9.5),'Algeria':(28.0,1.7),
    'Kenya':(-0.0,37.9),'Ghana':(7.9,-1.0),'Ethiopia':(9.1,40.5),
    'Nigeria':(9.1,8.7),'Libya':(26.3,17.2),
    'Argentina':(-38.4,-63.6),'Chile':(-35.7,-71.5),'Colombia':(4.6,-74.3),
    'Peru':(-9.2,-75.0),'Venezuela':(6.4,-66.6),'Ecuador':(-1.8,-78.2),
    'Uruguay':(-32.5,-55.8),'Bolivia (Plurinational State of)':(-16.3,-63.6),
    'Panama':(8.5,-80.8),'Guatemala':(15.8,-90.2),'Jamaica':(18.1,-77.3),
    'Cuba':(21.5,-77.8),'Puerto Rico':(18.2,-66.6),
    'Iran':(32.4,53.7),'Iraq':(33.2,43.7),'Yemen':(15.6,48.5),
    'Kazakhstan':(48.0,66.9),'Uzbekistan':(41.4,64.6),
    'Fiji':(-16.6,179.4),'Papua New Guinea':(-6.3,143.9),
    'Mongolia':(46.9,103.8),'Mauritius':(-20.3,57.6),
}

JUNK_PARTNERS = {
    'world','areas, nes','other','unspecified','not specified','total',
    'free zones','bunkers','low value trade','special categories',
    'other asia, nes','other europe, nes','other africa, nes',
    'other america, nes','other oceania',
}

# ── Column remap helper ───────────────────────────────────────────────────────
def remap_df(df, flow_code):
    """
    The raw Comtrade files have shifted columns due to encoding.
    True column mapping (discovered by inspection):
      isOriginalClassification -> HS numeric code
      partnerISO               -> country name
      refMonth                 -> year (2015-2024)
      fobvalue                 -> trade value (USD)
      cmdCode                  -> HS description text

    Currency conversion: UN Comtrade values are USD. We convert to EUR using
    the annual average rate (ECB reference data), stored in fx_rates.py.
    """
    out = pd.DataFrame()
    out['refYear']      = pd.to_numeric(df['refMonth'], errors='coerce').fillna(0).astype(int)
    out['flowCode']     = flow_code
    out['partnerISO']   = df['partnerISO'].astype(str).str.strip()
    out['cmdCode']      = df['isOriginalClassification'].astype(str).str.strip()
    out['cmdDesc']      = df['cmdCode'].astype(str).str[:80]   # description from original cmdCode col
    value_usd           = pd.to_numeric(df['fobvalue'], errors='coerce').fillna(0)
    # Apply year-specific EUR/USD rate
    fx_series = out['refYear'].map(USD_TO_EUR).fillna(FALLBACK_RATE)
    out['primaryValue'] = value_usd * fx_series  # now in EUR
    return out

# ── Load files ────────────────────────────────────────────────────────────────
print("[MedTech] Loading raw files...")
imp_path = os.path.join(RAW_DIR, "medtech_comtrade_imports.csv")
exp_path = os.path.join(RAW_DIR, "medtech_comtrade_exports.csv")

if not os.path.exists(imp_path):
    raise FileNotFoundError(f"Missing: {imp_path}")
if not os.path.exists(exp_path):
    raise FileNotFoundError(f"Missing: {exp_path}")

imp_raw = pd.read_csv(imp_path, encoding='latin-1', low_memory=False)
exp_raw = pd.read_csv(exp_path, encoding='latin-1', low_memory=False)
print(f"  Imports raw: {len(imp_raw):,} rows")
print(f"  Exports raw: {len(exp_raw):,} rows")

# ── Remap columns ─────────────────────────────────────────────────────────────
imp = remap_df(imp_raw, 'M')
exp = remap_df(exp_raw, 'X')
df  = pd.concat([imp, exp], ignore_index=True)

# ── Classify HS codes ─────────────────────────────────────────────────────────
df['hsLabel']  = df['cmdCode'].map(ALL_LABELS).fillna(df['cmdDesc'].str[:60])
df['isParent'] = df['cmdCode'].isin(PARENT_CODES)
df['isSub']    = df['cmdCode'].isin(SUB_CODES)
df['codeType'] = df['cmdCode'].apply(
    lambda x: 'parent' if x in PARENT_CODES else ('sub' if x in SUB_CODES else 'other')
)

# ── Remove junk partners ──────────────────────────────────────────────────────
before = len(df)
df = df[~df['partnerISO'].str.strip().str.lower().isin(JUNK_PARTNERS)].copy()
print(f"  Removed {before - len(df)} junk/aggregate partner rows")

# ── Geography enrichment ──────────────────────────────────────────────────────
df['continent']   = df['partnerISO'].map(CONTINENT_MAP).fillna('Other')
df['partner_lat'] = df['partnerISO'].map(lambda x: COORDS.get(x, (0,0))[0])
df['partner_lon'] = df['partnerISO'].map(lambda x: COORDS.get(x, (0,0))[1])

# ── Validation ────────────────────────────────────────────────────────────────
print(f"\n[MedTech] Consolidated: {len(df):,} rows")
print(f"  Years:      {sorted(df['refYear'].unique())}")
print(f"  HS codes:   {sorted(df['cmdCode'].unique())}")
print(f"  Countries:  {df['partnerISO'].nunique()} unique")
print(f"  Code types: {df['codeType'].value_counts().to_dict()}")

y23 = df[df['refYear']==2023]
exp_23 = y23[(y23['codeType']=='parent')&(y23['flowCode']=='X')]['primaryValue'].sum()
imp_23 = y23[(y23['codeType']=='parent')&(y23['flowCode']=='M')]['primaryValue'].sum()
print(f"\n  2023 parent exports: EUR {exp_23/1e9:.2f}bn  (converted from USD using 2023 rate)")
print(f"  2023 parent imports: EUR {imp_23/1e9:.2f}bn")
print(f"  2023 trade surplus:  EUR {(exp_23-imp_23)/1e9:.2f}bn")

# Continent coverage check
other_val = df[df['continent']=='Other']['primaryValue'].sum()
total_val = df['primaryValue'].sum()
print(f"\n  Continent coverage: {(1-other_val/total_val)*100:.1f}% mapped  ('Other' bucket = {other_val/total_val*100:.1f}% of value)")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(PROC_DIR, "medtech_new_comtrade.csv")
# Ensure cmdCode is saved as string to avoid int/str mismatch on reload
df['cmdCode'] = df['cmdCode'].astype(str)
df.to_csv(out_path, index=False)
print(f"\n[MedTech] Saved: {out_path}")
print(f"  Rows: {len(df):,} | Columns: {df.columns.tolist()}")
print("[MedTech] Done.")
