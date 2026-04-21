"""
consolidate_agri.py — Ireland Agriculture & Food Dashboard
Reads raw Comtrade XLSXs, CSO XLSXs, and FAOSTAT CSV.
Produces clean processed CSVs in data/processed/.

Raw files expected in data/raw/:
  agri_comtrade_hs0110_2010_2021_exp.xlsx  |  agri_comtrade_hs0110_2010_2021_imp.xlsx
  agri_comtrade_hs0110_2022_2024_exp.xlsx  |  agri_comtrade_hs0110_2022_2024_imp.xlsx
  agri_comtrade_hs1120_2010_2021_exp.xlsx  |  ... (same pattern for hs1120, hs2122)
  agri_cso_output.xlsx                     (CSO agricultural output 1990-2025)
  agri_cso_trade.xlsx                      (CSO food trade by month 1972-2026)
  faostat_agri.csv                         (FAOSTAT food balances 2010-2023)

Outputs:
  data/processed/agri_comtrade_consolidated.csv
  data/processed/agri_cso_output.csv
  data/processed/agri_cso_trade.csv
  data/processed/agri_faostat.csv
"""
import os, glob
import pandas as pd
import numpy as np
from fx_rates import USD_TO_EUR, FALLBACK_RATE

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(THIS_DIR, "data", "raw")
PROC_DIR = os.path.join(THIS_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ── ISO3 → country name ─────────────────────────────────────────────────────
ISO3_MAP = {
    'GBR':'United Kingdom','IRL':'Ireland','DEU':'Germany','FRA':'France',
    'NLD':'Netherlands','BEL':'Belgium','ESP':'Spain','ITA':'Italy',
    'POL':'Poland','SWE':'Sweden','DNK':'Denmark','NOR':'Norway',
    'FIN':'Finland','AUT':'Austria','CHE':'Switzerland','PRT':'Portugal',
    'LUX':'Luxembourg','GRC':'Greece','CZE':'Czechia','HUN':'Hungary',
    'ROU':'Romania','SVK':'Slovakia','SVN':'Slovenia','HRV':'Croatia',
    'BGR':'Bulgaria','EST':'Estonia','LVA':'Latvia','LTU':'Lithuania',
    'USA':'United States','CAN':'Canada','MEX':'Mexico','BRA':'Brazil',
    'ARG':'Argentina','CHL':'Chile','COL':'Colombia','PER':'Peru',
    'CHN':'China','JPN':'Japan','KOR':'Rep. of Korea','IND':'India',
    'SGP':'Singapore','HKG':'Hong Kong','MYS':'Malaysia','THA':'Thailand',
    'VNM':'Vietnam','IDN':'Indonesia','AUS':'Australia','NZL':'New Zealand',
    'ZAF':'South Africa','NGA':'Nigeria','EGY':'Egypt','MAR':'Morocco',
    'KEN':'Kenya','ETH':'Ethiopia','GHA':'Ghana','TZA':'Tanzania',
    'SAU':'Saudi Arabia','ARE':'United Arab Emirates','TUR':'Türkiye',
    'ISR':'Israel','UKR':'Ukraine','RUS':'Russia','BLR':'Belarus',
    'ISL':'Iceland','ALB':'Albania','SRB':'Serbia','MKD':'North Macedonia',
    'BIH':'Bosnia and Herzegovina','MNE':'Montenegro','MLT':'Malta',
    'CYP':'Cyprus','TWN':'Taiwan','PHL':'Philippines','PAK':'Pakistan',
    'BGD':'Bangladesh','LKA':'Sri Lanka','NPL':'Nepal',
    'JOR':'Jordan','LBN':'Lebanon','QAT':'Qatar','KWT':'Kuwait',
    'BHR':'Bahrain','OMN':'Oman','IRQ':'Iraq','IRN':'Iran',
    'DZA':'Algeria','LBY':'Libya','TUN':'Tunisia','SDN':'Sudan',
    'AGO':'Angola','MOZ':'Mozambique','ZMB':'Zambia','ZWE':'Zimbabwe',
    'CMR':'Cameroon','CIV':"Cote d'Ivoire",'SEN':'Senegal',
    'UGA':'Uganda','RWA':'Rwanda','MDG':'Madagascar',
    'ECU':'Ecuador','BOL':'Bolivia','URY':'Uruguay','VEN':'Venezuela',
    'GTM':'Guatemala','HND':'Honduras','SLV':'El Salvador',
    'CRI':'Costa Rica','PAN':'Panama','CUB':'Cuba','DOM':'Dominican Republic',
    'JAM':'Jamaica','TTO':'Trinidad and Tobago',
    'FRO':'Faroe Islands','GIB':'Gibraltar','BMU':'Bermuda',
    'NIC':'Nicaragua','PRY':'Paraguay',
}

CONTINENT_MAP = {
    'United Kingdom':'Europe','Ireland':'Europe','Germany':'Europe',
    'France':'Europe','Netherlands':'Europe','Belgium':'Europe',
    'Spain':'Europe','Italy':'Europe','Poland':'Europe','Sweden':'Europe',
    'Denmark':'Europe','Norway':'Europe','Finland':'Europe','Austria':'Europe',
    'Switzerland':'Europe','Portugal':'Europe','Luxembourg':'Europe',
    'Greece':'Europe','Czechia':'Europe','Hungary':'Europe','Romania':'Europe',
    'Slovakia':'Europe','Slovenia':'Europe','Croatia':'Europe',
    'Bulgaria':'Europe','Estonia':'Europe','Latvia':'Europe','Lithuania':'Europe',
    'Iceland':'Europe','Albania':'Europe','Serbia':'Europe',
    'Bosnia and Herzegovina':'Europe','North Macedonia':'Europe',
    'Montenegro':'Europe','Malta':'Europe','Cyprus':'Europe',
    'Faroe Islands':'Europe','Gibraltar':'Europe',
    'United States':'Americas','Canada':'Americas','Mexico':'Americas',
    'Brazil':'Americas','Argentina':'Americas','Chile':'Americas',
    'Colombia':'Americas','Peru':'Americas','Ecuador':'Americas',
    'Bolivia':'Americas','Uruguay':'Americas','Venezuela':'Americas',
    'Guatemala':'Americas','Honduras':'Americas','El Salvador':'Americas',
    'Costa Rica':'Americas','Panama':'Americas','Cuba':'Americas',
    'Dominican Republic':'Americas','Jamaica':'Americas',
    'Trinidad and Tobago':'Americas','Nicaragua':'Americas',
    'Paraguay':'Americas','Bermuda':'Americas',
    'China':'Asia','Japan':'Asia','Rep. of Korea':'Asia','India':'Asia',
    'Singapore':'Asia','Hong Kong':'Asia','Malaysia':'Asia','Thailand':'Asia',
    'Vietnam':'Asia','Indonesia':'Asia','Taiwan':'Asia','Philippines':'Asia',
    'Pakistan':'Asia','Bangladesh':'Asia','Sri Lanka':'Asia','Nepal':'Asia',
    'Saudi Arabia':'Middle East','United Arab Emirates':'Middle East',
    'Israel':'Middle East','Jordan':'Middle East','Lebanon':'Middle East',
    'Qatar':'Middle East','Kuwait':'Middle East','Bahrain':'Middle East',
    'Oman':'Middle East','Iraq':'Middle East','Iran':'Middle East',
    'Turkey':'Middle East','Türkiye':'Middle East',
    'Australia':'Oceania','New Zealand':'Oceania',
    'South Africa':'Africa','Nigeria':'Africa','Egypt':'Africa',
    'Morocco':'Africa','Kenya':'Africa','Ethiopia':'Africa','Ghana':'Africa',
    'Tanzania':'Africa','Algeria':'Africa','Libya':'Africa','Tunisia':'Africa',
    'Sudan':'Africa','Angola':'Africa','Mozambique':'Africa','Zambia':'Africa',
    'Zimbabwe':'Africa','Cameroon':'Africa',"Cote d'Ivoire":'Africa',
    'Senegal':'Africa','Uganda':'Africa','Rwanda':'Africa','Madagascar':'Africa',
    'Russia':'Europe/Asia','Ukraine':'Europe/Asia','Belarus':'Europe/Asia',
}

COORDS_MAP = {
    'United Kingdom':(55.4,-3.4),'Germany':(51.2,10.5),'France':(46.2,2.2),
    'Netherlands':(52.1,5.3),'Belgium':(50.5,4.5),'Spain':(40.0,-4.0),
    'Italy':(41.9,12.5),'Poland':(52.0,20.0),'Sweden':(63.0,16.0),
    'Denmark':(56.0,10.0),'Norway':(60.5,8.5),'Finland':(64.0,26.0),
    'Austria':(47.5,14.6),'Switzerland':(46.8,8.2),'Portugal':(39.4,-8.2),
    'Luxembourg':(49.6,6.1),'Greece':(39.1,21.8),'Czechia':(49.8,15.5),
    'Hungary':(47.2,19.5),'Romania':(45.9,24.9),'Slovakia':(48.7,19.7),
    'Slovenia':(46.2,14.8),'Croatia':(45.1,15.2),'Bulgaria':(42.7,25.5),
    'United States':(39.5,-98.4),'Canada':(56.1,-106.3),'Mexico':(23.6,-102.6),
    'Brazil':(-14.2,-51.9),'Argentina':(-38.4,-63.6),'Chile':(-35.7,-71.5),
    'China':(35.9,104.2),'Japan':(36.2,138.3),'Rep. of Korea':(35.9,127.8),
    'India':(20.6,78.9),'Singapore':(1.3,103.8),'Hong Kong':(22.3,114.2),
    'Malaysia':(4.2,108.0),'Australia':(-25.3,133.8),'New Zealand':(-40.9,174.9),
    'South Africa':(-30.6,22.9),'Nigeria':(9.1,8.7),'Egypt':(26.8,30.8),
    'Morocco':(31.8,-7.1),'Saudi Arabia':(23.9,45.1),
    'United Arab Emirates':(23.4,53.8),'Russia':(61.5,105.0),
    'Ukraine':(48.4,31.2),'Turkey':(39.0,35.0),'Türkiye':(39.0,35.0),
    'Israel':(31.0,34.9),'New Zealand':(-40.9,174.9),
}

JUNK_PARTNERS = {
    'world','areas, nes','other europe, nes','other asia, nes',
    'other africa, nes','other america, nes','other oceania, nes',
    'unspecified','total','not specified','free zones','bunkers',
    'low value trade','oecd','asean','eu','european union',
    'special categories','other','unknown', '0',
}

# ── HS chapter → commodity group ────────────────────────────────────────────
HS_GROUP_MAP = {
    1:'Live animals', 2:'Meat & offal', 3:'Fish & seafood',
    4:'Dairy, eggs & honey', 5:'Other animal products',
    6:'Live plants', 7:'Vegetables', 8:'Fruit & nuts',
    9:'Coffee, tea & spices', 10:'Cereals',
    11:'Milling products', 12:'Oil seeds & misc grains',
    13:'Lac, gums & resins', 14:'Vegetable plaiting materials',
    15:'Fats & oils', 16:'Meat & fish preparations',
    17:'Sugars & confectionery', 18:'Cocoa & chocolate',
    19:'Cereal, flour & bakery', 20:'Vegetable preparations',
    21:'Miscellaneous food preparations', 22:'Beverages & spirits',
}

def to_float(s):
    return pd.to_numeric(
        s.astype(str).str.strip().replace({'N/A':'','-':'','nan':'','None':'','':None}),
        errors='coerce').fillna(0.0)

def add_geo(df):
    df['continent']   = df['partnerdesc'].map(lambda x: CONTINENT_MAP.get(x,'Other'))
    df['partner_lat'] = df['partnerdesc'].map(lambda x: COORDS_MAP.get(x,(0,0))[0])
    df['partner_lon'] = df['partnerdesc'].map(lambda x: COORDS_MAP.get(x,(0,0))[1])
    return df

# ── 1. Comtrade ──────────────────────────────────────────────────────────────
print('\n=== AGRICULTURE COMTRADE ===')

# File pairs: (exp_path, imp_path)
hs_groups = ['hs0110', 'hs1120', 'hs2122']
periods   = ['2010_2021', '2022_2024']

frames = []
for hs in hs_groups:
    for period in periods:
        for flow in ['exp', 'imp']:
            fname = f'agri_comtrade_{hs}_{period}_{flow}.xlsx'
            path  = os.path.join(RAW_DIR, fname)
            if not os.path.exists(path):
                print(f'  MISSING: {fname}')
                continue

            df = pd.read_excel(path)

            # Year
            df['refyear'] = pd.to_numeric(df['refYear'], errors='coerce').astype('Int64')

            # Flow
            df['flowcode'] = df['flowCode'].astype(str).str.strip().str.upper()
            # Normalise — Comtrade uses 'X'/'M' in these files
            df['flowcode'] = df['flowcode'].map(lambda v: 'X' if v in ('X','EXPORT','EXPORTS') else 'M')

            # Partner — partnerISO is the ISO3 code
            df['partnerdesc'] = df['partnerISO'].astype(str).str.strip().map(
                lambda c: ISO3_MAP.get(c, c))
            # Also use partnerDesc as fallback if ISO3 not in map
            if 'partnerDesc' in df.columns:
                mask = ~df['partnerISO'].astype(str).str.strip().isin(ISO3_MAP)
                df.loc[mask, 'partnerdesc'] = df.loc[mask, 'partnerDesc'].astype(str).str.strip()

            # Filter junk partners and World (W00 / WLD)
            df = df[~df['partnerISO'].astype(str).str.strip().isin(['W00','WLD','0'])]
            df = df[~df['partnerdesc'].str.lower().isin(JUNK_PARTNERS)]

            # HS commodity
            df['cmdcode'] = pd.to_numeric(df['cmdCode'], errors='coerce').fillna(0).astype(int)
            df['cmddesc'] = df['cmdCode'].map(lambda c: HS_GROUP_MAP.get(int(c) if str(c).isdigit() else 0, str(c)))

            # Value — exports use fobvalue, imports use cifvalue
            # UN Comtrade values are USD; we convert to EUR using annual avg rate.
            if flow == 'exp':
                df['primaryvalue'] = to_float(df['fobvalue'])
            else:
                df['primaryvalue'] = to_float(df['cifvalue'])
            # Fallback to primaryValue if 0
            if 'primaryValue' in df.columns:
                mask = df['primaryvalue'] == 0
                df.loc[mask, 'primaryvalue'] = to_float(df.loc[mask, 'primaryValue'])

            df = df[(df['refyear'] >= 2010) & (df['refyear'] <= 2030)]

            # Apply year-specific EUR/USD rate
            fx_series = df['refyear'].map(USD_TO_EUR).fillna(FALLBACK_RATE)
            df['primaryvalue'] = df['primaryvalue'] * fx_series  # now in EUR

            df = add_geo(df)

            keep = ['refyear','flowcode','partnerdesc','cmdcode','cmddesc',
                    'primaryvalue','continent','partner_lat','partner_lon']
            df = df[keep].dropna(subset=['refyear'])

            total = df['primaryvalue'].sum()
            print(f'  {fname}: {len(df):,} rows | {flow.upper()} | '
                  f'${total/1e9:.1f}B | years {df["refyear"].min()}–{df["refyear"].max()}')
            frames.append(df)

if frames:
    ct = pd.concat(frames, ignore_index=True)
    ct['refyear'] = ct['refyear'].astype(int)
    out = os.path.join(PROC_DIR, 'agri_comtrade_consolidated.csv')
    ct.to_csv(out, index=False)
    print(f'\n✓ Comtrade consolidated: {ct.shape} → {out}')
    print(f'  Exports: ${ct[ct["flowcode"]=="X"]["primaryvalue"].sum()/1e9:.1f}B')
    print(f'  Imports: ${ct[ct["flowcode"]=="M"]["primaryvalue"].sum()/1e9:.1f}B')
    print(f'  Partners: {ct["partnerdesc"].nunique()} unique')
    print(f'  HS groups: {sorted(ct["cmdcode"].unique().tolist())}')
else:
    print('ERROR: No Comtrade data loaded.')

# ── 2 & 3. CSO Agricultural Output + Food Trade ──────────────────────────────
# Auto-detect which file contains output data (has 'Year' col) vs trade data
# (has 'Month' col) — handles cases where files were swapped during renaming.
print('\n=== CSO FILES (auto-detecting output vs trade) ===')

def read_unpivoted(path):
    """Read the Unpivoted sheet from a CSO xlsx, case-insensitively."""
    xl = pd.ExcelFile(path)
    sheet = next((s for s in xl.sheet_names if s.strip().lower() == 'unpivoted'), xl.sheet_names[0])
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    return df

cso_file_a = os.path.join(RAW_DIR, 'agri_cso_output.xlsx')
cso_file_b = os.path.join(RAW_DIR, 'agri_cso_trade.xlsx')

df_a = read_unpivoted(cso_file_a) if os.path.exists(cso_file_a) else None
df_b = read_unpivoted(cso_file_b) if os.path.exists(cso_file_b) else None

def has_year_col(df):
    return df is not None and any(c.lower() == 'year' for c in df.columns)

def has_month_col(df):
    return df is not None and any(c.lower() == 'month' for c in df.columns)

# Assign correctly whichever way they landed
if has_year_col(df_a):
    df_output, df_trade = df_a, df_b
elif has_year_col(df_b):
    df_output, df_trade = df_b, df_a
    print('  NOTE: CSO files were swapped — detected and corrected automatically.')
else:
    df_output, df_trade = None, None
    print('  ERROR: Could not identify CSO output file (no Year column found in either file).')

# Process output file (annual, has Year column)
print('\n=== CSO AGRICULTURAL OUTPUT ===')
if df_output is not None:
    year_col = next(c for c in df_output.columns if c.lower() == 'year')
    df_output[year_col] = pd.to_numeric(df_output[year_col], errors='coerce')
    df_output['VALUE']  = pd.to_numeric(df_output['VALUE'],   errors='coerce')
    df_output = df_output[df_output[year_col] >= 2000].dropna(subset=[year_col, 'VALUE'])
    df_output = df_output.rename(columns={
        'Statistic Label': 'metric',
        year_col:          'year',
        'UNIT':            'unit',
        'VALUE':           'value',
        'State':           'state',
    })
    out = os.path.join(PROC_DIR, 'agri_cso_output.csv')
    df_output.to_csv(out, index=False)
    print(f'  Shape: {df_output.shape} | Years: {int(df_output["year"].min())}–{int(df_output["year"].max())}')
    print(f'  Metrics: {df_output["metric"].nunique()} | Total value: {df_output["value"].sum():,.0f}')
    print(f'✓ Saved → {out}')
else:
    print('  MISSING or unreadable.')

# Process trade file (monthly, has Month column)
print('\n=== CSO FOOD TRADE (MONTHLY) ===')
if df_trade is not None:
    month_col = next((c for c in df_trade.columns if c.lower() == 'month'), None)
    if month_col is None:
        print('  ERROR: No Month column found in trade file.')
    else:
        df_trade['VALUE'] = pd.to_numeric(df_trade['VALUE'], errors='coerce')
        df_trade['year']  = df_trade[month_col].astype(str).str[:4].pipe(
            lambda s: pd.to_numeric(s, errors='coerce'))
        df_trade = df_trade[df_trade['year'] >= 2010].dropna(subset=['year', 'VALUE'])
        df_trade = df_trade.rename(columns={
            'Statistic Label':  'flow',
            month_col:          'month',
            'Commodity Group':  'commodity',
            'UNIT':             'unit',
            'VALUE':            'value',
        })
        out = os.path.join(PROC_DIR, 'agri_cso_trade.csv')
        df_trade.to_csv(out, index=False)
        print(f'  Shape: {df_trade.shape} | Years: {int(df_trade["year"].min())}–{int(df_trade["year"].max())}')
        print(f'  Flows: {df_trade["flow"].unique().tolist()}')
        print(f'  Total value: €{df_trade["value"].sum():,.0f} thousand')
        print(f'✓ Saved → {out}')
else:
    print('  MISSING or unreadable.')

# ── 4. FAOSTAT Food Balances ──────────────────────────────────────────────────
print('\n=== FAOSTAT ===')
fao_path = os.path.join(RAW_DIR, 'faostat_agri.csv')
if os.path.exists(fao_path):
    df_fao = pd.read_csv(fao_path)
    df_fao.columns = [c.strip() for c in df_fao.columns]
    df_fao['Value'] = pd.to_numeric(df_fao['Value'], errors='coerce')
    df_fao = df_fao.rename(columns={
        'Domain':'domain','Area':'area','Element':'element',
        'Item':'item','Year':'year','Unit':'unit','Value':'value',
    })
    df_fao = df_fao[['domain','area','element','item','year','unit','value']].copy()
    df_fao = df_fao.dropna(subset=['year','value'])
    out = os.path.join(PROC_DIR, 'agri_faostat.csv')
    df_fao.to_csv(out, index=False)
    print(f'  Shape: {df_fao.shape} | Years: {int(df_fao["year"].min())}–{int(df_fao["year"].max())}')
    print(f'  Elements: {df_fao["element"].unique().tolist()}')
    print(f'  Items: {df_fao["item"].nunique()} unique food categories')
    print(f'✓ Saved → {out}')
else:
    print(f'  MISSING: {fao_path}')

print('\n=== DONE ===')
print('Outputs in:', PROC_DIR)
for f in sorted(os.listdir(PROC_DIR)):
    if f.startswith('agri'):
        size = os.path.getsize(os.path.join(PROC_DIR, f)) / 1024
        print(f'  {f} ({size:.0f} KB)')
