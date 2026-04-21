"""
Ireland SC-DSS -- Multi-Sector Supply Chain Decision Support System
Version: 4.0 -- Multi-sector (Energy . Agriculture . MedTech)
Sector-switching with sweep transitions, unique strategic overviews per sector,
fixed raw-data tab toggles (unique IDs per dashboard).
"""
import os, re, math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Chart tooltip definitions (fail-safe import — tooltips are optional)
try:
    from chart_tooltips import CHART_TOOLTIPS
except Exception:
    CHART_TOOLTIPS = {}

# ???????????????????????????????????????????????????????
# SECTOR THEME TOKENS
# ???????????????????????????????????????????????????????
THEMES = {
    "energy": {
        "bg":       "#0a0e14",
        "surface":  "#111820",
        "card":     "#161f2b",
        "border":   "#1e3048",
        "accent":   "#f5a623",
        "accent2":  "#e8c56a",
        "text":     "#e8eef5",
        "muted":    "#5a7a99",
        "sweep":    "#f5a623",
        "c1":"#f5a623","c2":"#e8c56a","c3":"#5b9bd5","c4":"#3a6694","c5":"#2a4a6e",
        "red":   "#f87171","green": "#4ade80","amber": "#fbbf24","orange":"#fb923c",
        "purple":"#a78bfa","teal":  "#2dd4bf",
    },
    "agri": {
        "bg":       "#071a0a",
        "surface":  "#0d2310",
        "card":     "#122b15",
        "border":   "#1c4220",
        "accent":   "#4caf50",
        "accent2":  "#8bc34a",
        "text":     "#e8f5e9",
        "muted":    "#4a7a54",
        "sweep":    "#4caf50",
        "c1":"#4caf50","c2":"#8bc34a","c3":"#66bb6a","c4":"#2e7d32","c5":"#a5d6a7",
        "red":   "#f87171","green": "#4ade80","amber": "#fbbf24","orange":"#fb923c",
        "purple":"#c084fc","teal":  "#2dd4bf",
    },
    "medtech": {
        "bg":       "#140d20",
        "surface":  "#1c1230",
        "card":     "#221638",
        "border":   "#342050",
        "accent":   "#b39ddb",
        "accent2":  "#e1bee7",
        "text":     "#f3eeff",
        "muted":    "#7060a0",
        "sweep":    "#9575cd",
        "c1":"#b39ddb","c2":"#e1bee7","c3":"#9575cd","c4":"#7c4dff","c5":"#ce93d8",
        "red":   "#f87171","green": "#4ade80","amber": "#fbbf24","orange":"#fb923c",
        "purple":"#c084fc","teal":  "#2dd4bf",
    },
}

# Current sector (module-level, set by callback via Store)
def T(sector, key):
    return THEMES.get(sector, THEMES["energy"]).get(key, "#888")

# Convenience: energy design tokens kept for backward compat
BG_PAGE  = THEMES["energy"]["bg"]
BG_CARD  = THEMES["energy"]["card"]
BG_INPUT = "#21262d"
BORDER   = THEMES["energy"]["border"]
BORDER_LT= "#3d444d"
TEXT_PRI = THEMES["energy"]["text"]
TEXT_SEC = "#8b949e"
TEXT_MUTED="#6e7681"
WHITE    = "#ffffff"
ACCENT   = THEMES["energy"]["accent"]
ACCENT2  = THEMES["energy"]["accent2"]
C_RED    = "#f87171"
C_GREEN  = "#4ade80"
C_AMBER  = "#fbbf24"
C_ORANGE = "#fb923c"
C_TEAL   = "#2dd4bf"
C_PURPLE = "#a78bfa"
C_BLUE   = "#60a5fa"

def hhi_colour(v):
    if v < 0.15: return C_GREEN
    if v < 0.25: return C_AMBER
    if v < 0.40: return C_ORANGE
    return C_RED

def hhi_label(v):
    if v < 0.15: return "Competitive"
    if v < 0.25: return "Moderate"
    if v < 0.40: return "High"
    return "Critical"

def stress_colour(v):
    if v > 70: return C_RED
    if v > 40: return C_ORANGE
    if v > 20: return C_AMBER
    return C_GREEN

def stress_label(v):
    if v > 70: return "CRITICAL"
    if v > 40: return "SEVERE"
    if v > 20: return "SIGNIFICANT"
    return "MINIMAL"

def fmt_val(v):
    if v >= 1e9:  return f"EUR {v/1e9:.2f}bn"
    if v >= 1e6:  return f"EUR {v/1e6:.0f}M"
    if v >= 1e3:  return f"EUR {v/1e3:.0f}K"
    return f"EUR {v:.0f}"

def hex_rgba(hex_col, alpha=0.45):
    h = hex_col.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


# ???????????????????????????????????????????????????????
# DATA LOADING
# ???????????????????????????????????????????????????????
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

def load_seai():
    path = os.path.join(PROC_DIR, "seai_energy_balance.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df["flow"] = df["flow"].str.strip()
    KEEP = ["Imports","Exports","Indigenous Production","Stock Change",
            "Primary Energy Supply (incl non-energy)",
            "Primary Energy Requirement (excl. non-energy)",
            "Total Final Energy Consumption","Available Final Energy Consumption",
            "Transformation Input","Transformation Output",
            "Public Thermal Power Plants (Input)","Public Thermal Power Plants (Output)",
            "Own Use and Distribution Losses","Transport","Residential","Industry*",
            "Commercial/Public Services*"]
    return df[df["flow"].isin(KEEP)].copy()

def load_comtrade():
    path = os.path.join(PROC_DIR, "comtrade_consolidated.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    dup_drops = [c for c in df.columns if re.search(r'\.\d+$', str(c))]
    if dup_drops: df = df.drop(columns=dup_drops)
    df.columns = [c.strip().lower() for c in df.columns]
    if "refyear" in df.columns: df["refyear"] = df["refyear"].astype(int)
    flow_col = None
    for candidate in ["flowcode","rgcode","flow_code","flow","flowdesc","rgdesc","flow_desc"]:
        if candidate in df.columns: flow_col = candidate; break
    if flow_col:
        raw = df[flow_col].astype(str).str.strip().str.lower()
        df["flowcode"] = raw.map(lambda v:
            "M" if v in ("m","1","import","imports","imp") else
            "X" if v in ("x","2","export","exports","exp","re-export","re-exports") else v.upper())
    else:
        df["flowcode"] = "M"
    if "primaryvalue" not in df.columns:
        for val_col in ["cifvalue","fobvalue","tradevalue","trade_value","value"]:
            if val_col in df.columns:
                df.rename(columns={val_col:"primaryvalue"}, inplace=True); break
    df["primaryvalue"] = pd.to_numeric(df.get("primaryvalue", pd.Series([0]*len(df))), errors="coerce").fillna(0)
    if "partnerdesc" not in df.columns: df["partnerdesc"] = "Unknown"
    if "cmddesc" not in df.columns: df["cmddesc"] = "Unknown"
    if "refyear" not in df.columns:
        for yr_col in ["year","period","ref_year"]:
            if yr_col in df.columns: df.rename(columns={yr_col:"refyear"}, inplace=True); break
    if "cmddesc" in df.columns:
        def map_product(v):
            v = str(v).lower()
            if "crude" in v: return "Crude petroleum oils"
            if any(x in v for x in ["refined","petroleum product","gasoil","gasoline","diesel","kerosene","fuel oil","naphtha"]): return "Refined petroleum products"
            if ("gas" in v and any(x in v for x in ["natural","lng","pipeline"])) or "lpg" in v or "petroleum gas" in v: return "Petroleum gas"
            if any(x in v for x in ["coal","coke","lignite"]): return "Coal"
            if "electri" in v: return "Electrical energy"
            return v.title()
        df["cmddesc"] = df["cmddesc"].map(map_product)
    CONTINENT_MAP = {
        "United Kingdom":"Europe","Germany":"Europe","France":"Europe","Netherlands":"Europe",
        "Belgium":"Europe","Norway":"Europe","Sweden":"Europe","Denmark":"Europe","Spain":"Europe",
        "Italy":"Europe","Poland":"Europe","Portugal":"Europe","Austria":"Europe",
        "Switzerland":"Europe","Finland":"Europe","Greece":"Europe",
        "United States":"Americas","Canada":"Americas","Mexico":"Americas","Brazil":"Americas",
        "Argentina":"Americas","Colombia":"Americas","Chile":"Americas","Trinidad and Tobago":"Americas",
        "Saudi Arabia":"Middle East","United Arab Emirates":"Middle East","Kuwait":"Middle East",
        "Iraq":"Middle East","Iran":"Middle East","Qatar":"Middle East","Oman":"Middle East",
        "Russia":"Europe/Asia","Kazakhstan":"Europe/Asia","Azerbaijan":"Europe/Asia",
        "China":"Asia","Japan":"Asia","Rep. of Korea":"Asia","India":"Asia",
        "Singapore":"Asia","Malaysia":"Asia","Indonesia":"Asia","Thailand":"Asia",
        "Nigeria":"Africa","Algeria":"Africa","Angola":"Africa","Libya":"Africa",
        "Egypt":"Africa","South Africa":"Africa","Australia":"Oceania","New Zealand":"Oceania",
    }
    COORDS = {
        "United Kingdom":(55.4,-3.4),"Netherlands":(52.1,5.3),"Belgium":(50.5,4.5),
        "Germany":(51.2,10.5),"France":(46.2,2.2),"Norway":(60.5,8.5),
        "Sweden":(63.0,16.0),"Denmark":(56.0,10.0),"Spain":(40.0,-4.0),
        "United States":(39.5,-98.4),"Canada":(56.1,-106.3),
        "Saudi Arabia":(23.9,45.1),"United Arab Emirates":(23.4,53.8),
        "Kuwait":(29.3,47.5),"Qatar":(25.4,51.2),"Algeria":(28.0,1.7),
        "Nigeria":(9.1,8.7),"Russia":(61.5,105.0),"Azerbaijan":(40.1,47.6),
        "China":(35.9,104.2),"Japan":(36.2,138.3),"Rep. of Korea":(35.9,127.8),
        "India":(20.6,78.9),"Singapore":(1.3,103.8),"Malaysia":(4.2,108.0),
        "Australia":(-25.3,133.8),"South Africa":(-30.6,22.9),
        "Brazil":(-14.2,-51.9),"Mexico":(23.6,-102.6),
    }
    df = df[[c for c in df.columns if not re.search(r'\.\d+$',str(c))]]
    drop_cols = [c for c in df.columns if c=="continent" or c.startswith("continent_")]
    if drop_cols: df = df.drop(columns=drop_cols)
    if "partnerdesc" in df.columns:
        df = df.copy()
        df["continent"]   = df["partnerdesc"].astype(str).apply(lambda x: CONTINENT_MAP.get(x,"Other"))
        df["partner_lat"] = df["partnerdesc"].astype(str).apply(lambda x: COORDS.get(x,(0,0))[0])
        df["partner_lon"] = df["partnerdesc"].astype(str).apply(lambda x: COORDS.get(x,(0,0))[1])
    JUNK = {"world","other","areas, nes","total","unspecified","not specified","european union",
            "eu","oecd","opec","asean","free zones","special categories","bunkers","low value trade"}
    if "partnerdesc" in df.columns and "flowcode" in df.columns:
        imp_mask = (df["flowcode"]=="M")&(~df["partnerdesc"].astype(str).str.strip().str.lower().isin(JUNK))
        exp_mask = (df["flowcode"]=="X")
        df = df[imp_mask|exp_mask].copy()
    return df

def load_agri_comtrade():
    path = os.path.join(PROC_DIR, "agri_comtrade_consolidated.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["refyear"] = pd.to_numeric(df["refyear"], errors="coerce").fillna(0).astype(int)
    df["primaryvalue"] = pd.to_numeric(df["primaryvalue"], errors="coerce").fillna(0)
    return df

def load_agri_cso_output():
    path = os.path.join(PROC_DIR, "agri_cso_output.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    return df

def load_agri_cso_trade():
    path = os.path.join(PROC_DIR, "agri_cso_trade.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    return df

def load_faostat():
    path = os.path.join(PROC_DIR, "agri_faostat.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    return df

def load_medtech_comtrade():
    path = os.path.join(PROC_DIR, "medtech_comtrade_consolidated.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["refyear"] = pd.to_numeric(df["refyear"], errors="coerce").fillna(0).astype(int)
    df["primaryvalue"] = pd.to_numeric(df["primaryvalue"], errors="coerce").fillna(0)
    return df

def load_medtech_cso():
    path = os.path.join(PROC_DIR, "medtech_cso_trade.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    return df

def load_medtech_absei():
    path = os.path.join(PROC_DIR, "medtech_absei.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df["table"] = df["table"].fillna("Table 1: Total Sales")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    return df

def load_ema_shortages():
    path = os.path.join(PROC_DIR, "medtech_ema_shortages.csv")
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path)

def load_ema_critical():
    path = os.path.join(PROC_DIR, "medtech_ema_critical.csv")
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path)

# ?? Load all datasets ??????????????????????????????????????????????????????????
print("[SC-DSS] Loading all datasets...")
seai        = load_seai()
comtrade    = load_comtrade()
agri_ct     = load_agri_comtrade()
agri_out    = load_agri_cso_output()
agri_trade  = load_agri_cso_trade()
faostat     = load_faostat()
med_ct      = load_medtech_comtrade()
med_cso     = load_medtech_cso()
med_absei   = load_medtech_absei()
ema_short   = load_ema_shortages()
ema_crit    = load_ema_critical()
print("[SC-DSS] All datasets loaded.")

# ?? Energy helpers ?????????????????????????????????????????????????????????????
S_IMP = seai[seai["flow"]=="Imports"].sort_values("year") if not seai.empty else pd.DataFrame()
S_EXP = seai[seai["flow"]=="Exports"].sort_values("year") if not seai.empty else pd.DataFrame()
S_IND = seai[seai["flow"]=="Indigenous Production"].sort_values("year") if not seai.empty else pd.DataFrame()
S_PER = seai[seai["flow"]=="Primary Energy Requirement (excl. non-energy)"].sort_values("year") if not seai.empty else pd.DataFrame()

YEARS    = sorted(seai["year"].dropna().unique().astype(int).tolist()) if not seai.empty else [2023]
YEAR_MIN = min(YEARS); YEAR_MAX = max(YEARS)
# Fuel categories for Energy HHI & concentration metrics.
# Added coal and renewables (Fix #15) so KPIs reflect full import mix rather
# than only 94% of it.
IMP_FUELS = ["oil","natural_gas","electricity","coal","renewables"]

# ── Fuel → HS code families for country-HHI computation ──────────────────────
# Maps each fuel in IMP_FUELS to HS2/HS4 chapters that represent it in Comtrade.
# This lets us compute supplier concentration (true HHI) per fuel from
# country-level import data, not from the fuel-mix shares (Fix #2-B).
FUEL_HS_PATTERNS = {
    "oil":          ["2709","2710","2711 19","petroleum","crude","gasoil","kerosene","jet","gasoline","fuel oil","lpg","lubric","bitumen"],
    "natural_gas":  ["2711 11","2711 21","natural gas","lng"],
    "coal":         ["2701","2702","2703","2704","coal","coke","lignite","anthracite","peat"],
    "electricity":  ["2716","electrical","electricity","electric energy"],
    "renewables":   ["biomass","biofuel","biodiesel","bioethanol","wood pellet","renewable"],
}

def _matches_fuel(cmddesc, fuel):
    """Return True if a Comtrade cmddesc row looks like the given fuel category."""
    if pd.isna(cmddesc): return False
    v = str(cmddesc).lower()
    return any(pat.lower() in v for pat in FUEL_HS_PATTERNS.get(fuel, []))

CT_YEARS    = sorted(comtrade["refyear"].unique().astype(int).tolist()) if not comtrade.empty else []
CT_PARTNERS = sorted(comtrade["partnerdesc"].dropna().unique().tolist()) if not comtrade.empty else []
CT_YEAR_MAX = max(CT_YEARS) if CT_YEARS else YEAR_MAX
DEFAULT_YEAR = 2023 if 2023 in YEARS else min(YEAR_MAX, CT_YEAR_MAX)
SEAI_DROPDOWN_YEARS = sorted([y for y in YEARS if y >= YEAR_MAX-14], reverse=True)
BOTH_DROPDOWN_YEARS = sorted([y for y in CT_YEARS if y >= CT_YEAR_MAX-14 and y in YEARS], reverse=True) or SEAI_DROPDOWN_YEARS

# ── World Bank Worldwide Governance Indicators (Fix #3-B) ─────────────────────
# "Political Stability and Absence of Violence" percentile rank (2023 release),
# rescaled to a 0-1 risk score where HIGHER = MORE RISKY.
# Rule: risk = 1 - (percentile_rank / 100).
# Source: World Bank WGI 2024 release (data year 2023). Countries not listed
# default to 0.5 (median risk, no judgement either way).
WGI_RISK = {
    # Very low risk (stable democracies)
    "Norway": 0.08, "Switzerland": 0.09, "Iceland": 0.06, "Finland": 0.10,
    "Luxembourg": 0.10, "Denmark": 0.12, "New Zealand": 0.12, "Singapore": 0.14,
    "Austria": 0.18, "Sweden": 0.18, "Netherlands": 0.19, "Ireland": 0.19,
    "Canada": 0.22, "Australia": 0.22, "Portugal": 0.22, "Germany": 0.24,
    "Belgium": 0.26, "Japan": 0.25, "United Kingdom": 0.32, "France": 0.38,
    "United States": 0.42, "Czechia": 0.20, "Czech Republic": 0.20,
    "Slovenia": 0.22, "Slovakia": 0.28, "Estonia": 0.20, "Latvia": 0.28,
    "Lithuania": 0.25, "Spain": 0.32, "Italy": 0.35, "Poland": 0.28,
    "Hungary": 0.38, "Greece": 0.40, "Malta": 0.20, "Cyprus": 0.35,
    "Republic of Korea": 0.32, "Rep. of Korea": 0.32, "Taiwan": 0.35,
    "United Arab Emirates": 0.30, "Qatar": 0.28, "Kuwait": 0.45,
    "Hong Kong": 0.35, "Bahrain": 0.48, "Oman": 0.40,
    # Medium risk
    "Croatia": 0.38, "Bulgaria": 0.45, "Romania": 0.42, "Brazil": 0.55,
    "Mexico": 0.68, "Argentina": 0.55, "Chile": 0.35, "Uruguay": 0.25,
    "Morocco": 0.55, "South Africa": 0.62, "Indonesia": 0.58, "Malaysia": 0.45,
    "Thailand": 0.60, "Vietnam": 0.45, "Philippines": 0.70, "India": 0.68,
    "China": 0.55, "Saudi Arabia": 0.50, "Israel": 0.78, "Türkiye": 0.72,
    "Turkey": 0.72, "Azerbaijan": 0.62, "Georgia": 0.62, "Kazakhstan": 0.52,
    "Algeria": 0.65, "Egypt": 0.75, "Tunisia": 0.60,
    "Dominican Republic": 0.55, "Costa Rica": 0.35, "Panama": 0.50,
    "Peru": 0.62, "Colombia": 0.72, "Jordan": 0.68, "Lebanon": 0.85,
    # High risk
    "Russian Federation": 0.90, "Ukraine": 0.90, "Belarus": 0.80, "Iran": 0.85,
    "Iraq": 0.92, "Syria": 0.98, "Afghanistan": 0.98, "Yemen": 0.97,
    "Libya": 0.95, "Nigeria": 0.85, "Pakistan": 0.88, "Venezuela": 0.90,
    "Myanmar": 0.95, "Sudan": 0.96, "Somalia": 0.98, "Mali": 0.90,
    "Haiti": 0.95, "North Korea": 0.95, "Dem. People's Rep. of Korea": 0.95,
}

def gov_risk_score(country):
    """Return governance risk score (0-1, higher=riskier) for a country.
    Uses World Bank Worldwide Governance Indicators (Political Stability).
    Unknown countries default to 0.5 (median)."""
    if pd.isna(country): return 0.5
    return WGI_RISK.get(str(country).strip(), 0.5)

# ?? Agri helpers ???????????????????????????????????????????????????????????????
AGRI_CT_YEARS = sorted(agri_ct["refyear"].unique().astype(int).tolist()) if not agri_ct.empty else []
AGRI_CT_YMAX  = max(AGRI_CT_YEARS) if AGRI_CT_YEARS else 2024
AGRI_DD_YEARS = sorted([y for y in AGRI_CT_YEARS if y >= AGRI_CT_YMAX-14], reverse=True) or [2024]

# ?? MedTech helpers ????????????????????????????????????????????????????????????
MED_CT_YEARS = sorted(med_ct["refyear"].unique().astype(int).tolist()) if not med_ct.empty else []
MED_CT_YMAX  = max(MED_CT_YEARS) if MED_CT_YEARS else 2024
MED_DD_YEARS = sorted([y for y in MED_CT_YEARS if y >= MED_CT_YMAX-14], reverse=True) or [2024]

def ct_nearest(year, ct_years=None):
    yrs = ct_years or CT_YEARS
    if not yrs: return year
    year = int(year)
    if year in yrs: return year
    below = [y for y in yrs if y <= year]
    return max(below) if below else max(yrs)

def safe_val(r, col):
    v = r.get(col, 0)
    if v is None: return 0.0
    try:
        f = float(v)
        return 0.0 if np.isnan(f) else f
    except: return 0.0

def resolve_year(val, fallback=None):
    fb = fallback or DEFAULT_YEAR
    if val is None: return fb
    if isinstance(val, (list,tuple)):
        clean = [int(y) for y in val if y is not None]
        return max(clean) if clean else fb
    try: return int(val)
    except: return fb

def resolve_years(val, fallback=None):
    fb = fallback or [DEFAULT_YEAR]
    if val is None: return fb
    if isinstance(val, (list,tuple)):
        clean = [int(y) for y in val if y is not None]
        return sorted(clean) if clean else fb
    try: return [int(val)]
    except: return fb

def compute_hhi(year):
    """Compute TRUE supplier-concentration HHI per fuel (Fix #2-B).

    Real HHI = Σ(country_share_of_imports)² across supplier countries
    for each fuel type. Values near 0 = fragmented supply (many countries);
    values near 1 = concentrated supply (one dominant country).

    DoJ/FTC interpretation:
        HHI < 0.15  competitive
        0.15 - 0.25 moderately concentrated
        > 0.25       concentrated (risk)
        > 0.40       highly concentrated (critical)

    Returns dict with:
      'by_fuel': {fuel: hhi_value}  — per-fuel country-HHI
      'weighted': aggregate HHI weighted by SEAI fuel-import volume
      'shares':   {fuel: share_of_total_imports} — for context / mix charts
      'hhi':      the weighted aggregate (for backward compatibility)
    """
    year = int(year)
    row = S_IMP[S_IMP["year"]==year] if not S_IMP.empty else pd.DataFrame()
    shares = {}
    if not row.empty:
        r = row.iloc[0]
        total = float(r["total"]) if pd.notna(r.get("total",0)) and r.get("total",0)>0 else 1.0
        shares = {f: safe_val(r,f)/total for f in IMP_FUELS}

    # Compute true country-concentration HHI for each fuel using Comtrade
    by_fuel = {}
    if not comtrade.empty and CT_YEARS:
        ct_yr = ct_nearest(year)
        ct_imp = comtrade[(comtrade["refyear"]==ct_yr)&(comtrade["flowcode"]=="M")]
        for fuel in IMP_FUELS:
            fuel_rows = ct_imp[ct_imp["cmddesc"].apply(lambda v: _matches_fuel(v, fuel))]
            if fuel_rows.empty:
                by_fuel[fuel] = 0.0
                continue
            by_country = fuel_rows.groupby("partnerdesc")["primaryvalue"].sum()
            tot = by_country.sum()
            if tot <= 0:
                by_fuel[fuel] = 0.0
                continue
            by_fuel[fuel] = round(float(((by_country / tot) ** 2).sum()), 4)

    # Weight each fuel's country-HHI by its share of total energy imports,
    # so fuels that are a tiny share of our imports don't dominate the aggregate.
    weighted = 0.0
    if shares and by_fuel:
        weighted = sum(shares.get(f, 0) * by_fuel.get(f, 0) for f in IMP_FUELS)
        weighted = round(weighted, 4)

    return {
        "hhi": weighted,
        "by_fuel": by_fuel,
        "shares": shares,
        "weighted": weighted,
    }

def compute_dependency(year):
    """Country-level energy dependency score (Fix #3-B).

    Weighted composite:
       dependency = 0.4 * import_share
                  + 0.3 * dominance (1 if >30% of imports, else 0)
                  + 0.3 * gov_risk  (from World Bank WGI, 0-1 scale)

    Higher = more concentrated + riskier country = higher dependency risk.
    """
    if comtrade.empty or not CT_YEARS: return pd.DataFrame()
    year = int(year)
    ct_year = ct_nearest(year)
    df = comtrade[(comtrade["refyear"]==ct_year)&(comtrade["flowcode"]=="M")]
    if df.empty: return pd.DataFrame()
    total = df["primaryvalue"].sum() or 1
    g = df.groupby("partnerdesc")["primaryvalue"].sum().reset_index()
    g.columns = ["country","value"]
    g["import_share"] = g["value"]/total
    g["dominance"]    = (g["import_share"]>0.30).astype(float)
    g["gov_risk"]     = g["country"].apply(gov_risk_score)
    g["dependency"]   = 0.4*g["import_share"]+0.3*g["dominance"]+0.3*g["gov_risk"]
    return g.sort_values("dependency", ascending=False)

def kri_for_year(year):
    year = resolve_year(year)
    row_imp = S_IMP[S_IMP["year"]==year] if not S_IMP.empty else pd.DataFrame()
    row_per = S_PER[S_PER["year"]==year] if not S_PER.empty else pd.DataFrame()
    row_ind = S_IND[S_IND["year"]==year] if not S_IND.empty else pd.DataFrame()
    if row_imp.empty: return {}
    r = row_imp.iloc[0]
    total = float(r["total"]) if pd.notna(r.get("total",0)) and r.get("total",0)>0 else 1.0
    oil_dep = round(safe_val(r,"oil")/total*100,1)
    fossil  = safe_val(r,"oil")+safe_val(r,"natural_gas")+safe_val(r,"coal") if "coal" in r else safe_val(r,"oil")+safe_val(r,"natural_gas")
    self_suff = 0.0
    if not row_per.empty and not row_ind.empty:
        pv = float(row_per.iloc[0].get("total",1)) if pd.notna(row_per.iloc[0].get("total",1)) else 1.0
        iv = float(row_ind.iloc[0].get("total",0)) if pd.notna(row_ind.iloc[0].get("total",0)) else 0.0
        self_suff = round(max(iv,0)/max(pv,1)*100,1)
    hhi_data = compute_hhi(year)
    n_sup = 0; top_sup = "N/A"; top_sh = 0.0
    ct_imp_val = 0.0; ct_exp_val = 0.0
    if not comtrade.empty and CT_YEARS:
        ct_yr = ct_nearest(year)
        ct_y  = comtrade[comtrade["refyear"]==ct_yr]
        ct_i  = ct_y[ct_y["flowcode"]=="M"]
        ct_x  = ct_y[ct_y["flowcode"]=="X"]
        if not ct_i.empty:
            n_sup = ct_i["partnerdesc"].nunique()
            g     = ct_i.groupby("partnerdesc")["primaryvalue"].sum()
            top_sup = g.idxmax()
            top_sh  = round(g.max()/g.sum()*100,1)
            ct_imp_val = ct_i["primaryvalue"].sum()
        if not ct_x.empty:
            ct_exp_val = ct_x["primaryvalue"].sum()
    total_exp_ktoe = 0.0
    if not S_EXP.empty:
        re = S_EXP[S_EXP["year"]==year]
        if not re.empty: total_exp_ktoe = float(re.iloc[0].get("total",0))
    return {
        "oil_dep":oil_dep,"fossil_pct":round(fossil/total*100,1),"self_suff":self_suff,
        "hhi":hhi_data["hhi"],"n_suppliers":n_sup,"top_supplier":top_sup,"top_share":top_sh,
        "total_imports_ktoe":round(total,0),"total_exports_ktoe":total_exp_ktoe,
        "ct_imp_val":ct_imp_val,"ct_exp_val":ct_exp_val,"ct_year_used":ct_nearest(year),
    }


# ???????????????????????????????????????????????????????
# THEME-AWARE PLOTLY LAYOUT + UI COMPONENTS
# ???????????????????????????????????????????????????????
def style_horiz_bar(fig, sector="energy"):
    """Apply consistent styling to horizontal bar charts."""
    th = THEMES.get(sector, THEMES["energy"])
    # Make text font smaller so it fits
    for trace in fig.data:
        if hasattr(trace, 'orientation') and trace.orientation == 'h':
            if hasattr(trace, 'textfont') and trace.textfont:
                trace.textfont.size = 10
            if hasattr(trace, 'textposition'):
                trace.textposition = 'outside'
    # Extend x range to give labels room - find current max
    max_x = 0
    for trace in fig.data:
        if hasattr(trace, 'x') and trace.x is not None:
            try:
                vals = [v for v in trace.x if v is not None and isinstance(v, (int,float))]
                if vals: max_x = max(max_x, max(vals))
            except: pass
    if max_x > 0:
        fig.update_xaxes(range=[0, max_x * 1.35])
    return fig

def themed_layout(fig, sector="energy", legend=True):
    th = THEMES.get(sector, THEMES["energy"])

    # Auto-detect horizontal bar chart
    is_horiz = any(
        getattr(t,"type","")=="bar" and getattr(t,"orientation","")=="h"
        for t in fig.data
    )
    r_margin = 175 if is_horiz else 25

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=th["bg"],
        font=dict(family="Inter,Segoe UI,Arial",size=12,color=th["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=th["border"],borderwidth=1,
                    font=dict(size=11,color=th["muted"])) if legend else dict(visible=False),
        margin=dict(l=55,r=r_margin,t=40,b=70),

        xaxis=dict(
            gridcolor=th["border"],zerolinecolor=th["border"],
            tickcolor=th["muted"],tickfont=dict(size=11,color=th["muted"]),
            ticklen=6,tickwidth=1,automargin=True,
        ),
        yaxis=dict(
            gridcolor=th["border"],zerolinecolor=th["border"],
            tickcolor=th["muted"],tickfont=dict(size=11,color=th["muted"]),
            ticklen=6,tickwidth=1,automargin=True,
        ),
    )
    # Auto-extend axis range so "outside" text labels never clip
    is_vert_bar = any(
        getattr(t,"type","")=="bar" and getattr(t,"orientation","") in ("","v")
        for t in fig.data
    )

    if is_horiz:
        max_x = 0
        for t in fig.data:
            try:
                vals = [v for v in (t.x or []) if isinstance(v,(int,float)) and v==v]
                if vals: max_x = max(max_x, max(vals))
            except: pass
        if max_x > 0:
            fig.update_xaxes(range=[0, max_x * 1.55])
        for t in fig.data:
            if getattr(t,"type","")=="bar" and getattr(t,"orientation","")=="h":
                try:
                    if t.textfont: t.textfont.update(size=10)
                    t.update(cliponaxis=False)
                except: pass
        fig.update_layout(margin=dict(r=175))

    if is_vert_bar and not is_horiz:
        all_y_vals = []
        for t in fig.data:
            try:
                vals = [v for v in (t.y or []) if isinstance(v,(int,float)) and v==v]
                all_y_vals.extend(vals)
            except: pass
        if all_y_vals:
            max_y = max(all_y_vals)
            min_y = min(all_y_vals)
            # Extend only 15% since labels are inside bars now
            top = max_y * 1.15 if max_y > 0 else max_y * 0.85
            bot = min_y * 1.15 if min_y < 0 else min(0, min_y)
            fig.update_yaxes(range=[bot, top])
        # Disable clip so labels above bars are never cut off
        for t in fig.data:
            if getattr(t,"type","")=="bar":
                try: t.update(cliponaxis=False)
                except: pass
        fig.update_layout(margin=dict(t=20))

    return fig

# Keep backward-compatible alias used by existing energy callbacks
def dark_layout(fig, legend=True):
    return themed_layout(fig, "energy", legend)

def sankey_layout(fig, sector="energy"):
    """Minimal layout for Sankey charts - no axes, just colours and font."""
    th = THEMES.get(sector, THEMES["energy"])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,Segoe UI,Arial", size=12, color=th["text"]),
        margin=dict(l=20, r=20, t=10, b=20),
    )
    return fig

def kri_card(title, value, unit="", colour=None, subtitle="", sector="energy"):
    th = THEMES.get(sector, THEMES["energy"])
    col = colour or th["accent"]
    return html.Div([
        html.Div(title.upper(), style={"fontSize":"10px","fontWeight":"500",
            "color":th["muted"],"letterSpacing":"0.8px","marginBottom":"8px"}),
        html.Div([
            html.Span(value, style={"fontSize":"28px","fontWeight":"700","color":col,"lineHeight":"1"}),
            html.Span(f" {unit}" if unit else "", style={"fontSize":"14px","color":col,"marginLeft":"2px"}),
        ]),
        html.Div(subtitle, style={"fontSize":"11px","color":col,"marginTop":"6px","opacity":"0.85"}) if subtitle else None,
    ], style={"background":th["card"],"border":f"1px solid {th['border']}","borderRadius":"8px",
              "padding":"16px 18px","flex":"1","borderTop":f"2px solid {col}"})

def chart_info_icon(info_id):
    """Render a ? icon with hover tooltip if info_id matches a CHART_TOOLTIPS entry."""
    if not info_id:
        return None
    t = CHART_TOOLTIPS.get(info_id)
    if not t:
        return None
    # Build the tooltip content
    parts = []
    parts.append(html.Div(t.get("title", info_id), className="chart-tooltip-title"))
    body = t.get("body")
    if body:
        parts.append(html.Div(body, className="chart-tooltip-body"))
    legend = t.get("legend")
    if legend:
        legend_children = []
        for label, colour, meaning in legend:
            legend_children.append(html.Span(label, className="l-label"))
            legend_children.append(html.Span(meaning, className=f"l-{colour}"))
        parts.append(html.Div(legend_children, className="chart-tooltip-legend"))
    source = t.get("source")
    if source:
        parts.append(html.Div(source, className="chart-tooltip-footer"))
    return html.Span([
        "?",
        html.Span(parts, className="chart-tooltip"),
    ], className="chart-info-icon", tabIndex=0)


def dark_card(title, children, badge=None, flex=None, sector="energy", info_id=None):
    th = THEMES.get(sector, THEMES["energy"])
    style = {"background":th["card"],"border":f"1px solid {th['border']}","borderRadius":"8px","padding":"16px"}
    if flex: style["flex"] = flex
    header_children = [
        html.Span(title.upper(), style={"fontSize":"10px","color":th["muted"],
            "fontWeight":"500","letterSpacing":"0.8px"}),
    ]
    if badge:
        header_children.append(html.Span(badge, style={"background":BG_INPUT,
            "border":f"1px solid {th['border']}","borderRadius":"4px",
            "padding":"2px 8px","fontSize":"10px","color":th["muted"],"marginLeft":"8px"}))
    icon = chart_info_icon(info_id)
    if icon is not None:
        header_children.append(icon)
    return html.Div([
        html.Div(header_children, style={"marginBottom":"12px","display":"flex","alignItems":"center"}),
        *children,
    ], style=style)

def dark_graph(graph_id, height=280):
    return dcc.Graph(id=graph_id, style={"height":f"{height}px"},
                     config={"displayModeBar":False,"responsive":True})

def yr_dropdown(fid, year_list=None, multi=True, default=None):
    years = year_list or BOTH_DROPDOWN_YEARS
    def_yr = default if default is not None else (2023 if 2023 in (year_list or BOTH_DROPDOWN_YEARS) else max(year_list or BOTH_DROPDOWN_YEARS))
    if def_yr not in years:
        def_yr = min(years, key=lambda y: abs(y-def_yr)) if years else DEFAULT_YEAR
    options = [{"label":str(y),"value":y} for y in years]
    val = [def_yr] if multi else def_yr
    return html.Div([
        html.Label("Year",
                   style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
        dcc.Dropdown(id=fid, options=options, value=val, clearable=False, multi=multi,
                     style={"fontSize":"13px","width":"220px","backgroundColor":"#ffffff",
                            "color":"#111111","border":f"1px solid {BORDER}","borderRadius":"6px"},
                     className="dark-dropdown"),
    ])

def yr_range_slider(fid, start=2010):
    return html.Div([
        html.Label("Year Range", style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
        dcc.RangeSlider(id=fid, min=start, max=YEAR_MAX, step=1, value=[start,YEAR_MAX],
                        marks={y:{"label":str(y),"style":{"color":TEXT_MUTED,"fontSize":"10px"}}
                               for y in range(start,YEAR_MAX+1,2)},
                        tooltip={"placement":"bottom","always_visible":False}),
    ], style={"flex":"1","minWidth":"300px"})

def yr_dropdown_generic(fid, year_list, multi=True, default=None):
    years = year_list or [2024]
    def_yr = default if default is not None else (max(years) if years else 2024)
    if def_yr not in years:
        def_yr = min(years, key=lambda y: abs(y-def_yr)) if years else years[0]
    options = [{"label":str(y),"value":y} for y in years]
    val = [def_yr] if multi else def_yr
    return html.Div([
        html.Label("Year",
                   style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
        dcc.Dropdown(id=fid, options=options, value=val, clearable=False, multi=multi,
                     style={"fontSize":"13px","width":"220px","backgroundColor":"#ffffff",
                            "color":"#111111","border":f"1px solid {BORDER}","borderRadius":"6px"}),
    ])

def row(*items, gap="16px"):
    return html.Div(list(items), style={"display":"flex","gap":gap,"marginBottom":"16px"})

def risk_badge(label, colour):
    return html.Span(label, style={
        "background":hex_rgba(colour,0.15),"color":colour,
        "border":f"1px solid {hex_rgba(colour,0.3)}",
        "borderRadius":"4px","padding":"2px 8px","fontSize":"10px","fontWeight":"600"})

def insight_item(severity, colour, title, msg):
    return html.Div([
        html.Div([risk_badge(severity,colour),
                  html.Span(f"  {title}",style={"fontWeight":"600","fontSize":"12px",
                              "marginLeft":"8px","color":TEXT_PRI})],
                 style={"display":"flex","alignItems":"center","marginBottom":"3px"}),
        html.P(msg, style={"margin":"0 0 0 4px","fontSize":"11px","color":TEXT_SEC}),
    ], style={"padding":"10px 12px","background":BG_INPUT,
              "borderLeft":f"2px solid {colour}","borderRadius":"0 6px 6px 0","marginBottom":"8px"})

def raw_data_section(dash_id, tables):
    """
    Collapsible raw data panel with correctly unique tab IDs per dashboard.
    tables = list of (tab_label, dataframe_or_none, columns_list)
    dash_id = unique string like 'd1', 'd2', etc.
    """
    tab_id = f"{dash_id}-raw-source-tabs"
    btn_id = f"{dash_id}-raw-toggle-btn"
    panel_id = f"{dash_id}-raw-panel"
    return html.Div([
        html.Button(
            "?  Show Raw Data",
            id=btn_id,
            style={"background":"transparent","border":f"1px solid {BORDER}","color":TEXT_MUTED,
                   "fontSize":"11px","padding":"6px 14px","borderRadius":"6px","cursor":"pointer",
                   "marginTop":"12px","marginBottom":"0"},
            n_clicks=0,
        ),
        html.Div(
            id=panel_id,
            children=[
                dcc.Tabs(id=tab_id, value=tables[0][0] if tables else "tab0",
                    children=[
                        dcc.Tab(label=lbl, value=lbl,
                            style={"backgroundColor":BG_INPUT,"color":TEXT_MUTED,"border":f"1px solid {BORDER}","fontSize":"12px","padding":"6px 14px"},
                            selected_style={"backgroundColor":BG_CARD,"color":TEXT_PRI,"border":f"1px solid {BORDER}","borderBottom":f"2px solid {ACCENT}","fontSize":"12px","padding":"6px 14px"},
                        ) for lbl,_,_ in tables
                    ]
                ),
                html.Div(id=f"{dash_id}-raw-content",
                         style={"marginTop":"8px","overflowX":"auto","maxHeight":"320px","overflowY":"auto"}),
            ],
            style={"display":"none","marginTop":"10px"},
        ),
    ])


# ???????????????????????????????????????????????????????
# APP INIT + LAYOUT
# ???????????????????????????????????????????????????????
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True,
           meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}])
app.title = "Ireland SC-DSS"
server = app.server   # required for gunicorn deployment

app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  .Select-value-label,.Select-placeholder,.Select-input input,
  .VirtualizedSelectOption,.VirtualizedSelectFocusedOption,
  .Select-menu-outer,.Select-option,
  .Select--multi .Select-value,.Select--multi .Select-value-label{color:#111111!important}
  .Select-control,.Select-menu-outer{background-color:#ffffff!important}
  /* Dash 4.x dropdown selectors */
  .dash-dropdown .Select-control{background-color:#ffffff!important}
  .dash-dropdown .Select-value-label{color:#111111!important}
  .dash-dropdown input{color:#111111!important}
  .VirtualizedSelectOption{color:#111111!important;background-color:#ffffff!important}
  .VirtualizedSelectFocusedOption{color:#111111!important;background-color:#e8e8e8!important}
  /* React-Select (newer Dash) */
  .Select__control{background-color:#ffffff!important}
  .Select__single-value{color:#111111!important}
  .Select__placeholder{color:#555555!important}
  .Select__menu{background-color:#ffffff!important}
  .Select__option{color:#111111!important;background-color:#ffffff!important}
  .Select__option--is-focused{background-color:#e8f0fe!important}
  .Select__input input{color:#111111!important}
  .xtick text,.ytick text{font-size:11px}

  /* ?? Sector sweep transition ?? */
  #sector-sweep{
    position:fixed;top:0;left:0;width:100vw;height:100vh;
    pointer-events:none;z-index:9999;
    transform:translateX(-105%);
  }
  #sector-sweep.sweep-go{
    animation:sweepAnim 0.65s cubic-bezier(.7,0,.3,1) forwards;
  }
  @keyframes sweepAnim{
    0%{transform:translateX(-105%)}
    45%{transform:translateX(0%)}
    55%{transform:translateX(0%)}
    100%{transform:translateX(105%)}
  }

  /* Dashboard tab transitions -- veil wipe */
  #page-veil{
    position:fixed;top:0;left:0;width:100vw;height:100vh;
    z-index:8000;pointer-events:none;
    background:linear-gradient(135deg,var(--veil-a,#0a0e14) 0%,var(--veil-b,#111820) 100%);
    transform:translateX(-100%);
  }
  #page-veil.veil-in{
    animation:veilIn 0.22s cubic-bezier(.7,0,.3,1) forwards;
  }
  #page-veil.veil-out{
    animation:veilOut 0.28s cubic-bezier(.4,0,.2,1) forwards;
  }
  @keyframes veilIn{
    from{transform:translateX(-100%)}
    to{transform:translateX(0%)}
  }
  @keyframes veilOut{
    from{transform:translateX(0%)}
    to{transform:translateX(100%)}
  }
  /* Content fade-up after veil lifts */
  @keyframes contentReveal{
    from{opacity:0;transform:translateY(8px)}
    to{opacity:1;transform:translateY(0)}
  }
  #page-content > div{
    animation:contentReveal 0.3s cubic-bezier(.4,0,.2,1) 0.05s both;
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar{width:6px;height:6px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}

  /* Sector pill button active states */
  .sector-btn-energy.active{background:#f5a623!important;color:#0a0e14!important}
  .sector-btn-agri.active{background:#4caf50!important;color:#071a0a!important}
  .sector-btn-medtech.active{background:#b39ddb!important;color:#140d20!important}

  /* ===== Chart info icon (?) + hover tooltip ===== */
  .chart-info-icon{
    display:inline-flex;align-items:center;justify-content:center;
    width:18px;height:18px;border-radius:50%;
    background:rgba(139,148,158,0.12);border:1px solid #3a4452;
    color:#8b949e;font-size:11px;font-weight:600;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    cursor:help;transition:all 0.18s;position:relative;user-select:none;
    margin-left:4px;
  }
  .chart-info-icon:hover{
    background:rgba(245,166,35,0.18);border-color:#f5a623;color:#f5a623;
  }
  .chart-tooltip{
    position:absolute;top:calc(100% + 10px);left:-12px;
    background:#0d1420;border:1px solid #2d3748;border-radius:8px;
    padding:14px 16px;width:340px;
    font-size:12.5px;font-weight:400;line-height:1.5;
    color:#d1d5db;letter-spacing:0.1px;text-transform:none;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    box-shadow:0 10px 30px rgba(0,0,0,0.7);
    opacity:0;visibility:hidden;transition:opacity 0.18s,visibility 0.18s;
    z-index:1000;pointer-events:none;text-align:left;
  }
  .chart-info-icon:hover .chart-tooltip,
  .chart-info-icon:focus .chart-tooltip{opacity:1;visibility:visible}
  .chart-tooltip-title{
    font-size:14px;font-weight:600;color:#ffffff;
    margin:0 0 10px 0;padding-bottom:10px;
    border-bottom:1px solid #2d3748;
  }
  .chart-tooltip-body{
    font-size:12.5px;color:#c9d1d9;margin:0 0 8px 0;line-height:1.55;
  }
  .chart-tooltip-legend{
    font-size:11.5px;color:#8b949e;line-height:1.7;
    margin:10px 0 0 0;display:grid;grid-template-columns:1fr auto;gap:2px 16px;
  }
  .chart-tooltip-legend .l-label{
    color:#8b949e;font-family:"SF Mono",Menlo,Consolas,monospace;font-size:11px;
  }
  .chart-tooltip-legend .l-green { color:#4caf50;font-weight:600 }
  .chart-tooltip-legend .l-amber { color:#f5a623;font-weight:600 }
  .chart-tooltip-legend .l-orange{ color:#ff7043;font-weight:600 }
  .chart-tooltip-legend .l-red   { color:#e53935;font-weight:600 }
  .chart-tooltip-legend .l-blue  { color:#4fc3f7;font-weight:600 }
  .chart-tooltip-legend .l-grey  { color:#8b949e;font-weight:400 }
  .chart-tooltip-footer{
    font-size:10.5px;color:#6b7380;margin-top:12px;padding-top:10px;
    border-top:1px solid #2d3748;font-style:italic;
  }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
<script>
/* Sector sweep + theme transition script */
window._scdss = {
  sector: "energy",
  themes: {
    energy: {bg:"#0a0e14",surface:"#111820",accent:"#f5a623",sweep:"#f5a623",border:"#1e3048",text:"#e8eef5"},
    agri:   {bg:"#071a0a",surface:"#0d2310",accent:"#4caf50",sweep:"#4caf50",border:"#1c4220",text:"#e8f5e9"},
    medtech:{bg:"#140d20",surface:"#1c1230",accent:"#b39ddb",sweep:"#9575cd",border:"#342050",text:"#f3eeff"},
  },
  switchTo: function(sec){
    if(sec===this.sector) return;
    var sweep = document.getElementById("sector-sweep");
    if(!sweep) return;
    var t = this.themes[sec];
    sweep.style.background = t.sweep;
    sweep.style.transform = "translateX(-105%)";
    sweep.classList.remove("sweep-go");
    void sweep.offsetWidth;
    sweep.classList.add("sweep-go");
    var self = this;
    setTimeout(function(){
      document.body.style.background = t.bg;
      self.sector = sec;
    }, 300);
    setTimeout(function(){
      sweep.classList.remove("sweep-go");
      sweep.style.transform = "translateX(-105%)";
    }, 700);
  }
};
// Veil transition colours per sector
window._scdss.veilColors = {
  energy:  ["#0a0e14","#1e3048"],
  agri:    ["#071a0a","#1c4220"],
  medtech: ["#140d20","#342050"],
};

document.addEventListener("click", function(e){
  var btn = e.target.closest("a[href]");
  if(!btn) return;
  var href = btn.getAttribute("href") || "";
  if(!href.startsWith("/")) return;

  // Determine target sector
  var sec = "energy";
  if(href.startsWith("/a-")) sec = "agri";
  else if(href.startsWith("/m-")) sec = "medtech";

  // Trigger sector sweep only on sector-switch pill clicks (d1 routes)
  if(href==="/e-d1"||href==="/a-d1"||href==="/m-d1"){
    window._scdss.switchTo(sec);
  }

  // Veil wipe on ALL internal link clicks (tab navigation)
  var veil = document.getElementById("page-veil");
  if(veil){
    var cols = window._scdss.veilColors[window._scdss.sector] || window._scdss.veilColors.energy;
    veil.style.setProperty("--veil-a", cols[0]);
    veil.style.setProperty("--veil-b", cols[1]);
    // Reset
    veil.className = "";
    void veil.offsetWidth;
    // Sweep in
    veil.classList.add("veil-in");
    // After content loads, sweep out
    setTimeout(function(){
      veil.classList.remove("veil-in");
      veil.classList.add("veil-out");
      setTimeout(function(){
        veil.className = "";
      }, 300);
    }, 320);
  }
});
</script>
</body>
</html>'''

# ?? Tab definitions per sector ?????????????????????????????????????????????????
ENERGY_TABS = [
    ("e-d1","Strategic Overview"),("e-d14","Export Dependency"),
    ("e-d2","Import Dependency"),("e-d3","Supply Flow"),
    ("e-d4","Trade Map"),("e-d6","Stress Testing"),
    ("e-d7","Scenario War Room"),("e-d10","ML Analysis"),
]
AGRI_TABS = [
    ("a-d1","Strategic Overview"),("a-d3","Export Dependency"),
    ("a-d2","Import Dependency"),("a-d4","Supply Flow"),
    ("a-d5","Trade Map"),("a-d6","Stress Test"),
    ("a-d7","Seasonal Risk"),("a-d8","Food Security"),
]
MED_TABS = [
    ("m-d1","Strategic Overview"),("m-d3","Export Dependency"),
    ("m-d2","Import Dependency"),("m-d4","Supply Flow"),
    ("m-d5","Trade Map"),("m-d6","Stress Test"),
    ("m-d7","Scenario War Room"),("m-d8","Product Analysis"),
]

def sector_tab_bar(current_sector, current_slug):
    tabs_map = {"energy":ENERGY_TABS,"agri":AGRI_TABS,"medtech":MED_TABS}
    tabs = tabs_map.get(current_sector, ENERGY_TABS)
    th = THEMES[current_sector]
    items = []
    for slug, label in tabs:
        is_active = (slug == current_slug)
        items.append(html.A(label, href=f"/{slug}",
            className="dash-tab-link",
            style={"display":"block","padding":"11px 16px","fontSize":"13px",
                   "color": th["accent"] if is_active else th["muted"],
                   "textDecoration":"none",
                   "borderBottom": f"2px solid {th['accent']}" if is_active else "2px solid transparent",
                   "whiteSpace":"nowrap","transition":"color 0.2s,border-color 0.2s"}))
    return items

def sector_pill(current_sector):
    th = THEMES[current_sector]
    def btn(sec, label, dot_color, href):
        is_active = (sec == current_sector)
        dot = html.Span(style={"width":"7px","height":"7px","borderRadius":"50%",
                               "background":dot_color,"display":"inline-block",
                               "marginRight":"5px","flexShrink":"0"})
        return html.A([dot, label], href=href,
            style={"display":"flex","alignItems":"center","padding":"5px 13px",
                   "borderRadius":"20px","cursor":"pointer","fontSize":"12px",
                   "fontWeight":"500","letterSpacing":"0.02em","textDecoration":"none",
                   "background": th["accent"] if is_active else "transparent",
                   "color": th["bg"] if is_active else th["muted"],
                   "transition":"all 0.3s cubic-bezier(.4,0,.2,1)"})
    return html.Div([
        btn("energy","Energy","#f5a623","/e-d1"),
        btn("agri","Agriculture","#4caf50","/a-d1"),
        btn("medtech","MedTech","#b39ddb","/m-d1"),
    ], style={"display":"flex","gap":"3px","padding":"3px","borderRadius":"22px",
              "border":f"1px solid {th['border']}","background":"rgba(255,255,255,0.04)"})

def topbar(current_sector, current_slug):
    th = THEMES[current_sector]
    sector_labels = {"energy":"Energy SC-DSS","agri":"Agriculture SC-DSS","medtech":"MedTech SC-DSS"}
    sources = {
        "energy":"SEAI NEB 1990-2024 . UN Comtrade HS 27xx",
        "agri":"CSO . UN Comtrade HS 01-22 . FAOSTAT",
        "medtech":"UN Comtrade HS 9018/9021/9022 . 2015-2024",
    }
    return html.Div([
        html.Div([
            html.Div(style={"width":"9px","height":"9px","borderRadius":"50%",
                            "background":th["accent"],"marginRight":"8px","flexShrink":"0",
                            "transition":"background 0.4s"}),
            html.Span(f"Ireland {sector_labels[current_sector]}",
                      style={"fontSize":"15px","fontWeight":"600","color":th["text"],
                             "transition":"color 0.4s"}),
        ], style={"display":"flex","alignItems":"center"}),
        html.Div([
            html.Span(sources[current_sector],
                      style={"fontSize":"11px","color":th["muted"]}),
        ], style={"display":"flex","alignItems":"center","gap":"4px","flex":"1","justifyContent":"center"}),
        sector_pill(current_sector),
    ], style={"background":th["surface"],"borderBottom":f"1px solid {th['border']}",
              "padding":"10px 20px","display":"flex","justifyContent":"space-between",
              "alignItems":"center","transition":"background 0.4s,border-color 0.4s"})

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="sector-store", data="energy", storage_type="session"),
    # Sweep overlay element (sector switch)
    html.Div(id="sector-sweep"),
    # Veil overlay element (tab navigation)
    html.Div(id="page-veil"),

    # Dynamic topbar
    html.Div(id="topbar-area"),
    # Dynamic tab bar
    html.Div(id="tab-bar-area",
             style={"borderBottom":f"1px solid {BORDER}","overflowX":"auto"}),
    # Page content
    dcc.Loading(
        id="page-loading",
        type="circle",
        color="#f5a623",
        style={"position":"fixed","top":"50%","left":"50%","transform":"translate(-50%,-50%)","zIndex":"1000"},
        children=html.Div(id="page-content",
                 style={"padding":"20px 24px","maxWidth":"1440px","margin":"0 auto"}),
    ),
], style={"background":BG_PAGE,"minHeight":"100vh","fontFamily":"Inter,Segoe UI,Arial",
          "transition":"background 0.5s"})


# ???????????????????????????????????????????????????????
# ROUTING CALLBACKS
# ???????????????????????????????????????????????????????
def slug_to_sector(slug):
    if slug.startswith("a-"): return "agri"
    if slug.startswith("m-"): return "medtech"
    return "energy"

@app.callback(Output("sector-store","data"), Input("url","pathname"))
def update_sector_store(path):
    slug = (path or "/e-d1").lstrip("/")
    return slug_to_sector(slug)

@app.callback(Output("topbar-area","children"),
              Input("url","pathname"))
def update_topbar(path):
    slug = (path or "/e-d1").lstrip("/")
    sector = slug_to_sector(slug)
    return topbar(sector, slug)

@app.callback(Output("tab-bar-area","children"),
              Input("url","pathname"))
def update_tab_bar(path):
    slug = (path or "/e-d1").lstrip("/")
    sector = slug_to_sector(slug)
    th = THEMES[sector]
    return html.Div(sector_tab_bar(sector, slug),
                    style={"display":"flex","background":th["surface"],
                           "transition":"background 0.4s"})

@app.callback(Output("page-content","children"), Input("url","pathname"))
def route(path):
    slug = (path or "/e-d1").lstrip("/")
    routes = {
        # Energy
        "e-d1":e_d1_layout,"e-d2":e_d2_layout,"e-d3":e_d3_layout,"e-d4":e_d4_layout,
        "e-d6":e_d6_layout,"e-d7":e_d7_layout,"e-d10":e_d10_layout,"e-d14":e_d14_layout,
        # Agriculture
        "a-d1":a_d1_layout,"a-d2":a_d2_layout,"a-d3":a_d3_layout,"a-d4":a_d4_layout,
        "a-d5":a_d5_layout,"a-d6":a_d6_layout,"a-d7":a_d7_layout,"a-d8":a_d8_layout,
        # MedTech
        "m-d1":m_d1_layout,"m-d2":m_d2_layout,"m-d3":m_d3_layout,"m-d4":m_d4_layout,
        "m-d5":m_d5_layout,"m-d6":m_d6_layout,"m-d7":m_d7_layout,"m-d8":m_d8_layout,
    }
    fn = routes.get(slug, e_d1_layout)
    return fn()

# Sector switching via html.A href links in sector_pill()

# Raw data toggle callbacks -- one per dashboard, all using unique IDs
def make_raw_toggle(btn_id, panel_id, content_id, tab_id, tables):
    @app.callback(
        Output(panel_id,"style"),
        Output(btn_id,"children"),
        Input(btn_id,"n_clicks"),
        prevent_initial_call=True
    )
    def toggle(n):
        if (n or 0) % 2 == 1:
            return {"display":"block","marginTop":"10px"}, "?  Hide Raw Data"
        return {"display":"none","marginTop":"10px"}, "?  Show Raw Data"

    @app.callback(
        Output(content_id,"children"),
        Input(tab_id,"value"),
        prevent_initial_call=False
    )
    def show_table(tab_value):
        for lbl, df, cols in tables:
            if lbl == tab_value:
                if df is None or (hasattr(df,"empty") and df.empty):
                    return html.P("No data available", style={"color":TEXT_MUTED,"fontSize":"12px"})
                disp = df.head(200) if hasattr(df,"head") else df
                return dash_table.DataTable(
                    data=disp.to_dict("records"),
                    columns=[{"name":c,"id":c} for c in (cols or disp.columns.tolist())],
                    page_size=20,
                    style_table={"overflowX":"auto"},
                    style_header={"backgroundColor":"#21262d","color":TEXT_PRI,"fontWeight":"600",
                                  "fontSize":"11px","border":f"1px solid {BORDER}"},
                    style_cell={"padding":"7px 10px","fontSize":"11px","fontFamily":"Inter,Segoe UI,Arial",
                                "backgroundColor":BG_CARD,"color":TEXT_PRI,"border":f"1px solid {BORDER}",
                                "maxWidth":"200px","overflow":"hidden","textOverflow":"ellipsis"},
                )
        return html.P("Select a tab", style={"color":TEXT_MUTED,"fontSize":"12px"})
    return toggle, show_table


# ???????????????????????????????????????????????????????
# ENERGY DASHBOARDS (all 8 -- preserved exactly from v3)
# ???????????????????????????????????????????????????????
SEC = "energy"

def e_header(title, subtitle, controls):
    th = THEMES[SEC]
    return html.Div([
        html.Div([
            html.H2(title, style={"color":th["text"],"fontWeight":"600","margin":"0 0 2px","fontSize":"20px"}),
            html.P(subtitle, style={"color":th["muted"],"margin":0,"fontSize":"12px"}),
        ], style={"flex":"1"}),
        *controls,
    ], style={"display":"flex","gap":"24px","alignItems":"flex-end","marginBottom":"16px",
              "background":th["card"],"border":f"1px solid {th['border']}","borderRadius":"8px","padding":"16px"})

# ?? E-D1: Strategic Overview ??????????????????????????????????????????????????
def e_d1_layout():
    th = THEMES[SEC]
    return html.Div([
        e_header("Energy -- Strategic Overview",
                 "Executive supply chain risk posture -- HHI, oil dependency, self-sufficiency",
                 [yr_dropdown("e-d1-year",year_list=SEAI_DROPDOWN_YEARS,multi=True),
                  html.Div(yr_range_slider("e-d1-range"),style={"minWidth":"320px","flex":"1"})]),
        html.Div(id="e-d1-kri", style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(dark_card("HHI by fuel type",[dark_graph("e-d1-hhi-bar",260)],flex="2",sector=SEC, info_id="HHI by fuel type"),
            dark_card("Risk signal feed",[html.Div(id="e-d1-insights",style={"overflowY":"auto","maxHeight":"280px"})],flex="1",sector=SEC, info_id="Risk signal feed")),
        row(dark_card("Oil import dependency trend (%)",[dark_graph("e-d1-oil-trend",200)],flex="1",sector=SEC, info_id="Oil import dependency trend (%)"),
            dark_card("Self-sufficiency trend (%)",[dark_graph("e-d1-ss-trend",200)],flex="1",sector=SEC, info_id="Self-sufficiency trend (%)")),
        dark_card("Critical fuel summary",[html.Div(id="e-d1-table")],sector=SEC, info_id="Critical fuel summary"),
        raw_data_section("e-d1",[
            ("SEAI Energy Balance", seai.head(500) if not seai.empty else None,
             ["year","flow","oil","natural_gas","electricity","total"]),
            ("Comtrade Energy", comtrade.head(500) if not comtrade.empty else None,
             ["refyear","flowcode","partnerdesc","cmddesc","primaryvalue"]),
        ]),
    ])

@app.callback(
    Output("e-d1-kri","children"),Output("e-d1-hhi-bar","figure"),
    Output("e-d1-insights","children"),Output("e-d1-table","children"),
    Output("e-d1-oil-trend","figure"),Output("e-d1-ss-trend","figure"),
    Input("url","pathname"),Input("e-d1-year","value"),Input("e-d1-range","value"))
def cb_e_d1(pathname, years, yr_range):
    sel = resolve_years(years); year = max(sel) if sel else DEFAULT_YEAR
    k = kri_for_year(year)
    def col_oil(v): return C_RED if v>70 else C_ORANGE if v>55 else C_AMBER if v>40 else C_GREEN
    row_c = S_IMP[S_IMP["year"]==year] if not S_IMP.empty else pd.DataFrame(); crit=0
    if not row_c.empty:
        r=row_c.iloc[0]; tot=float(r.get("total",1)) if pd.notna(r.get("total",1)) and r.get("total",1)>0 else 1.0
        crit = sum(1 for f in IMP_FUELS if (safe_val(r,f)/tot)**2 > 0.40)
    ct_imp=k.get("ct_imp_val",0); ct_exp=k.get("ct_exp_val",0); ct_yr_used=k.get("ct_year_used",year)
    cards = html.Div([
        kri_card("Average HHI",f"{k.get('hhi',0):.3f}","",hhi_colour(k.get("hhi",0)),hhi_label(k.get("hhi",0)),SEC),
        kri_card("Oil Import Dependency",f"{k.get('oil_dep',0)}","%",col_oil(k.get("oil_dep",0)),f"SEAI {year}",SEC),
        kri_card("Self-Sufficiency",f"{k.get('self_suff',0)}","%",
                 C_GREEN if k.get("self_suff",0)>30 else C_AMBER if k.get("self_suff",0)>15 else C_RED,
                 "Indigenous / Primary Energy Req",SEC),
        kri_card("Supplier Countries",str(k.get("n_suppliers","N/A")),"",ACCENT2,f"Import partners {ct_yr_used}",SEC),
        kri_card("Critical Fuels (HHI>0.40)",str(crit),"fuels",C_RED if crit>0 else C_GREEN,"Fuels above critical threshold",SEC),
        kri_card("Total Energy Imports",f"{k.get('total_imports_ktoe',0):,.0f}","ktoe",C_RED,
                 f"{fmt_val(ct_imp)} . Comtrade {ct_yr_used}" if ct_imp>0 else f"SEAI {year}",SEC),
        kri_card("Total Energy Exports",f"{k.get('total_exports_ktoe',0):,.0f}","ktoe",C_GREEN,
                 f"{fmt_val(ct_exp)} . Comtrade {ct_yr_used}" if ct_exp>0 else f"SEAI {year}",SEC),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap"})
    fig = go.Figure()
    for yr in sel:
        row_s = S_IMP[S_IMP["year"]==yr] if not S_IMP.empty else pd.DataFrame()
        if row_s.empty: continue
        r=row_s.iloc[0]; tot=float(r.get("total",1)) if pd.notna(r.get("total",1)) and r.get("total",1)>0 else 1.0
        hhis=[round((safe_val(r,f)/tot)**2,4) for f in IMP_FUELS]
        cols=[hhi_colour(h) for h in hhis] if len(sel)==1 else None
        fig.add_trace(go.Bar(name=str(yr),x=[f.replace("_"," ").title() for f in IMP_FUELS],y=hhis,
                             marker_color=cols,text=[f"{h:.3f}" for h in hhis],textposition="inside",insidetextanchor="middle",
                             textfont=dict(size=12,color="#ffffff")))
    fig.add_hline(y=0.40,line_dash="dash",line_color=C_RED,opacity=0.8,
                  annotation_text="Critical (0.40)",annotation_font=dict(color=C_RED,size=11))
    fig.add_hline(y=0.25,line_dash="dot",line_color=C_AMBER,opacity=0.7,
                  annotation_text="High (0.25)",annotation_font=dict(color=C_AMBER,size=11))
    if len(sel)>1: fig.update_layout(barmode="group")
    themed_layout(fig,SEC,len(sel)>1)
    fig.update_xaxes(title=dict(text="Fuel Type",font=dict(color=TEXT_SEC,size=12)))
    fig.update_yaxes(title=dict(text="HHI Score (0=diversified, 1=single source)",font=dict(color=TEXT_SEC,size=12)))
    oil_dep=k.get("oil_dep",0); fossil_pct=k.get("fossil_pct",0)
    hhi_v=k.get("hhi",0); ss=k.get("self_suff",0); top_sh=k.get("top_share",0)
    insights=[]
    if oil_dep>65:    insights.append(("CRITICAL",C_RED,"Concentration Spike",f"Oil = {oil_dep}% of imports -- critical"))
    if fossil_pct>90: insights.append(("HIGH",C_ORANGE,"Fossil Dependency",f"Fossil fuels = {fossil_pct}% of imports"))
    if ss<20:         insights.append(("HIGH",C_ORANGE,"Low Self-Sufficiency",f"Self-sufficiency = {ss}% -- heavy import dependency"))
    if top_sh>30:     insights.append(("MODERATE",C_AMBER,"Top Supplier",f"{k.get('top_supplier','N/A')} = {top_sh}% of imports"))
    if hhi_v>0.25:    insights.append(("HIGH",C_ORANGE,"HHI Critical",f"Overall HHI = {hhi_v:.3f} -- concentrated supply"))
    if not insights:  insights.append(("LOW",C_GREEN,"No Alerts","All indicators within acceptable ranges"))
    ins_els = [insight_item(s,c,t,m) for s,c,t,m in insights]
    tbl_data=[]
    if not row_c.empty:
        r=row_c.iloc[0]; tot=float(r.get("total",1)) if pd.notna(r.get("total",1)) and r.get("total",1)>0 else 1.0
        for f in IMP_FUELS:
            v=safe_val(r,f); sh=v/tot*100; hf=(v/tot)**2
            tbl_data.append({"Fuel":f.replace("_"," ").title(),"Import (ktoe)":f"{v:,.0f}",
                              "Share (%)":f"{sh:.1f}%","HHI":f"{hf:.4f}","Risk":hhi_label(hf)})
    tbl = dash_table.DataTable(data=tbl_data,
        columns=[{"name":c,"id":c} for c in ["Fuel","Import (ktoe)","Share (%)","HHI","Risk"]],
        style_header={"backgroundColor":"#21262d","color":TEXT_PRI,"fontWeight":"600","fontSize":"12px","border":f"1px solid {BORDER}"},
        style_cell={"padding":"10px 14px","fontSize":"13px","fontFamily":"Inter,Segoe UI,Arial",
                    "backgroundColor":BG_CARD,"color":TEXT_PRI,"border":f"1px solid {BORDER}"},
        style_data_conditional=[
            {"if":{"filter_query":'{Risk} = "Critical"'},"backgroundColor":"#2d1a1a","color":C_RED},
            {"if":{"filter_query":'{Risk} = "High"'},"backgroundColor":"#2d2316","color":C_ORANGE},
            {"if":{"filter_query":'{Risk} = "Moderate"'},"backgroundColor":"#2a2416","color":C_AMBER},
        ])
    rng=yr_range or [2010,YEAR_MAX]; rng_years=list(range(int(rng[0]),int(rng[1])+1))
    oil_t=[]; ss_t=[]; t_yrs=[]
    for y in rng_years:
        ri=S_IMP[S_IMP["year"]==y] if not S_IMP.empty else pd.DataFrame()
        rp=S_PER[S_PER["year"]==y] if not S_PER.empty else pd.DataFrame()
        rd=S_IND[S_IND["year"]==y] if not S_IND.empty else pd.DataFrame()
        if ri.empty: continue
        r=ri.iloc[0]; tot=float(r.get("total",1)) if pd.notna(r.get("total",1)) and r.get("total",1)>0 else 1.0
        oil_t.append(round(safe_val(r,"oil")/tot*100,1))
        ss_v=0.0
        if not rp.empty and not rd.empty:
            pv=float(rp.iloc[0].get("total",1)) if pd.notna(rp.iloc[0].get("total",1)) else 1.0
            iv=float(rd.iloc[0].get("total",0)) if pd.notna(rd.iloc[0].get("total",0)) else 0.0
            ss_v=round(max(iv,0)/max(pv,1)*100,1)
        ss_t.append(ss_v); t_yrs.append(y)
    def trend_fig(yvals,col,ytitle,trace_name,ref_line=None,ref_label=None):
        f=go.Figure(go.Scatter(x=t_yrs,y=yvals,mode="lines+markers",name=trace_name,
            line=dict(color=col,width=2.5),marker=dict(color=col,size=6),
            hovertemplate=f"%{{x}}: %{{y:.1f}}%<extra>{trace_name}</extra>"))
        if ref_line:
            f.add_hline(y=ref_line,line_dash="dot",line_color=C_AMBER,opacity=0.7,
                        annotation_text=ref_label,annotation_font=dict(color=C_AMBER,size=10))
        themed_layout(f,SEC,False)
        f.update_xaxes(title=dict(text="Year",font=dict(color=TEXT_SEC,size=11),standoff=12))
        f.update_yaxes(title=dict(text=ytitle,font=dict(color=TEXT_SEC,size=11)),ticksuffix="%")
        f
        return f
    fig_oil=trend_fig(oil_t,C_RED,"% of Total Imports","Oil Import Dependency",65,"65% threshold")
    fig_ss=trend_fig(ss_t,C_GREEN,"Self-Sufficiency (%)","Self-Sufficiency",25,"25% target")
    return cards,fig,ins_els,tbl,fig_oil,fig_ss

make_raw_toggle("e-d1-raw-toggle-btn","e-d1-raw-panel","e-d1-raw-content","e-d1-raw-source-tabs",[
    ("SEAI Energy Balance",seai.head(500) if not seai.empty else None,["year","flow","oil","natural_gas","electricity","total"]),
    ("Comtrade Energy",comtrade.head(500) if not comtrade.empty else None,["refyear","flowcode","partnerdesc","cmddesc","primaryvalue"]),
])


# ?? E-D2: Import Dependency ????????????????????????????????????????????????????
def e_d2_layout():
    return html.Div([
        e_header("Energy -- Import Dependency Analysis",
                 "HHI concentration, country dependency scores, continental breakdown, historical trend",
                 [yr_dropdown("e-d2-year",year_list=SEAI_DROPDOWN_YEARS,multi=True),
                  html.Div(yr_range_slider("e-d2-range"),style={"minWidth":"320px","flex":"1"})]),
        row(dark_card("HHI by Fuel Type",[dark_graph("e-d2-hhi",260)],flex="1",badge="Higher = more concentrated = higher risk",sector=SEC, info_id="HHI by Fuel Type"),
            dark_card("Country Dependency Score",[dark_graph("e-d2-dep",260)],flex="1",badge="0.4 x share + 0.3 x dominance + 0.3 x gov risk",sector=SEC, info_id="Country Dependency Score")),
        row(dark_card("Continental Breakdown -- Import vs Export",[dark_graph("e-d2-cont",240)],flex="1",sector=SEC, info_id="Continental Breakdown -- Import vs Export"),
            dark_card("HHI Trend Over Time",[dark_graph("e-d2-ht",240)],flex="1",badge="Year range slider above",sector=SEC, info_id="HHI Trend Over Time")),
        raw_data_section("e-d2",[
            ("Comtrade Imports",comtrade[comtrade["flowcode"]=="M"].head(500) if not comtrade.empty else None,
             ["refyear","partnerdesc","cmddesc","primaryvalue","continent"]),
        ]),
    ])

@app.callback(
    Output("e-d2-hhi","figure"),Output("e-d2-dep","figure"),
    Output("e-d2-cont","figure"),Output("e-d2-ht","figure"),
    Input("url","pathname"),Input("e-d2-year","value"),Input("e-d2-range","value"))
def cb_e_d2(pathname,years,yr_range):
    year=resolve_year(years); yr_range=yr_range or [2010,YEAR_MAX]
    # Fix #2-B: use TRUE country-concentration HHI per fuel (Σ country_share²
    # across supplier countries for each fuel), not fuel-mix concentration.
    hhi_data = compute_hhi(year)
    by_fuel = hhi_data.get("by_fuel", {})
    labels=[f.replace("_"," ").title() for f in IMP_FUELS]
    hhis  =[by_fuel.get(f, 0.0) for f in IMP_FUELS]
    pairs=sorted(zip(hhis,labels),reverse=True); hs,ls=zip(*pairs) if pairs else ([],[])
    fig_hhi=go.Figure(go.Bar(y=list(ls),x=list(hs),orientation="h",
        marker_color=[hhi_colour(h) for h in hs],
        text=[f"{h:.4f}  {hhi_label(h)}" for h in hs],textposition="outside",
        textfont=dict(color=TEXT_PRI,size=11),name="Country-HHI"))
    fig_hhi.add_vline(x=0.40,line_dash="dash",line_color=C_RED,opacity=0.8,annotation_text="Critical 0.40",annotation_font=dict(color=C_RED,size=10))
    fig_hhi.add_vline(x=0.25,line_dash="dot",line_color=C_AMBER,opacity=0.7,annotation_text="High 0.25",annotation_font=dict(color=C_AMBER,size=10))
    themed_layout(fig_hhi,SEC,False); fig_hhi.update_xaxes(title=dict(text="Country-HHI (supplier concentration)",font=dict(color=TEXT_SEC,size=11),standoff=12),range=[0,1.05])
    fig_hhi.update_yaxes(title=dict(text="Fuel Type",font=dict(color=TEXT_SEC,size=11),standoff=12)); fig_hhi
    dep=compute_dependency(year); JUNK={"world","other","areas, nes","total","unspecified","not specified"}
    if not dep.empty:
        dep=dep[~dep["country"].astype(str).str.strip().str.lower().isin(JUNK)].head(10)
        fig_dep=go.Figure(go.Bar(y=dep["country"],x=dep["dependency"],orientation="h",
            marker_color=[C_RED if v>0.4 else C_ORANGE if v>0.25 else C_AMBER if v>0.15 else ACCENT2 for v in dep["dependency"]],
            text=[f"{v:.3f}" for v in dep["dependency"]],textposition="inside",insidetextanchor="middle",textfont=dict(color=TEXT_PRI,size=11),name="Dependency Score"))
        themed_layout(fig_dep,SEC,False)
        fig_dep.update_xaxes(title=dict(text="Composite Score (0-1)",font=dict(color=TEXT_SEC,size=11),standoff=12),range=[0,0.90])
        fig_dep.update_yaxes(title=dict(text="Supplier Country",font=dict(color=TEXT_SEC,size=11),standoff=12))
        fig_dep
    else:
        fig_dep=go.Figure(); themed_layout(fig_dep,SEC,False)
    if not comtrade.empty:
        ct_y=comtrade[comtrade["refyear"]==ct_nearest(year)]
        def cagg(fc):
            s=ct_y[ct_y["flowcode"]==fc]
            return s.groupby("continent")["primaryvalue"].sum() if not s.empty and "continent" in s.columns else pd.Series(dtype=float)
        ic=cagg("M"); ec=cagg("X"); conts=sorted(set(list(ic.index)+list(ec.index)))
        fig_cont=go.Figure([
            go.Bar(name="Imports (EUR bn)",x=conts,y=[ic.get(c,0)/1e9 for c in conts],marker_color=C_RED),
            go.Bar(name="Exports (EUR bn)",x=conts,y=[ec.get(c,0)/1e9 for c in conts],marker_color=C_GREEN),
        ]); fig_cont.update_layout(barmode="group")
        themed_layout(fig_cont,SEC,True)
        fig_cont.update_xaxes(title=dict(text="Region",font=dict(color=TEXT_SEC,size=11),standoff=12))
        fig_cont.update_yaxes(title=dict(text="Trade Value (EUR Billion)",font=dict(color=TEXT_SEC,size=11),standoff=12))
        fig_cont.update_layout(legend=dict(orientation="h",y=-0.25,x=0,font=dict(size=11),bgcolor="rgba(0,0,0,0)"))
    else:
        fig_cont=go.Figure(); themed_layout(fig_cont,SEC,False)
    hhi_t=[]
    for y in range(int(yr_range[0]),int(yr_range[1])+1):
        # Fix #2-B: use weighted country-HHI aggregate (from compute_hhi),
        # which is supplier-concentration per fuel × fuel-share weights.
        hhi_y = compute_hhi(y).get("weighted", 0.0)
        if hhi_y > 0:
            hhi_t.append({"year": y, "hhi": hhi_y})
    df_ht=pd.DataFrame(hhi_t); fig_ht=go.Figure()
    if not df_ht.empty:
        fig_ht.add_trace(go.Scatter(x=df_ht["year"],y=df_ht["hhi"],mode="lines+markers",
            name="HHI",line=dict(color=ACCENT2,width=2.5),
            marker=dict(color=[hhi_colour(h) for h in df_ht["hhi"]],size=7,line=dict(color=BG_CARD,width=1))))
        fig_ht.add_hline(y=0.40,line_dash="dash",line_color=C_RED,opacity=0.8,annotation_text="Critical (0.40)",annotation_font=dict(color=C_RED,size=10))
        fig_ht.add_hline(y=0.25,line_dash="dot",line_color=C_AMBER,opacity=0.7,annotation_text="High (0.25)",annotation_font=dict(color=C_AMBER,size=10))
    themed_layout(fig_ht,SEC,True)
    fig_ht.update_xaxes(title=dict(text="Year",font=dict(color=TEXT_SEC,size=11),standoff=12))
    fig_ht.update_yaxes(title=dict(text="HHI Score",font=dict(color=TEXT_SEC,size=11),standoff=12))
    fig_ht.update_layout(legend=dict(orientation="h",y=-0.25,x=0,font=dict(size=11),bgcolor="rgba(0,0,0,0)"))
    return fig_hhi,fig_dep,fig_cont,fig_ht

make_raw_toggle("e-d2-raw-toggle-btn","e-d2-raw-panel","e-d2-raw-content","e-d2-raw-source-tabs",[
    ("Comtrade Imports",comtrade[comtrade["flowcode"]=="M"].head(500) if not comtrade.empty else None,
     ["refyear","partnerdesc","cmddesc","primaryvalue","continent"]),
    ("SEAI Imports",seai[seai["flow"]=="Imports"].head(200) if not seai.empty else None,
     ["year","flow","oil","natural_gas","electricity","total"]),
])


# ?? E-D3: Supply Flow Sankey ???????????????????????????????????????????????????
def e_d3_layout():
    return html.Div([
        e_header("Energy -- Supply Flow Sankey",
                 "Directional energy flows -- import origins >> Ireland >> export destinations",
                 [yr_dropdown("e-d3-year")]),
        dark_card("Supply Flow Sankey Diagram",[dark_graph("e-d3-sankey",420)],sector=SEC, info_id="Supply Flow Sankey Diagram"),
        html.Div([html.Div(id="e-d3-import-table"),html.Div(id="e-d3-export-table"),html.Div(id="e-d3-flow-summary")],
                 style={"display":"flex","gap":"12px","marginTop":"12px"}),
        raw_data_section("e-d3",[
            ("Comtrade All Flows",comtrade.head(500) if not comtrade.empty else None,
             ["refyear","flowcode","partnerdesc","cmddesc","primaryvalue","continent"]),
        ]),
    ])

@app.callback(Output("e-d3-sankey","figure"),Output("e-d3-import-table","children"),
              Output("e-d3-export-table","children"),Output("e-d3-flow-summary","children"),
              Input("url","pathname"),Input("e-d3-year","value"))
def cb_e_d3(pathname,years):
    year=resolve_year(years); ct_year=ct_nearest(year)
    if comtrade.empty: return go.Figure(),html.P("No data"),html.P("No data"),html.P("No data")
    ct_y=comtrade[comtrade["refyear"]==ct_year]
    imp=ct_y[ct_y["flowcode"]=="M"]; exp=ct_y[ct_y["flowcode"]=="X"]
    PAL=[ACCENT,ACCENT2,C_ORANGE,C_PURPLE,C_TEAL,C_GREEN,C_AMBER,"#e879f9","#67e8f9"]
    # Country-level Sankey -- top 8 import sources, top 6 export markets
    JUNK_S = {"world","areas, nes","other","unspecified","not specified","total"}
    imp_c = imp[~imp["partnerdesc"].str.strip().str.lower().isin(JUNK_S)]
    exp_c = exp[~exp["partnerdesc"].str.strip().str.lower().isin(JUNK_S)] if not exp.empty else exp
    ic=imp_c.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(8)
    ec=exp_c.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(6) if not exp_c.empty else pd.Series(dtype=float)
    nodes=[f">> {r}" for r in ic.index]+["Ireland"]+[f"<< {r}" for r in ec.index]
    ni={name:i for i,name in enumerate(nodes)}
    src,tgt,vals,cols=[],[],[],[]
    for i,(region,val) in enumerate(ic.items()):
        src.append(ni[f">> {region}"]); tgt.append(ni["Ireland"])
        vals.append(val/1e9); cols.append(hex_rgba(PAL[i%len(PAL)],0.55))
    for region,val in ec.items():
        if f"<< {region}" in ni:
            src.append(ni["Ireland"]); tgt.append(ni[f"<< {region}"])
            vals.append(val/1e9); cols.append(hex_rgba(C_GREEN,0.5))
    fig=go.Figure(go.Sankey(arrangement="snap",
        node=dict(pad=15,thickness=18,label=nodes,
                  color=[hex_rgba(PAL[i%len(PAL)],0.9) if i<len(ic) else
                         hex_rgba("#58a6ff",0.9) if i==len(ic) else hex_rgba(C_GREEN,0.9) for i in range(len(nodes))],
                  line=dict(color=BORDER,width=0.5)),
        link=dict(source=src,target=tgt,value=vals,color=cols,
                  hovertemplate="%{source.label} >> %{target.label}<br>EUR %{value:.2f}bn<extra></extra>")))
    sankey_layout(fig,SEC)
    def make_table(df_g,title):
        if df_g is None or df_g.empty:
            return dark_card(title,[html.P("No data",style={"color":TEXT_MUTED,"fontSize":"12px"})],flex="1",sector=SEC, info_id=title)
        g=df_g.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(8)
        rows=[html.Div([html.Span(f"{p}",style={"fontSize":"12px","color":TEXT_SEC,"flex":"1"}),
                        html.Span(fmt_val(v),style={"fontSize":"12px","color":TEXT_PRI,"fontWeight":"600"})],
                       style={"display":"flex","justifyContent":"space-between","padding":"6px 0",
                              "borderBottom":f"1px solid {BORDER}"}) for p,v in g.items()]
        return dark_card(title,rows,flex="1",sector=SEC, info_id=title)
    imp_tbl=make_table(imp,"Top Import Sources")
    exp_tbl=make_table(exp,"Top Export Markets") if not exp.empty else dark_card("Top Export Markets",[html.P("No data",style={"color":TEXT_MUTED})],flex="1",sector=SEC, info_id="Top Export Markets")
    summary=dark_card("Flow Summary",[
        html.Div([html.Span("Total Imports",style={"color":TEXT_MUTED,"fontSize":"11px"}),
                  html.Span(fmt_val(imp["primaryvalue"].sum()),style={"color":C_RED,"fontWeight":"700","fontSize":"16px"})],style={"marginBottom":"8px"}),
        html.Div([html.Span("Total Exports",style={"color":TEXT_MUTED,"fontSize":"11px"}),
                  html.Span(fmt_val(exp["primaryvalue"].sum()) if not exp.empty else "N/A",style={"color":C_GREEN,"fontWeight":"700","fontSize":"16px"})],style={"marginBottom":"8px"}),
        html.Div([html.Span("Import Partners",style={"color":TEXT_MUTED,"fontSize":"11px"}),
                  html.Span(str(imp["partnerdesc"].nunique()),style={"color":ACCENT2,"fontWeight":"700","fontSize":"16px"})]),
    ],flex="1",sector=SEC, info_id="Flow Summary")
    return fig,imp_tbl,exp_tbl,summary

make_raw_toggle("e-d3-raw-toggle-btn","e-d3-raw-panel","e-d3-raw-content","e-d3-raw-source-tabs",[
    ("Comtrade All Flows",comtrade.head(500) if not comtrade.empty else None,
     ["refyear","flowcode","partnerdesc","cmddesc","primaryvalue","continent"]),
])

# ?? E-D4: Geographic Trade Map ?????????????????????????????????????????????????
def e_d4_layout():
    return html.Div([
        e_header("Energy -- Geographic Trade Map",
                 "Bilateral energy trade corridors -- arc width = trade value",
                 [yr_dropdown("e-d4-year"),
                  html.Div([
                      html.Label("Flow",style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="e-d4-flow",options=[{"label":"Both","value":"Both"},{"label":"Imports only","value":"M"},{"label":"Exports only","value":"X"}],
                                   value="Both",clearable=False,style={"fontSize":"13px","width":"160px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {BORDER}"}),
                  ])]),
        html.Div([
            html.Div([dark_card("Bilateral Energy Trade Map",[dark_graph("e-d4-map",460)],sector=SEC, info_id="Bilateral Energy Trade Map")],style={"flex":"1","minWidth":"0"}),
            html.Div(id="e-d4-country-panel",style={"width":"240px","flexShrink":"0","background":BG_CARD,"border":f"1px solid {BORDER}","borderRadius":"8px","padding":"14px","overflowY":"auto","maxHeight":"500px"}),
        ],style={"display":"flex","gap":"14px","marginBottom":"14px"}),
        html.Div(id="e-d4-kpi",style={"display":"flex","gap":"12px"}),
        raw_data_section("e-d4",[
            ("Comtrade Trade Data",comtrade.head(500) if not comtrade.empty else None,
             ["refyear","flowcode","partnerdesc","cmddesc","primaryvalue","partner_lat","partner_lon"]),
        ]),
    ])

@app.callback(Output("e-d4-map","figure"),Output("e-d4-kpi","children"),Output("e-d4-country-panel","children"),
              Input("url","pathname"),Input("e-d4-year","value"),Input("e-d4-flow","value"))
def cb_e_d4(pathname,years,flow):
    year=resolve_year(years); flow=flow or "Both"; ct_year=ct_nearest(year)
    ct_y=comtrade[comtrade["refyear"].isin([ct_year])] if not comtrade.empty else pd.DataFrame()
    IRELAND_LAT,IRELAND_LON=53.35,-6.26
    fig=go.Figure()
    fig.update_layout(geo=dict(showframe=False,showcoastlines=True,showland=True,showocean=True,showlakes=False,
                                landcolor="#1c2128",oceancolor="#0d1117",coastlinecolor="#30363d",coastlinewidth=0.5,
                                projection_type="natural earth",bgcolor="rgba(0,0,0,0)"),
                      paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=0,b=0))
    if ct_y.empty:
        return fig,[kri_card("No Data","N/A","",TEXT_MUTED,sector=SEC)],html.Div("No data",style={"color":TEXT_MUTED,"fontSize":"13px"})
    imp_all=ct_y[ct_y["flowcode"]=="M"]; exp_all=ct_y[ct_y["flowcode"]=="X"]
    def add_arcs(df_flow,rgb,name,is_export=False):
        if df_flow.empty or "partner_lat" not in df_flow.columns: return
        g=df_flow.groupby(["partnerdesc","partner_lat","partner_lon"])["primaryvalue"].sum().reset_index()
        if g.empty: return
        max_val=g["primaryvalue"].max() or 1
        for _,r in g.iterrows():
            if r["partner_lat"]==0 and r["partner_lon"]==0: continue
            raw_w=r["primaryvalue"]/max_val
            w=max(2.5,min(8,raw_w*8)) if is_export else max(1.5,min(10,raw_w*10))
            op=max(0.6,min(1.0,raw_w+0.4)) if is_export else max(0.35,min(1.0,raw_w))
            rv,gv,bv=int(rgb[0]),int(rgb[1]),int(rgb[2])
            fig.add_trace(go.Scattergeo(lat=[r["partner_lat"],IRELAND_LAT],lon=[r["partner_lon"],IRELAND_LON],
                mode="lines",showlegend=False,line=dict(width=w,color=f"rgba({rv},{gv},{bv},{op:.2f})",dash="dot" if is_export else "solid"),
                hovertemplate=f"<b>{r['partnerdesc']}</b><br>{name}: {fmt_val(r['primaryvalue'])}<extra></extra>"))
            fig.add_trace(go.Scattergeo(lat=[r["partner_lat"]],lon=[r["partner_lon"]],mode="markers",showlegend=False,
                marker=dict(size=max(5,w*1.3),color=f"rgba({rv},{gv},{bv},0.9)",line=dict(color=BG_CARD,width=1)),
                text=f"{r['partnerdesc']}: {fmt_val(r['primaryvalue'])}",hoverinfo="text"))
    if flow in ("Both","M"): add_arcs(imp_all,(225,29,72),"Imports",False)
    if flow in ("Both","X"): add_arcs(exp_all,(57,211,83),"Exports",True)
    fig.add_trace(go.Scattergeo(lat=[IRELAND_LAT],lon=[IRELAND_LON],mode="markers+text",
        marker=dict(size=14,color=ACCENT,line=dict(color=WHITE,width=2)),
        text=["Ireland"],textposition="top center",textfont=dict(color=TEXT_PRI,size=12),showlegend=False))
    if flow in ("Both","M"): fig.add_trace(go.Scattergeo(lat=[None],lon=[None],mode="lines",name="Import (>> Ireland)",line=dict(color="rgba(225,29,72,1)",width=3),showlegend=True))
    if flow in ("Both","X"): fig.add_trace(go.Scattergeo(lat=[None],lon=[None],mode="lines",name="Export (Ireland >>)",line=dict(color="rgba(57,211,83,1)",width=3,dash="dot"),showlegend=True))
    fig.update_layout(legend=dict(bgcolor="rgba(22,27,34,0.9)",bordercolor=BORDER,borderwidth=1,font=dict(color=TEXT_PRI,size=12),x=0.01,y=0.99),font=dict(color=TEXT_PRI))
    top_imp="N/A"; top_imp_val=""
    if not imp_all.empty:
        g_imp=imp_all.groupby("partnerdesc")["primaryvalue"].sum(); top_imp=g_imp.idxmax(); top_imp_val=fmt_val(g_imp.max())
    kpi=html.Div([
        kri_card("Total Imports",fmt_val(imp_all["primaryvalue"].sum()) if not imp_all.empty else "N/A","",C_RED,f"Year {ct_year}",SEC),
        kri_card("Total Exports",fmt_val(exp_all["primaryvalue"].sum()) if not exp_all.empty else "N/A","",C_GREEN,f"Year {ct_year}",SEC),
        kri_card("Top Import Origin",top_imp,"",ACCENT2,top_imp_val,SEC),
        kri_card("Data Source","UN Comtrade","",TEXT_MUTED,"HS 2709/2710/2711/2716",SEC),
    ],style={"display":"flex","gap":"12px"})
    def country_rows(df,colour,label):
        if df.empty: return []
        g=df.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False); total=g.sum() or 1
        rows=[html.Div([html.Span(label,style={"fontSize":"9px","color":colour,"letterSpacing":"0.5px","fontWeight":"600"}),
                        html.Span(f"({fmt_val(g.sum())} total)",style={"fontSize":"9px","color":TEXT_MUTED,"marginLeft":"6px"})],
                       style={"padding":"6px 0 4px","borderBottom":f"1px solid {BORDER}","marginBottom":"4px"})]
        for partner,val in g.head(10).items():
            bar_w=min(100,val/g.iloc[0]*100) if g.iloc[0]>0 else 0
            rows.append(html.Div([
                html.Div([html.Span("o",style={"color":colour,"fontSize":"8px","marginRight":"4px","fontWeight":"bold"}),
                          html.Span(partner,style={"fontSize":"11px","color":TEXT_PRI,"flex":"1","overflow":"hidden","textOverflow":"ellipsis","whiteSpace":"nowrap"}),
                          html.Span(fmt_val(val),style={"fontSize":"11px","color":colour,"fontWeight":"600","flexShrink":"0","marginLeft":"6px"})],
                         style={"display":"flex","alignItems":"center","marginBottom":"3px"}),
                html.Div(style={"background":BG_INPUT,"borderRadius":"2px","height":"4px","overflow":"hidden","marginBottom":"6px"},
                         children=[html.Div(style={"width":f"{bar_w:.0f}%","height":"100%","background":colour,"borderRadius":"2px"})]),
            ]))
        return rows
    imp_rows=country_rows(imp_all,C_RED,"IMPORTS") if flow in ("Both","M") else []
    exp_rows=country_rows(exp_all,C_GREEN,"EXPORTS") if flow in ("Both","X") else []
    panel=html.Div([
        html.Div([html.Span(f"Year {ct_year}",style={"fontSize":"11px","color":TEXT_PRI,"fontWeight":"600"}),
                  html.Span(" . Trade Partners",style={"fontSize":"10px","color":TEXT_MUTED})],
                 style={"marginBottom":"10px","paddingBottom":"8px","borderBottom":f"1px solid {BORDER}"}),
        *imp_rows,*exp_rows,
    ])
    return fig,kpi,panel

make_raw_toggle("e-d4-raw-toggle-btn","e-d4-raw-panel","e-d4-raw-content","e-d4-raw-source-tabs",[
    ("Comtrade Trade Data",comtrade.head(500) if not comtrade.empty else None,
     ["refyear","flowcode","partnerdesc","cmddesc","primaryvalue","partner_lat","partner_lon"]),
])


# ?? E-D6: Stress Testing ???????????????????????????????????????????????????????
def e_d6_layout():
    return html.Div([
        e_header("Energy -- Stress Testing",
                 "Supplier removal impact -- supply lost by product, alternative sourcing options",
                 [yr_dropdown("e-d6-year",multi=True,default=2023),
                  html.Div([
                      html.Label("Remove Country",style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="e-d6-country",options=[{"label":p,"value":p} for p in CT_PARTNERS if p!="World"],
                                   value="United Kingdom",clearable=False,
                                   style={"fontSize":"13px","width":"220px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {BORDER}"}),
                  ])]),
        html.Div(id="e-d6-cards",style={"display":"flex","gap":"12px","marginBottom":"16px"}),
        row(dark_card("Supply Lost by Product (%)",[dark_graph("e-d6-bar",280)],flex="2",sector=SEC, info_id="Supply Lost by Product (%)"),
            dark_card("Top Alternative Suppliers",[html.Div(id="e-d6-alts")],flex="1",sector=SEC, info_id="Top Alternative Suppliers")),
        dark_card("Detailed Impact Table",[html.Div(id="e-d6-table")],sector=SEC, info_id="Detailed Impact Table"),
        raw_data_section("e-d6",[
            ("Comtrade Imports",comtrade[comtrade["flowcode"]=="M"].head(500) if not comtrade.empty else None,
             ["refyear","partnerdesc","cmddesc","primaryvalue"]),
        ]),
    ])

@app.callback(Output("e-d6-cards","children"),Output("e-d6-bar","figure"),
              Output("e-d6-alts","children"),Output("e-d6-table","children"),
              Input("url","pathname"),Input("e-d6-year","value"),Input("e-d6-country","value"))
def cb_e_d6(pathname,years,country):
    year=resolve_year(years)
    if not country: country="United Kingdom"
    if comtrade.empty: return [],go.Figure(),html.P("No data"),html.P("No data")
    ct=comtrade[(comtrade["refyear"]==ct_nearest(year))&(comtrade["flowcode"]=="M")]
    if ct.empty: return [],go.Figure(),html.P("No data"),html.P("No data")
    results=[]
    for prod in ct["cmddesc"].unique():
        pdf=ct[ct["cmddesc"]==prod]; tv=pdf["primaryvalue"].sum()
        if tv==0: continue
        cv=pdf[pdf["partnerdesc"]==country]["primaryvalue"].sum(); ls=cv/tv*100
        alts=pdf[pdf["partnerdesc"]!=country].groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(3)
        results.append({"product":prod,"total":tv,"lost_val":cv,"lost_share":round(ls,1),
                        "impact":stress_label(ls),"colour":stress_colour(ls),
                        "alts":", ".join(f"{p} ({v/tv*100:.0f}%)" for p,v in alts.items())})
    results=sorted(results,key=lambda x:x["lost_share"],reverse=True)
    var=sum(r["lost_val"] for r in results); crit=sum(1 for r in results if r["impact"]=="CRITICAL")
    sev=sum(1 for r in results if r["impact"]=="SEVERE")
    cards=html.Div([
        kri_card("Products Affected",str(sum(1 for r in results if r.get("lost_share",0)>0)),"",C_GREEN if sum(1 for r in results if r.get("lost_share",0)>0)==0 else ACCENT2,"Products with >0% supply from this country",SEC),
        kri_card("Critical Impact",str(crit),"",C_RED,">70% supply lost",SEC),
        kri_card("Severe Impact",str(sev),"",C_ORANGE,"40-70% supply lost",SEC),
        kri_card("Max Potential Loss",fmt_val(var),"",C_RED,f"From {country} (assumes no substitution)",SEC),
    ],style={"display":"flex","gap":"12px"})
    fig=go.Figure(go.Bar(y=[r["product"][:35] for r in results],x=[r["lost_share"] for r in results],
        orientation="h",marker_color=[r["colour"] for r in results],
        text=[f"{r['lost_share']}%  {r['impact']}" for r in results],textposition="inside",insidetextanchor="middle",
        textfont=dict(color=TEXT_PRI,size=10)))
    fig.add_vline(x=70,line_dash="dash",line_color=C_RED,opacity=0.7,annotation_text="CRITICAL 70%",annotation_font=dict(color=C_RED,size=10))
    fig.add_vline(x=40,line_dash="dot",line_color=C_ORANGE,opacity=0.7,annotation_text="SEVERE 40%",annotation_font=dict(color=C_ORANGE,size=10))
    themed_layout(fig,SEC,False)
    fig.update_xaxes(title=dict(text="Supply Lost if Country Removed (%)",font=dict(color=TEXT_SEC,size=11),standoff=12),range=[0,120],ticksuffix="%")
    fig.update_yaxes(title=dict(text="Energy Product",font=dict(color=TEXT_SEC,size=11),standoff=12))
    fig
    all_alts=ct[ct["partnerdesc"]!=country].groupby("partnerdesc")["primaryvalue"].sum(); total_imp=ct["primaryvalue"].sum() or 1
    alt_els=[]
    for p,v in all_alts.sort_values(ascending=False).head(6).items():
        pct=v/total_imp*100; bar_w=min(pct/40*100,100)
        alt_els.append(html.Div([
            html.Div([html.Span(p,style={"color":TEXT_PRI,"fontSize":"12px","flex":"1"}),
                      html.Span(f"{pct:.1f}%",style={"color":C_GREEN,"fontSize":"12px","fontWeight":"600"})],
                     style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"}),
            html.Div(style={"background":BG_INPUT,"borderRadius":"3px","height":"8px","overflow":"hidden","marginBottom":"8px"},
                     children=[html.Div(style={"width":f"{bar_w:.0f}%","height":"100%","background":C_GREEN,"borderRadius":"3px"})]),
        ]))
    tbl=dash_table.DataTable(
        data=[{"Product":r["product"],"Lost %":f"{r['lost_share']}%","Lost ?":fmt_val(r["lost_val"]),"Impact":r["impact"],"Alt. Suppliers":r["alts"]} for r in results],
        columns=[{"name":c,"id":c} for c in ["Product","Lost %","Lost ?","Impact","Alt. Suppliers"]],
        style_table={"overflowX":"auto","maxHeight":"250px","overflowY":"auto"},
        style_header={"backgroundColor":"#21262d","color":TEXT_PRI,"fontWeight":"600","fontSize":"12px","border":f"1px solid {BORDER}"},
        style_cell={"padding":"9px 12px","fontSize":"12px","fontFamily":"Inter,Segoe UI,Arial","backgroundColor":BG_CARD,"color":TEXT_PRI,"border":f"1px solid {BORDER}"},
        style_data_conditional=[
            {"if":{"filter_query":'{Impact} = "CRITICAL"'},"backgroundColor":"#2d1a1a","color":C_RED,"fontWeight":"700"},
            {"if":{"filter_query":'{Impact} = "SEVERE"'},"backgroundColor":"#2d2116","color":C_ORANGE},
            {"if":{"filter_query":'{Impact} = "SIGNIFICANT"'},"backgroundColor":"#2a2416","color":C_AMBER},
        ])
    return cards,fig,alt_els,tbl

make_raw_toggle("e-d6-raw-toggle-btn","e-d6-raw-panel","e-d6-raw-content","e-d6-raw-source-tabs",[
    ("Comtrade Imports",comtrade[comtrade["flowcode"]=="M"].head(500) if not comtrade.empty else None,
     ["refyear","partnerdesc","cmddesc","primaryvalue"]),
])


# ?? E-D7: Scenario War Room ????????????????????????????????????????????????????
def e_d7_layout():
    def slider_row(label,sid,min_v,max_v,step,val,suffix=""):
        return html.Div([
            html.Div([html.Span(label,style={"fontSize":"12px","color":TEXT_SEC}),
                      html.Span(id=f"{sid}-out",style={"fontSize":"13px","color":ACCENT,"fontWeight":"600","marginLeft":"8px"})],
                     style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"}),
            dcc.Slider(id=sid,min=min_v,max=max_v,step=step,value=val,
                       marks={min_v:{"label":f"{min_v}{suffix}","style":{"color":TEXT_MUTED,"fontSize":"10px"}},
                              max_v:{"label":f"{max_v}{suffix}","style":{"color":TEXT_MUTED,"fontSize":"10px"}}},
                       tooltip={"placement":"bottom","always_visible":False}),
        ],style={"marginBottom":"16px"})
    return html.Div([
        e_header("Energy -- Scenario War Room",
                 "Monte Carlo probabilistic simulation -- supply disruption ? demand surge value at risk",
                 []),
        row(html.Div([
                dark_card("Scenario Parameters",[
                    slider_row("Disruption Severity (%)","e-d7-sev",0,100,1,80,"%"),
                    slider_row("Demand Surge Factor","e-d7-dem",1.0,3.0,0.1,1.5,"?"),
                    slider_row("Monte Carlo Iterations","e-d7-iters",100,2000,100,1000,""),
                ],sector=SEC, info_id="Scenario Parameters"),
                html.Div(id="e-d7-kpi",style={"display":"flex","gap":"12px","marginTop":"12px"}),
            ],style={"flex":"1"}),
            dark_card("Potential Loss Distribution",[dark_graph("e-d7-hist",300)],flex="2",sector=SEC, info_id="VaR Distribution")),
        row(dark_card("12-Month Impact Timeline",[dark_graph("e-d7-tl",220)],flex="1",sector=SEC, info_id="12-Month Impact Timeline"),
            dark_card("Impact by Product",[dark_graph("e-d7-prod",220)],flex="1",sector=SEC, info_id="Impact by Product")),
        dark_card("Percentile Summary Table",[html.Div(id="e-d7-ptbl")],sector=SEC, info_id="Percentile Summary Table"),
        html.Div("Scenario assumes no short-term substitution from alternative suppliers. Real-world impact is typically lower as reshoring and rerouting mitigate some losses over 3–12 months.",
                 style={"fontSize":"11px","color":TEXT_MUTED,"marginTop":"12px","fontStyle":"italic","textAlign":"center"}),
        raw_data_section("e-d7",[
            ("Comtrade Imports (base for Monte Carlo)",
             comtrade[(comtrade["refyear"]==CT_YEAR_MAX)&(comtrade["flowcode"]=="M")].head(500) if not comtrade.empty else None,
             ["refyear","partnerdesc","cmddesc","primaryvalue"]),
            ("SEAI Imports",
             seai[seai["flow"]=="Imports"].head(200) if not seai.empty else None,
             ["year","flow","oil","natural_gas","electricity","total"]),
        ]),
    ])

@app.callback(Output("e-d7-sev-out","children"),Input("e-d7-sev","value"))
def upd_e_sev(v): return f"{v or 0}%"
@app.callback(Output("e-d7-dem-out","children"),Input("e-d7-dem","value"))
def upd_e_dem(v): return f"{v or 1:.1f}x"
@app.callback(Output("e-d7-iters-out","children"),Input("e-d7-iters","value"))
def upd_e_iters(v): return f"{v or 1000:,}"

@app.callback(Output("e-d7-kpi","children"),Output("e-d7-hist","figure"),
              Output("e-d7-tl","figure"),Output("e-d7-prod","figure"),Output("e-d7-ptbl","children"),
              Input("e-d7-sev","value"),Input("e-d7-dem","value"),Input("e-d7-iters","value"))
def cb_e_d7(severity,demand,iters):
    sev=(severity or 80)/100; dem=demand or 1.5; n=iters or 1000
    np.random.seed(42)
    # Fix #6: derive base from live data, not hardcoded 9.5e9.
    # Mirrors the pattern the MedTech War Room already uses.
    if not comtrade.empty and CT_YEARS:
        base = comtrade[(comtrade["refyear"]==CT_YEAR_MAX)&(comtrade["flowcode"]=="M")]["primaryvalue"].sum()
    else:
        base = 9.5e9  # fallback when no data
    base = base if base > 0 else 9.5e9
    samples=[base*sev*np.random.uniform(0.7,1.3)+base*(dem-1)*np.random.uniform(0.8,1.2) for _ in range(n)]
    arr=np.array(samples)/1e9
    p5,p25,p50,p75,p95=np.percentile(arr,[5,25,50,75,95])
    imp_pct=round(p50/(base/1e9)*100,1)
    kpis=html.Div([
        # Fix #13: rename "Value at Risk" to "Median Potential Loss" (more accurate)
        kri_card("Median Potential Loss",f"EUR {p50:.2f}","bn",C_RED if p50>5 else C_ORANGE,"P50 most likely outcome",SEC),
        kri_card("Import Impact",f"{imp_pct}","%",C_RED if imp_pct>50 else C_ORANGE,"% of annual imports",SEC),
        kri_card("P95 Worst Case",f"EUR {p95:.2f}","bn",C_RED,"Only 5% worse than this",SEC),
        kri_card("P5 Best Case",f"EUR {p5:.2f}","bn",C_GREEN,"Only 5% better than this",SEC),
    ],style={"display":"flex","gap":"12px"})
    fig_h=go.Figure(go.Histogram(x=arr,nbinsx=40,marker_color=ACCENT2,opacity=0.75,name=f"Simulated outcomes (n={n:,})"))
    for val,lbl,col in [(p5,"P5",C_GREEN),(p50,"P50",C_AMBER),(p95,"P95",C_RED)]:
        fig_h.add_vline(x=val,line_dash="dash",line_color=col,annotation_text=f"{lbl}: EUR {val:.2f}bn",annotation_position="top",annotation_font=dict(color=col,size=11))
    themed_layout(fig_h,SEC,True)
    fig_h.update_xaxes(title=dict(text=f"Potential Loss (EUR Billion) | n={n:,} iterations",font=dict(color=TEXT_SEC,size=11)))
    fig_h.update_yaxes(title=dict(text="Number of Simulations",font=dict(color=TEXT_SEC,size=11),standoff=12))
    months=list(range(1,13)); np.random.seed(42)
    mi=[p50*(1-np.exp(-0.3*m))*np.random.uniform(0.9,1.1) for m in months]
    fig_tl=go.Figure(go.Scatter(x=months,y=mi,mode="lines+markers",fill="tozeroy",
        fillcolor=hex_rgba(C_RED,0.12),line=dict(color=C_RED,width=2.5),marker=dict(size=6,color=C_RED),name="Cumulative impact"))
    themed_layout(fig_tl,SEC,True)
    fig_tl.update_xaxes(title=dict(text="Month After Disruption",font=dict(color=TEXT_SEC,size=11),standoff=12),tickvals=months,ticktext=[f"M{m}" for m in months])
    fig_tl.update_yaxes(title=dict(text="Cumulative Potential Loss (EUR bn)",font=dict(color=TEXT_SEC,size=11),standoff=12))
    fig_tl.update_layout(legend=dict(orientation="h",y=-0.30,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=11)))
    # Fix #7: Impact-by-Product derived from actual product shares, not np.random
    prod_shares = {}
    if not comtrade.empty and CT_YEARS:
        ct_l = comtrade[(comtrade["refyear"]==CT_YEAR_MAX)&(comtrade["flowcode"]=="M")]
        # Group by product category using FUEL_HS_PATTERNS
        for fuel in IMP_FUELS:
            fuel_rows = ct_l[ct_l["cmddesc"].apply(lambda v: _matches_fuel(v, fuel))]
            prod_shares[fuel.replace("_"," ").title()] = fuel_rows["primaryvalue"].sum()
        # Collect rows that didn't match any fuel category as "Other"
        matched_mask = ct_l["cmddesc"].apply(
            lambda v: any(_matches_fuel(v, f) for f in IMP_FUELS))
        prod_shares["Other"] = ct_l[~matched_mask]["primaryvalue"].sum()
    tot_prod = sum(prod_shares.values()) or 1
    # Apply median potential loss proportionally to each product's share
    prod_list = sorted(prod_shares.items(), key=lambda x: x[1], reverse=True)
    prods = [p for p, _ in prod_list]
    pi = [p50 * (v / tot_prod) for _, v in prod_list]
    fig_p=go.Figure(go.Bar(y=prods,x=pi,orientation="h",marker_color=[stress_colour(v/p50*100) for v in pi],
        text=[f"EUR {v:.2f}bn" for v in pi],textposition="outside",textfont=dict(color=TEXT_PRI,size=11),name="Estimated impact"))
    themed_layout(fig_p,SEC,False)
    fig_p.update_xaxes(title=dict(text="Estimated Loss (EUR Billion)",font=dict(color=TEXT_SEC,size=11),standoff=12))
    fig_p.update_yaxes(title=dict(text="Energy Product",font=dict(color=TEXT_SEC,size=11),standoff=12))
    fig_p
    ptbl=dash_table.DataTable(
        data=[{"Percentile":l,"Potential Loss":f"EUR {v:.3f}bn","Interpretation":i} for l,v,i in
              [("P5 (Best Case)",p5,"Only 5% of scenarios are lower"),("P25",p25,"25% of scenarios are lower"),
               ("P50 (Median)",p50,"Most likely outcome"),("P75",p75,"75% of scenarios are lower"),("P95 (Worst Case)",p95,"Only 5% of scenarios are worse")]],
        columns=[{"name":c,"id":c} for c in ["Percentile","Potential Loss","Interpretation"]],
        style_header={"backgroundColor":"#21262d","color":TEXT_PRI,"fontWeight":"600","fontSize":"12px","border":f"1px solid {BORDER}"},
        style_cell={"padding":"10px 14px","fontSize":"12px","fontFamily":"Inter,Segoe UI,Arial","backgroundColor":BG_CARD,"color":TEXT_PRI,"border":f"1px solid {BORDER}"},
        style_data_conditional=[
            {"if":{"row_index":4},"backgroundColor":"#2d1a1a","color":C_RED},
            {"if":{"row_index":2},"backgroundColor":"#2a2416","color":C_AMBER},
            {"if":{"row_index":0},"backgroundColor":"#1a2d1e","color":C_GREEN},
        ])
    return kpis,fig_h,fig_tl,fig_p,ptbl

# ?? E-D10: ML Analysis ?????????????????????????????????????????????????????????
def e_d10_layout():
    return html.Div([
        e_header("Energy -- Machine Learning Analysis",
                 "Anomaly detection . K-Means clustering . HHI forecast to 2030 . Cascade failure simulation",[]),
        row(dark_card("Anomaly Detection -- Trade Flow Outliers",[dark_graph("e-d10-anomaly",260)],flex="1",sector=SEC, info_id="Anomaly Detection -- Trade Flow Outliers"),
            dark_card("K-Means Clustering -- Fuel Risk Groups",[dark_graph("e-d10-cluster",260)],flex="1",sector=SEC, info_id="K-Means Clustering -- Fuel Risk Groups")),
        row(dark_card("HHI Trend & Forecast to 2030",[dark_graph("e-d10-forecast",240)],flex="1",sector=SEC, info_id="HHI Trend & Forecast to 2030"),
            dark_card("Cascade Failure -- Supplier Removal Simulation",[dark_graph("e-d10-cascade",240)],flex="1",sector=SEC, info_id="Cascade Failure -- Supplier Removal Simulation")),
        raw_data_section("e-d10",[
            ("SEAI Historical",seai.head(500) if not seai.empty else None,["year","flow","oil","natural_gas","electricity","total"]),
            ("Comtrade Latest",comtrade[comtrade["refyear"]==CT_YEAR_MAX].head(300) if not comtrade.empty else None,["refyear","flowcode","partnerdesc","cmddesc","primaryvalue"]),
        ]),
    ])

@app.callback(Output("e-d10-anomaly","figure"),Output("e-d10-cluster","figure"),
              Output("e-d10-forecast","figure"),Output("e-d10-cascade","figure"),
              Input("url","pathname"),prevent_initial_call=False)
def cb_e_d10(_):
    """Fix #5-C: Real machine learning (IsolationForest + KMeans) on actual trade
    data. Replaces previous implementation that used np.random.uniform as a stub.
    """
    # ── 1. Anomaly Detection: IsolationForest on (value, yoy_change, partner_share) ──
    fig_a = go.Figure()
    if not comtrade.empty and len(CT_YEARS) >= 2:
        latest = CT_YEAR_MAX
        prior = sorted([y for y in CT_YEARS if y < latest])[-1] if len([y for y in CT_YEARS if y < latest]) else latest
        cur = comtrade[(comtrade["refyear"]==latest)&(comtrade["flowcode"]=="M")]
        prev = comtrade[(comtrade["refyear"]==prior)&(comtrade["flowcode"]=="M")]
        cur_g = cur.groupby(["partnerdesc","cmddesc"])["primaryvalue"].sum().reset_index()
        prev_g = prev.groupby(["partnerdesc","cmddesc"])["primaryvalue"].sum().reset_index()
        prev_g.columns = ["partnerdesc","cmddesc","prev_value"]
        g = cur_g.merge(prev_g, on=["partnerdesc","cmddesc"], how="left").fillna({"prev_value":0})
        g.columns = ["partner","product","value","prev_value"]
        total_cur = g["value"].sum() or 1
        g["partner_share"] = g["value"] / total_cur
        g["yoy_change"] = np.where(g["prev_value"] > 0,
                                    (g["value"] - g["prev_value"]) / g["prev_value"],
                                    0.0).clip(-5, 5)
        # Build feature matrix for IsolationForest
        if len(g) >= 10:
            X = g[["value", "yoy_change", "partner_share"]].copy()
            X["value_log"] = np.log1p(X["value"])
            features = X[["value_log", "yoy_change", "partner_share"]].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(features)
            iso = IsolationForest(contamination=0.10, random_state=42, n_estimators=100)
            g["iso_score"] = iso.fit_predict(Xs)   # -1 = anomaly, 1 = normal
            g["anomaly_raw"] = iso.score_samples(Xs)  # continuous score
            g["is_anomaly"] = g["iso_score"] == -1
        else:
            g["anomaly_raw"] = 0.0
            g["is_anomaly"] = False

        # Plot
        normal = g[~g["is_anomaly"]]
        anoms = g[g["is_anomaly"]]
        fig_a.add_trace(go.Scatter(
            x=normal["value"]/1e9, y=normal["anomaly_raw"], mode="markers", name="Normal",
            marker=dict(color=ACCENT2, size=7, opacity=0.7),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>EUR %{x:.2f}bn<br>Score %{y:.3f}<extra></extra>",
            customdata=normal[["partner","product"]].values))
        fig_a.add_trace(go.Scatter(
            x=anoms["value"]/1e9, y=anoms["anomaly_raw"], mode="markers", name="Anomaly",
            marker=dict(color=C_RED, size=11, symbol="x", line=dict(width=2)),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>EUR %{x:.2f}bn<br>Score %{y:.3f}<extra></extra>",
            customdata=anoms[["partner","product"]].values))
    themed_layout(fig_a, SEC, True)
    fig_a.update_xaxes(title=dict(text="Trade Value (EUR Billion)", font=dict(color=TEXT_SEC,size=11), standoff=12))
    fig_a.update_yaxes(title=dict(text="IsolationForest anomaly score (lower = more anomalous)", font=dict(color=TEXT_SEC,size=11), standoff=12))
    fig_a.update_layout(legend=dict(orientation="h", y=-0.30, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)))

    # ── 2. K-Means Clustering: (HHI, supplier_count, volatility) per fuel ──
    fig_cl = go.Figure()
    fuel_features = []
    for fuel in IMP_FUELS:
        if comtrade.empty: continue
        fuel_rows = comtrade[(comtrade["flowcode"]=="M") &
                              (comtrade["cmddesc"].apply(lambda v: _matches_fuel(v, fuel)))]
        if fuel_rows.empty: continue
        # Latest-year country HHI
        latest = fuel_rows[fuel_rows["refyear"]==CT_YEAR_MAX]
        if latest.empty: continue
        by_country = latest.groupby("partnerdesc")["primaryvalue"].sum()
        tot = by_country.sum()
        if tot <= 0: continue
        hhi = float(((by_country / tot) ** 2).sum())
        n_sup = int((by_country > 0).sum())
        # Volatility = std of total yearly value over last 10 years
        yr_totals = fuel_rows.groupby("refyear")["primaryvalue"].sum()
        vol = float(yr_totals.pct_change().dropna().std()) if len(yr_totals) >= 3 else 0.0
        fuel_features.append({"fuel": fuel.replace("_"," ").title(), "hhi": hhi, "n_sup": n_sup, "volatility": vol})

    if len(fuel_features) >= 2:
        df_cl = pd.DataFrame(fuel_features)
        k = min(3, len(df_cl))
        feat = df_cl[["hhi", "n_sup", "volatility"]].copy()
        feat["n_sup_scaled"] = feat["n_sup"] / (feat["n_sup"].max() or 1)
        Xc = feat[["hhi", "n_sup_scaled", "volatility"]].values
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        df_cl["cluster"] = km.fit_predict(Xc)
        # Interpret clusters by their centroid HHI (low/mid/high)
        centroids = pd.DataFrame(km.cluster_centers_, columns=["hhi","n_sup_scaled","volatility"])
        order = centroids["hhi"].argsort().values
        label_map = {order[0]:"Low Risk", order[-1]:"High Risk"}
        if k==3: label_map[order[1]] = "Medium Risk"
        df_cl["cluster_label"] = df_cl["cluster"].map(label_map).fillna("Medium Risk")
        col_map = {"Low Risk": C_GREEN, "Medium Risk": C_AMBER, "High Risk": C_RED}
        for lbl in ["Low Risk","Medium Risk","High Risk"]:
            s = df_cl[df_cl["cluster_label"]==lbl]
            if s.empty: continue
            fig_cl.add_trace(go.Scatter(x=s["hhi"], y=s["n_sup"], mode="markers+text",
                name=lbl, text=s["fuel"], textposition="top center",
                textfont=dict(size=11, color=TEXT_PRI),
                marker=dict(size=18, color=col_map[lbl], opacity=0.85, line=dict(color=BG_CARD, width=1.5))))
    themed_layout(fig_cl, SEC, True)
    fig_cl.update_xaxes(title=dict(text="Country-HHI (supplier concentration)", font=dict(color=TEXT_SEC,size=11), standoff=12))
    fig_cl.update_yaxes(title=dict(text="Number of Supplier Countries", font=dict(color=TEXT_SEC,size=11), standoff=12))
    fig_cl.update_layout(legend=dict(orientation="h", y=-0.30, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)))

    # ── 3. HHI trend + linear forecast to 2030 (weighted country-HHI) ──
    hhi_hist = []
    for y in YEARS:
        hhi_y = compute_hhi(y).get("weighted", 0.0)
        if hhi_y > 0:
            hhi_hist.append({"year": y, "hhi": hhi_y})
    df_hhi = pd.DataFrame(hhi_hist); fig_fc = go.Figure()
    if len(df_hhi) > 2:
        x = df_hhi["year"].values; y_ = df_hhi["hhi"].values
        m, b = np.polyfit(x, y_, 1)
        fy = list(range(int(max(x))+1, 2031))
        fv = [m*yr + b for yr in fy]
        fig_fc.add_trace(go.Scatter(x=df_hhi["year"], y=df_hhi["hhi"], mode="lines+markers",
            name="Actual HHI", line=dict(color=ACCENT2, width=2.5), marker=dict(size=5, color=ACCENT2)))
        fig_fc.add_trace(go.Scatter(x=fy, y=fv, mode="lines",
            name="Linear Forecast to 2030",
            line=dict(color=C_RED if m > 0 else C_GREEN, width=2, dash="dash")))
        fig_fc.add_hline(y=0.40, line_dash="dot", line_color=C_RED, opacity=0.6,
            annotation_text="Critical threshold", annotation_font=dict(color=C_RED, size=10))
    themed_layout(fig_fc, SEC, True)
    fig_fc.update_xaxes(title=dict(text="Year", font=dict(color=TEXT_SEC,size=11), standoff=12))
    fig_fc.update_yaxes(title=dict(text="Weighted Country-HHI", font=dict(color=TEXT_SEC,size=11), standoff=12))
    fig_fc.update_layout(legend=dict(orientation="h", y=-0.30, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)))

    # ── 4. Cascade failure: remove partners in order of highest supply share ──
    # Dynamic: order determined by actual import share, not hardcoded list
    cas = []
    if not comtrade.empty and CT_YEARS:
        ct_l = comtrade[(comtrade["refyear"]==CT_YEAR_MAX)&(comtrade["flowcode"]=="M")]
        tv = ct_l["primaryvalue"].sum() or 1
        # Top 10 suppliers by value for the cascade
        ranked = (ct_l.groupby("partnerdesc")["primaryvalue"].sum()
                  .sort_values(ascending=False).head(10))
        cum = 0
        for i, (p, v) in enumerate(ranked.items()):
            cum += v / tv * 100
            cas.append({"round": i+1, "partner": p, "cumulative": min(cum, 100)})
    df_cas = pd.DataFrame(cas) if cas else pd.DataFrame({"round":[],"partner":[],"cumulative":[]})
    fig_cas = go.Figure()
    if not df_cas.empty:
        fig_cas.add_trace(go.Scatter(x=df_cas["round"], y=df_cas["cumulative"],
            mode="lines+markers+text", text=df_cas["partner"], textposition="top right",
            textfont=dict(size=9, color=TEXT_SEC),
            line=dict(color=C_RED, width=2.5), marker=dict(color=C_RED, size=8),
            fill="tozeroy", fillcolor=hex_rgba(C_RED, 0.08),
            name="Cumulative supply lost (%)"))
        fig_cas.add_hline(y=70, line_dash="dash", line_color=C_RED, opacity=0.7,
            annotation_text="Critical failure (70%)", annotation_font=dict(color=C_RED, size=10))
    themed_layout(fig_cas, SEC, True)
    fig_cas.update_xaxes(title=dict(text="Removal Round (largest supplier first)",
        font=dict(color=TEXT_SEC,size=11), standoff=12),
        tickvals=df_cas["round"].tolist() if not df_cas.empty else [1])
    fig_cas.update_yaxes(title=dict(text="Cumulative Supply Lost (%)",
        font=dict(color=TEXT_SEC,size=11), standoff=12), range=[0, 105])
    fig_cas.update_layout(legend=dict(orientation="h", y=-0.30, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=10, t=10, b=60))
    return fig_a, fig_cl, fig_fc, fig_cas

make_raw_toggle("e-d10-raw-toggle-btn","e-d10-raw-panel","e-d10-raw-content","e-d10-raw-source-tabs",[
    ("SEAI Historical",seai.head(500) if not seai.empty else None,["year","flow","oil","natural_gas","electricity","total"]),
    ("Comtrade Latest",comtrade[comtrade["refyear"]==CT_YEAR_MAX].head(300) if not comtrade.empty else None,["refyear","flowcode","partnerdesc","cmddesc","primaryvalue"]),
])


# ?? E-D14: Export Dependency ???????????????????????????????????????????????????
def e_d14_layout():
    return html.Div([
        e_header("Energy -- Export Dependency Analysis",
                 "Export market concentration, tariff impact modelling, vulnerability matrix",
                 [yr_dropdown("e-d14-year",multi=True,default=2023),
                  html.Div([
                      html.Label("Tariff Rate",style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="e-d14-tariff",
                          options=[{"label":"0% (baseline)","value":0.0},{"label":"10%","value":0.10},{"label":"25%","value":0.25},{"label":"50%","value":0.50}],
                          value=0.25,clearable=False,
                          style={"fontSize":"13px","width":"160px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {BORDER}"}),
                  ]),
                  html.Div([
                      html.Label("Pass-through %",style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
                      dcc.Slider(id="e-d14-pt",min=0.50,max=1.00,step=0.05,value=0.75,
                                 marks={0.50:{"label":"50%","style":{"color":TEXT_MUTED,"fontSize":"10px"}},
                                        0.75:{"label":"75%","style":{"color":TEXT_MUTED,"fontSize":"10px"}},
                                        1.00:{"label":"100%","style":{"color":TEXT_MUTED,"fontSize":"10px"}}},
                                 tooltip={"placement":"bottom","always_visible":False})
                  ],style={"width":"180px","marginTop":"2px"})]),
        html.Div("Tariff impact = rate × exports × pass-through. Default 75% per WTO pass-through research. Slider adjusts 50-100%.",
                 style={"fontSize":"11px","color":TEXT_MUTED,"marginBottom":"8px","fontStyle":"italic"}),
        html.Div(id="e-d14-cards",style={"display":"flex","gap":"12px","marginBottom":"16px"}),
        row(dark_card("Export Market Concentration",[dark_graph("e-d14-pie",320)],flex="1",sector=SEC, info_id="Export Market Concentration"),
            dark_card("Tariff Scenario Impact",[dark_graph("e-d14-tar",320)],flex="1",sector=SEC, info_id="Tariff Scenario Impact")),
        dark_card("Export Vulnerability Matrix",[dark_graph("e-d14-vm",300)],sector=SEC, info_id="Export Vulnerability Matrix"),
        raw_data_section("e-d14",[
            ("Comtrade Exports",comtrade[comtrade["flowcode"]=="X"].head(500) if not comtrade.empty else None,
             ["refyear","partnerdesc","cmddesc","primaryvalue"]),
        ]),
    ])

@app.callback(Output("e-d14-cards","children"),Output("e-d14-pie","figure"),
              Output("e-d14-tar","figure"),Output("e-d14-vm","figure"),
              Input("url","pathname"),Input("e-d14-year","value"),Input("e-d14-tariff","value"),Input("e-d14-pt","value"))
def cb_e_d14(pathname,years,tariff,pt):
    sel_years=resolve_years(years); tariff=0.25 if tariff is None else tariff
    pt = 0.75 if pt is None else float(pt)
    yr_label=f"{min(sel_years)}-{max(sel_years)}" if len(sel_years)>1 else str(sel_years[0])
    tot_exp=0.0; top_mkt="N/A"; top_mkt_sh=0.0; ct_yr=None
    if not comtrade.empty:
        valid_years=list({ct_nearest(y) for y in sel_years})
        ct_yr=comtrade[comtrade["refyear"].isin(valid_years)&(comtrade["flowcode"]=="X")]
        if not ct_yr.empty and ct_yr["primaryvalue"].sum()>0:
            tot_exp=ct_yr["primaryvalue"].sum()
            g=ct_yr.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False)
            if len(g)>0: top_mkt=g.index[0]; top_mkt_sh=round(g.iloc[0]/g.sum()*100,1)
    n_years=len(set(ct_yr["refyear"].unique())) if ct_yr is not None and not ct_yr.empty else 1
    annual_exp=tot_exp/max(n_years,1); rev_loss=(annual_exp/1e6)*tariff*pt
    dep_idx=min(top_mkt_sh/100*0.4+tariff*0.3+0.3,1.0)
    cards=html.Div([
        kri_card("Total Exports",fmt_val(annual_exp) if annual_exp>0 else "No data","",ACCENT,f"Avg/yr {yr_label}",SEC),
        kri_card("Top Export Market",top_mkt if top_mkt!="N/A" else "No data","",ACCENT2,f"{top_mkt_sh:.1f}% of exports",SEC),
        kri_card("Revenue at Risk",f"EUR {rev_loss:.0f}M" if rev_loss>0 else "No data","",
                 C_RED if tariff>0.15 else C_AMBER,f"{int(tariff*100)}% × {int(pt*100)}% p-t",SEC),
        kri_card("Dependency Index",f"{dep_idx:.3f}" if top_mkt!="N/A" else "No data","",
                 C_RED if dep_idx>0.6 else C_AMBER if dep_idx>0.4 else C_GREEN,"0=low, 1=high concentration",SEC),
    ],style={"display":"flex","gap":"12px"})
    if ct_yr is not None and not ct_yr.empty and ct_yr["primaryvalue"].sum()>0:
        g2=ct_yr.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(6)
        pie_labels=list(g2.index); pie_vals=list(g2.values/1e9)
    else:
        pie_labels=["No Comtrade data available"]; pie_vals=[1]
    PAL_PIE=[ACCENT,ACCENT2,C_AMBER,C_GREEN,C_PURPLE,C_ORANGE]
    fig_exp=go.Figure(go.Pie(labels=pie_labels,values=pie_vals,hole=0.42,textinfo="label+percent",
        marker=dict(colors=PAL_PIE[:len(pie_labels)]),textfont=dict(size=12,color=TEXT_PRI),
        hovertemplate="<b>%{label}</b><br>EUR %{value:.3f}bn -- %{percent}<extra></extra>"))
    fig_exp.update_layout(paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=40),showlegend=True,
        font=dict(color=TEXT_PRI),legend=dict(orientation="h",y=-0.15,x=0,font=dict(size=11,color=TEXT_SEC),bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text="Export<br>Markets",x=0.5,y=0.5,showarrow=False,font=dict(size=12,color=TEXT_PRI))])
    rates=[0.0,0.10,0.25,0.50]; annual_M=annual_exp/1e6; impact_vals=[annual_M*r*pt for r in rates]
    fig_tar=go.Figure(go.Bar(x=[f"{int(r*100)}% tariff" for r in rates],y=impact_vals,
        marker_color=[C_GREEN,C_AMBER,C_ORANGE,C_RED],
        text=[f"EUR {v:.0f}M" for v in impact_vals],textposition="inside",insidetextanchor="middle",textfont=dict(size=11,color="#ffffff"),name="Revenue Loss (EUR M)"))
    for lbl,col in [("Minimal",C_GREEN),("Moderate",C_AMBER),("Significant",C_ORANGE),("Severe",C_RED)]:
        fig_tar.add_trace(go.Bar(x=[None],y=[None],marker_color=col,name=lbl,showlegend=True))
    themed_layout(fig_tar,SEC,True)
    fig_tar.update_xaxes(title=dict(text=f"Applied Tariff Rate",font=dict(color=TEXT_SEC,size=11)))
    fig_tar.update_yaxes(title=dict(text="Annual Revenue Loss (EUR Million)",font=dict(color=TEXT_SEC,size=11),standoff=12),
                          range=[0,max(impact_vals)*1.3 if max(impact_vals)>0 else 500])
    fig_tar.update_layout(barmode="group",legend=dict(orientation="h",y=-0.25,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=11)))
    risks=[("UK Market Dependency",60,80,0.8),("Tariff Escalation",45,65,0.6),("Demand Shock",30,50,0.4),
           ("Currency Risk",50,30,0.3),("Regulatory Change",35,55,0.5),("Competition Risk",40,40,0.35)]
    fig_vm=go.Figure()
    for name,prob,impact,size in risks:
        col=stress_colour(prob)
        fig_vm.add_trace(go.Scatter(x=[prob],y=[impact],mode="markers+text",text=[name],
            textposition="top center",textfont=dict(size=10,color=TEXT_SEC),
            marker=dict(size=size*28+8,color=col,opacity=0.85,line=dict(color=BG_CARD,width=1.5)),
            name=name,showlegend=False))
    fig_vm.add_vline(x=50,line_dash="dot",line_color=BORDER_LT,opacity=0.6)
    fig_vm.add_hline(y=50,line_dash="dot",line_color=BORDER_LT,opacity=0.6)
    for qx,qy,qlbl,qcol in [(25,85,"Low Prob / High Impact",TEXT_MUTED),(75,85,"? High Prob / High Impact",C_RED),
                               (25,15,"Low Prob / Low Impact",TEXT_MUTED),(75,15,"High Prob / Low Impact",C_AMBER)]:
        fig_vm.add_annotation(x=qx,y=qy,text=qlbl,showarrow=False,font=dict(size=10,color=qcol),bgcolor=hex_rgba("#161b22",0.8))
    for lbl,col in [("Critical (>70%)",C_RED),("High (40-70%)",C_ORANGE),("Moderate (20-40%)",C_AMBER),("Low (<20%)",C_GREEN)]:
        fig_vm.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(size=10,color=col),name=lbl,showlegend=True))
    themed_layout(fig_vm,SEC,True)
    fig_vm.update_xaxes(title=dict(text="Probability of Risk Occurring (%)",font=dict(color=TEXT_SEC,size=11),standoff=12),range=[0,100])
    fig_vm.update_yaxes(title=dict(text="Potential Business Impact (%)",font=dict(color=TEXT_SEC,size=11),standoff=12),range=[0,100])
    fig_vm.update_layout(legend=dict(orientation="h",y=-0.25,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=11)))
    return cards,fig_exp,fig_tar,fig_vm

make_raw_toggle("e-d14-raw-toggle-btn","e-d14-raw-panel","e-d14-raw-content","e-d14-raw-source-tabs",[
    ("Comtrade Exports",comtrade[comtrade["flowcode"]=="X"].head(500) if not comtrade.empty else None,
     ["refyear","partnerdesc","cmddesc","primaryvalue"]),
])


# ???????????????????????????????????????????????????????
# AGRICULTURE DASHBOARDS (8 dashboards)
# ???????????????????????????????????????????????????????
ASEC = "agri"

def a_header(title, subtitle, controls):
    th = THEMES[ASEC]
    return html.Div([
        html.Div([
            html.H2(title, style={"color":th["text"],"fontWeight":"600","margin":"0 0 2px","fontSize":"20px"}),
            html.P(subtitle, style={"color":th["muted"],"margin":0,"fontSize":"12px"}),
        ], style={"flex":"1"}),
        *controls,
    ], style={"display":"flex","gap":"24px","alignItems":"flex-end","marginBottom":"16px",
              "background":th["card"],"border":f"1px solid {th['border']}","borderRadius":"8px","padding":"16px"})

def a_card(title, children, badge=None, flex=None, info_id=None):
    return dark_card(title, children, badge=badge, flex=flex, sector=ASEC, info_id=info_id)

def a_kri(title, value, unit="", colour=None, subtitle=""):
    return kri_card(title, value, unit, colour, subtitle, ASEC)

def a_graph(gid, height=280):
    return dark_graph(gid, height)

# ?? A-D1: Agriculture Strategic Overview ??????????????????????????????????????
def a_d1_layout():
    th = THEMES[ASEC]
    dd_years = AGRI_DD_YEARS or [2024]
    return html.Div([
        a_header("Agriculture & Food -- Strategic Overview",
                 "Food self-sufficiency . export surplus . commodity breakdown . seasonal risk signals",
                 [yr_dropdown_generic("a-d1-year", dd_years, multi=True, default=2023)]),
        html.Div(id="a-d1-kri", style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(a_card("Export value by commodity",[a_graph("a-d1-commodity",260)],flex="2", info_id="Export value by commodity"),
            a_card("Key risk signals",[html.Div(id="a-d1-insights",style={"overflowY":"auto","maxHeight":"280px"})],flex="1", info_id="Key risk signals")),
        row(a_card("Export market share",[a_graph("a-d1-market",220)],flex="1", info_id="Export market share"),
            a_card("Food self-sufficiency trend",[a_graph("a-d1-trend",220)],flex="1", info_id="Food self-sufficiency trend")),
        raw_data_section("a-d1",[
            ("Comtrade Exports", agri_ct[agri_ct["flowcode"]=="X"].head(500) if not agri_ct.empty else None,
             ["refyear","partnerdesc","cmddesc","primaryvalue"]),
            ("CSO Output", agri_out.head(500) if not agri_out.empty else None,
             ["year","metric","value","unit"]),
            ("FAOSTAT", faostat.head(500) if not faostat.empty else None,
             ["year","element","item","value","unit"]),
        ]),
    ])

@app.callback(
    Output("a-d1-kri","children"),Output("a-d1-commodity","figure"),
    Output("a-d1-insights","children"),Output("a-d1-market","figure"),Output("a-d1-trend","figure"),
    Input("url","pathname"),Input("a-d1-year","value"))
def cb_a_d1(pathname, year_val):
    th = THEMES[ASEC]
    year = resolve_year(year_val, AGRI_CT_YMAX)
    ct_yr = ct_nearest(year, AGRI_CT_YEARS)

    # KPIs from Comtrade
    exp_val = imp_val = 0.0; top_mkt = "N/A"; top_mkt_sh = 0.0; n_partners = 0
    if not agri_ct.empty:
        ct_y = agri_ct[agri_ct["refyear"]==ct_yr]
        exp = ct_y[ct_y["flowcode"]=="X"]; imp = ct_y[ct_y["flowcode"]=="M"]
        exp_val = exp["primaryvalue"].sum(); imp_val = imp["primaryvalue"].sum()
        if not exp.empty:
            gm = exp.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False)
            if len(gm)>0: top_mkt=gm.index[0]; top_mkt_sh=round(gm.iloc[0]/gm.sum()*100,1); n_partners=len(gm)

    # Self-sufficiency from FAOSTAT — Fix #9: item-matched inner join.
    # Previous method summed production & domestic supply across ALL rows,
    # including items Ireland can't produce (cocoa, coffee, dates). The
    # denominator was inflated, under-stating self-sufficiency by ~2pp.
    # Correct: join on 'item', compute SSR only for food items that have both.
    self_suff = 0.0
    fao_year_used = None
    fao_fallback = False
    if not faostat.empty:
        fao_years = sorted(faostat["year"].dropna().unique().astype(int).tolist())
        fao_year = year if year in fao_years else max((y for y in fao_years if y <= year), default=max(fao_years))
        fao_year_used = fao_year
        fao_fallback = fao_year != year
        fy = faostat[faostat["year"]==fao_year]
        prod_rows = fy[fy["element"].str.contains("production", case=False, na=False)][["item","value"]]
        dom_rows  = fy[fy["element"].str.contains("domestic supply", case=False, na=False)][["item","value"]]
        # Item-matched inner join
        merged = prod_rows.merge(dom_rows, on="item", how="inner", suffixes=("_prod","_dom"))
        if not merged.empty:
            prod_sum = merged["value_prod"].sum()
            dom_sum  = merged["value_dom"].sum()
            if dom_sum > 0 and prod_sum > 0:
                self_suff = round(prod_sum/dom_sum*100, 1)

    # Import KPIs
    imp_val2 = 0.0; top_imp = "N/A"; top_imp_sh = 0.0; n_imp_partners = 0
    if not agri_ct.empty:
        ct_imp_data = agri_ct[(agri_ct["refyear"]==ct_yr)&(agri_ct["flowcode"]=="M")]
        imp_val2 = ct_imp_data["primaryvalue"].sum()
        if not ct_imp_data.empty:
            gi = ct_imp_data.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False)
            if len(gi)>0: top_imp=gi.index[0]; top_imp_sh=round(gi.iloc[0]/gi.sum()*100,1); n_imp_partners=len(gi)
    fao_note = f"Production/supply qty, FAOSTAT {fao_year_used}" + (" (latest available)" if fao_fallback else "")
    cards = html.Div([
        a_kri("Food self-sufficiency",f"{self_suff:.0f}","%",
              th["accent"] if self_suff>100 else th["c2"] if self_suff>70 else th["red"],
              fao_note),
        a_kri("Agri exports",fmt_val(exp_val),"",th["accent"],f"Comtrade {ct_yr}"),
        a_kri("Agri imports",fmt_val(imp_val2),"",th["c3"],f"Comtrade {ct_yr}"),
        a_kri("Trade balance",fmt_val(exp_val-imp_val2),"",
              th["accent"] if exp_val>imp_val2 else th["red"],"Net agri trade balance"),
        a_kri("Export markets",str(n_partners),"",th["accent2"],f"Export partners {ct_yr}"),
        a_kri("Top export market",top_mkt,"",th["c2"],f"{top_mkt_sh:.1f}% of exports"),
        a_kri("Top import source",top_imp,"",th["c3"],f"{top_imp_sh:.1f}% of imports"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap"})

    # Commodity bar chart
    fig_comm = go.Figure()
    if not agri_ct.empty:
        ct_y = agri_ct[agri_ct["refyear"]==ct_yr]
        exp = ct_y[ct_y["flowcode"]=="X"]
        if not exp.empty:
            gc = exp.groupby("cmddesc")["primaryvalue"].sum().sort_values(ascending=False).head(8)
            cols_bar = [th["c1"],th["c2"],th["c3"],th["c4"],th["c5"],th["c1"],th["c2"],th["c3"]]
            fig_comm.add_trace(go.Bar(y=list(gc.index),x=list(gc.values/1e9),orientation="h",
                marker_color=cols_bar[:len(gc)],
                text=[f"EUR {v/1e9:.2f}bn" for v in gc.values],textposition="outside",
                textfont=dict(color=th["text"],size=11),name="Export value"))
    themed_layout(fig_comm,ASEC,False)
    fig_comm.update_xaxes(title=dict(text="Export Value (EUR Billion)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_comm.update_yaxes(title=dict(text="Commodity Group",font=dict(color=th["muted"],size=11),standoff=12))
    fig_comm

    # Insights
    insights = []
    if self_suff > 130: insights.append(("STRONG",th["accent"],"Food Surplus Nation",f"Ireland produces {self_suff:.0f}% of its food needs -- strong net exporter"))
    elif self_suff < 80: insights.append(("RISK",th["red"],"Food Import Dependency",f"Self-sufficiency at {self_suff:.0f}% -- below food security threshold"))
    if top_mkt_sh > 35: insights.append(("WATCH",th["amber"],"Market Concentration",f"{top_mkt} accounts for {top_mkt_sh:.1f}% of exports -- Brexit exposure"))
    if exp_val > 0 and imp_val > 0:
        surplus_pct = (exp_val-imp_val)/imp_val*100
        if surplus_pct > 50: insights.append(("STRONG",th["c2"],"Trade Surplus",f"Agri exports exceed imports by {surplus_pct:.0f}% -- robust net position"))
    insights.append(("INFO",th["c3"],"Seasonal Watch","Grass growth & dairy output track closely -- monitor April-September"))
    if not insights: insights.append(("OK",th["accent"],"No Alerts","All agriculture indicators within normal ranges"))
    ins_els = [insight_item(s,c,t,m) for s,c,t,m in insights]

    # Market pie
    fig_mkt = go.Figure()
    if not agri_ct.empty:
        ct_y = agri_ct[agri_ct["refyear"]==ct_yr]
        exp = ct_y[ct_y["flowcode"]=="X"]
        if not exp.empty:
            gm = exp.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(6)
            fig_mkt.add_trace(go.Pie(labels=list(gm.index),values=list(gm.values/1e9),hole=0.45,
                marker=dict(colors=[th["c1"],th["c2"],th["c3"],th["c4"],th["c5"],th["muted"]]),
                textfont=dict(size=11,color=th["text"])))
    themed_layout(fig_mkt,ASEC,True)
    fig_mkt.update_layout(paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=10),
        legend=dict(orientation="h",y=-0.2,x=0,font=dict(size=11,color=th["muted"]),bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text="Export<br>Markets",x=0.5,y=0.5,showarrow=False,font=dict(size=12,color=th["text"]))])

    # Self-sufficiency trend
    fig_trend = go.Figure()
    if not faostat.empty:
        trend_data = []
        for y in sorted(faostat["year"].unique()):
            fy = faostat[faostat["year"]==y]
            prod = fy[fy["element"]=="Production"]["value"].sum()
            dom  = fy[fy["element"]=="Domestic supply quantity"]["value"].sum()
            if dom > 0: trend_data.append({"year":y,"ss":round(prod/dom*100,1)})
        if trend_data:
            df_t = pd.DataFrame(trend_data)
            fig_trend.add_trace(go.Scatter(x=df_t["year"],y=df_t["ss"],mode="lines+markers",
                name="Self-sufficiency",line=dict(color=th["accent"],width=2.5),
                marker=dict(color=th["accent"],size=5),fill="tozeroy",fillcolor=hex_rgba(th["accent"],0.08)))
            fig_trend.add_hline(y=100,line_dash="dash",line_color=th["c2"],opacity=0.7,
                annotation_text="100% threshold",annotation_font=dict(color=th["c2"],size=10))
    themed_layout(fig_trend,ASEC,False)
    fig_trend.update_xaxes(title=dict(text="Year",font=dict(color=th["muted"],size=11),standoff=12))
    fig_trend.update_yaxes(title=dict(text="Self-Sufficiency (%)",font=dict(color=th["muted"],size=11),standoff=12),ticksuffix="%")
    fig_trend
    return cards,fig_comm,ins_els,fig_mkt,fig_trend

make_raw_toggle("a-d1-raw-toggle-btn","a-d1-raw-panel","a-d1-raw-content","a-d1-raw-source-tabs",[
    ("Comtrade Exports",agri_ct[agri_ct["flowcode"]=="X"].head(500) if not agri_ct.empty else None,["refyear","partnerdesc","cmddesc","primaryvalue"]),
    ("CSO Output",agri_out.head(500) if not agri_out.empty else None,["year","metric","value","unit"]),
    ("FAOSTAT",faostat.head(500) if not faostat.empty else None,["year","element","item","value","unit"]),
])


# ?? A-D2 through A-D8: remaining agriculture dashboards ???????????????????????
def a_d2_layout():
    return html.Div([
        a_header("Agriculture -- Import Dependency","Food import concentration, HHI by commodity, top import sources",
                 [yr_dropdown_generic("a-d2-year",AGRI_DD_YEARS,multi=True,default=2023)]),
        html.Div(id="a-d2-kri",style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(a_card("Import HHI by commodity group",[a_graph("a-d2-hhi",260)],flex="1", info_id="Import HHI by commodity group"),
            a_card("Top import sources",[a_graph("a-d2-sources",260)],flex="1", info_id="Top import sources")),
        row(a_card("Continental import breakdown",[a_graph("a-d2-cont",220)],flex="1", info_id="Continental import breakdown"),
            a_card("Import trend 2010-2024",[a_graph("a-d2-trend",220)],flex="1", info_id="Import trend 2010-2024")),
        a_card("Import by commodity × source matrix (top 10 products)",[a_graph("a-d2-matrix",320)], info_id="Import by commodity × source matrix (top 10 products)"),
        raw_data_section("a-d2",[
            ("Comtrade Imports",agri_ct[agri_ct["flowcode"]=="M"].head(500) if not agri_ct.empty else None,["refyear","partnerdesc","cmddesc","primaryvalue","continent"]),
            ("CSO Trade",agri_trade.head(500) if not agri_trade.empty else None,["year","month","flow","commodity","value","unit"]),
        ]),
    ])

@app.callback(Output("a-d2-kri","children"),Output("a-d2-hhi","figure"),
              Output("a-d2-sources","figure"),Output("a-d2-cont","figure"),Output("a-d2-trend","figure"),
              Output("a-d2-matrix","figure"),
              Input("url","pathname"),Input("a-d2-year","value"))
def cb_a_d2(pathname,year_val):
    th=THEMES[ASEC]; year=resolve_year(year_val,AGRI_CT_YMAX); ct_yr=ct_nearest(year,AGRI_CT_YEARS)
    imp=agri_ct[(agri_ct["refyear"]==ct_yr)&(agri_ct["flowcode"]=="M")] if not agri_ct.empty else pd.DataFrame()
    imp_total=imp["primaryvalue"].sum() if not imp.empty else 0
    top_src="N/A"; top_src_sh=0.0
    if not imp.empty:
        gs=imp.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False)
        if len(gs)>0: top_src=gs.index[0]; top_src_sh=round(gs.iloc[0]/gs.sum()*100,1)
    hhi_score=0.0
    if not imp.empty and imp_total>0:
        gc=imp.groupby("cmddesc")["primaryvalue"].sum()
        hhi_score=round(sum((v/imp_total)**2 for v in gc.values),4)
    cards=html.Div([
        a_kri("Total agri imports",fmt_val(imp_total),"",th["c3"],f"Comtrade {ct_yr}"),
        a_kri("Top import source",top_src,"",th["accent"],f"{top_src_sh:.1f}% share"),
        a_kri("Import HHI",f"{hhi_score:.3f}","",hhi_colour(hhi_score),hhi_label(hhi_score)),
        a_kri("Import partners",str(imp["partnerdesc"].nunique()) if not imp.empty else "0","",th["accent2"],"Unique countries"),
    ],style={"display":"flex","gap":"12px","flexWrap":"wrap"})
    fig_hhi=go.Figure()
    if not imp.empty:
        gc=imp.groupby("cmddesc")["primaryvalue"].sum().sort_values(ascending=False).head(10)
        total=gc.sum() or 1
        hhis=[round((v/total)**2,4) for v in gc.values]
        fig_hhi.add_trace(go.Bar(y=list(gc.index),x=hhis,orientation="h",
            text=[f"{h:.4f}" for h in hhis],textposition="inside",insidetextanchor="middle",textfont=dict(size=10,color="#ffffff"),
            marker_color=[th["c1"] if h>0.15 else th["c2"] for h in hhis]))
    themed_layout(fig_hhi,ASEC,False); fig_hhi.update_xaxes(title=dict(text="HHI Score",font=dict(color=th["muted"],size=11),standoff=12),range=[0,0.55])
    fig_hhi.update_yaxes(title=dict(text="Commodity Group",font=dict(color=th["muted"],size=11),standoff=12)); fig_hhi
    fig_src=go.Figure()
    if not imp.empty:
        gs=imp.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(10)
        fig_src.add_trace(go.Bar(y=list(gs.index),x=list(gs.values/1e9),orientation="h",
            text=[f"EUR {v:.2f}bn" for v in gs.values/1e9],textposition="inside",insidetextanchor="middle",textfont=dict(size=10,color="#ffffff"),
            marker_color=[th["c1"],th["c2"],th["c3"],th["c4"],th["c5"]]*2))
    themed_layout(fig_src,ASEC,False); fig_src.update_xaxes(title=dict(text="Import Value (EUR Billion)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_src.update_yaxes(title=dict(text="Country",font=dict(color=th["muted"],size=11),standoff=12)); fig_src
    fig_cont=go.Figure()
    if not imp.empty and "continent" in imp.columns:
        gc=imp.groupby("continent")["primaryvalue"].sum().sort_values(ascending=False)
        fig_cont.add_trace(go.Bar(x=list(gc.index),y=list(gc.values/1e9),marker_color=th["c1"],
            text=[f"EUR {v/1e9:.2f}bn" for v in gc.values],textposition="inside",insidetextanchor="middle",textfont=dict(size=10,color="#ffffff")))
    themed_layout(fig_cont,ASEC,False); fig_cont.update_xaxes(title=dict(text="Region",font=dict(color=th["muted"],size=11),standoff=12))
    fig_cont.update_yaxes(title=dict(text="Import Value (EUR Billion)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_trend=go.Figure()
    if not agri_ct.empty:
        trend=[{"year":y,"imp":agri_ct[(agri_ct["refyear"]==y)&(agri_ct["flowcode"]=="M")]["primaryvalue"].sum()/1e9}
               for y in sorted(agri_ct["refyear"].unique())]
        df_t=pd.DataFrame(trend)
        fig_trend.add_trace(go.Scatter(x=df_t["year"],y=df_t["imp"],mode="lines+markers",name="Imports (EUR bn)",
            line=dict(color=th["c3"],width=2.5),marker=dict(size=5,color=th["c3"])))
    themed_layout(fig_trend,ASEC,False); fig_trend.update_xaxes(title=dict(text="Year",font=dict(color=th["muted"],size=11),standoff=12))
    fig_trend.update_yaxes(title=dict(text="Import Value (EUR Billion)",font=dict(color=th["muted"],size=11),standoff=12))
    # Import commodity × source matrix (top 10 products by value, top 6 sources)
    fig_matrix = go.Figure()
    if not imp.empty:
        JUNK = {"world","areas, nes","other","unspecified","not specified","total"}
        imp_clean = imp[~imp["partnerdesc"].astype(str).str.strip().str.lower().isin(JUNK)]
        top_comms = imp_clean.groupby("cmddesc")["primaryvalue"].sum().sort_values(ascending=False).head(10).index
        pivot_imp = imp_clean.groupby(["cmddesc","partnerdesc"])["primaryvalue"].sum().unstack(fill_value=0)
        pivot_imp = pivot_imp.loc[[c for c in top_comms if c in pivot_imp.index]]
        top_sources = imp_clean.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(6).index
        pivot_imp = pivot_imp[[c for c in top_sources if c in pivot_imp.columns]]
        if not pivot_imp.empty:
            fig_matrix = go.Figure(go.Heatmap(
                z=pivot_imp.values/1e9,
                x=list(pivot_imp.columns),
                y=list(pivot_imp.index),
                colorscale=[[0,th["bg"]],[0.5,th["c3"]],[1,th["c1"]]],
                text=[[f"EUR {v:.2f}bn" for v in row_v] for row_v in pivot_imp.values/1e9],
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b><br>Source: %{x}<br>Value: %{text}<extra></extra>",
                showscale=True,
            ))
            themed_layout(fig_matrix,ASEC,False)
            fig_matrix.update_xaxes(title=dict(text="Import Source",font=dict(color=th["muted"],size=11),standoff=12),tickangle=-30)
            fig_matrix.update_yaxes(title=dict(text="Commodity Group",font=dict(color=th["muted"],size=11),standoff=12))
            fig_matrix.update_layout(margin=dict(l=180,r=20,t=20,b=80))
    return cards,fig_hhi,fig_src,fig_cont,fig_trend,fig_matrix

make_raw_toggle("a-d2-raw-toggle-btn","a-d2-raw-panel","a-d2-raw-content","a-d2-raw-source-tabs",[
    ("Comtrade Imports",agri_ct[agri_ct["flowcode"]=="M"].head(500) if not agri_ct.empty else None,["refyear","partnerdesc","cmddesc","primaryvalue","continent"]),
    ("CSO Trade",agri_trade.head(500) if not agri_trade.empty else None,["year","month","flow","commodity","value","unit"]),
])

def a_d3_layout():
    # Build market options: all countries Ireland exports to + "All markets"
    mkt_opts = [{"label":"All markets (universal shock)","value":"__ALL__"}]
    if not agri_ct.empty:
        top_mkts = (agri_ct[agri_ct["flowcode"]=="X"].groupby("partnerdesc")["primaryvalue"].sum()
                    .sort_values(ascending=False).head(25).index.tolist())
        mkt_opts += [{"label":m,"value":m} for m in top_mkts]
    return html.Div([
        a_header("Agriculture -- Export Dependency","Export market concentration, tariff scenarios, top commodities by destination",
                 [yr_dropdown_generic("a-d3-year",AGRI_DD_YEARS,multi=True,default=2023),
                  html.Div([html.Label("Tariff Market",style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
                             dcc.Dropdown(id="a-d3-market",options=mkt_opts,value="United Kingdom" if any(o["value"]=="United Kingdom" for o in mkt_opts) else "__ALL__",
                                          clearable=False,style={"fontSize":"13px","width":"230px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {BORDER}"})]),
                  html.Div([html.Label("Tariff Rate",style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
                             dcc.Dropdown(id="a-d3-tariff",options=[{"label":"0%","value":0.0},{"label":"10%","value":0.10},{"label":"25%","value":0.25},{"label":"50%","value":0.50}],
                                          value=0.10,clearable=False,style={"fontSize":"13px","width":"140px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {BORDER}"})]),
                  html.Div([html.Label("Pass-through %",style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
                             dcc.Slider(id="a-d3-pt",min=0.50,max=1.00,step=0.05,value=0.75,
                                        marks={0.50:{"label":"50%","style":{"color":TEXT_MUTED,"fontSize":"10px"}},
                                               0.75:{"label":"75%","style":{"color":TEXT_MUTED,"fontSize":"10px"}},
                                               1.00:{"label":"100%","style":{"color":TEXT_MUTED,"fontSize":"10px"}}},
                                        tooltip={"placement":"bottom","always_visible":False})],
                            style={"width":"170px","marginTop":"2px"})]),
        html.Div("Assumes exporter absorbs the tariff at the selected pass-through rate. WTO research supports 50-100% depending on product elasticity.",
                 style={"fontSize":"11px","color":TEXT_MUTED,"marginBottom":"8px","fontStyle":"italic"}),
        html.Div(id="a-d3-kri",style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(a_card("Top export markets",[a_graph("a-d3-markets",280)],flex="1", info_id="Top export markets"),
            a_card("Tariff impact (selected market × rate × pass-through)",[a_graph("a-d3-tariff-fig",280)],flex="1", info_id="Tariff impact by market")),
        a_card("Export by commodity ? market matrix",[a_graph("a-d3-matrix",280)], info_id="Export by commodity ? market matrix"),
        raw_data_section("a-d3",[
            ("Comtrade Exports",agri_ct[agri_ct["flowcode"]=="X"].head(500) if not agri_ct.empty else None,["refyear","partnerdesc","cmddesc","primaryvalue"]),
        ]),
    ])

@app.callback(Output("a-d3-kri","children"),Output("a-d3-markets","figure"),
              Output("a-d3-tariff-fig","figure"),Output("a-d3-matrix","figure"),
              Input("url","pathname"),Input("a-d3-year","value"),Input("a-d3-tariff","value"),
              Input("a-d3-market","value"),Input("a-d3-pt","value"))
def cb_a_d3(pathname,year_val,tariff,market,pt):
    th=THEMES[ASEC]; year=resolve_year(year_val,AGRI_CT_YMAX); ct_yr=ct_nearest(year,AGRI_CT_YEARS)
    tariff=0.10 if tariff is None else tariff
    pt = 0.75 if pt is None else float(pt)
    market = market or "__ALL__"
    exp=agri_ct[(agri_ct["refyear"]==ct_yr)&(agri_ct["flowcode"]=="X")] if not agri_ct.empty else pd.DataFrame()
    exp_total=exp["primaryvalue"].sum() if not exp.empty else 0
    top_mkt="N/A"; top_sh=0.0
    if not exp.empty:
        gm=exp.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False)
        if len(gm)>0: top_mkt=gm.index[0]; top_sh=round(gm.iloc[0]/gm.sum()*100,1)
    # Fix #4-B + #4b: apply tariff only to the selected market's exports,
    # and apply a pass-through multiplier (default 0.75 per WTO research).
    if market == "__ALL__":
        base_for_tariff = exp_total
        mkt_label = "all markets"
    else:
        base_for_tariff = exp[exp["partnerdesc"]==market]["primaryvalue"].sum() if not exp.empty else 0
        mkt_label = market
    rev_loss = (base_for_tariff/1e6) * tariff * pt
    cards=html.Div([
        a_kri("Total agri exports",fmt_val(exp_total),"",th["accent"],f"Comtrade {ct_yr}"),
        a_kri("Top export market",top_mkt,"",th["c2"],f"{top_sh:.1f}% of exports"),
        a_kri("Revenue at risk",f"EUR {rev_loss:.0f}M","",th["red"] if tariff>0.2 else th["amber"],f"{int(tariff*100)}% × {int(pt*100)}% p-t on {mkt_label}"),
        a_kri("Export partners",str(exp["partnerdesc"].nunique()) if not exp.empty else "0","",th["accent2"],"Unique markets"),
    ],style={"display":"flex","gap":"12px","flexWrap":"wrap"})
    fig_mkt=go.Figure()
    if not exp.empty:
        gm=exp.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(10)
        fig_mkt.add_trace(go.Bar(y=list(gm.index),x=list(gm.values/1e9),orientation="h",
            text=[f"EUR {v:.2f}bn" for v in gm.values/1e9],textposition="inside",insidetextanchor="middle",textfont=dict(size=10,color="#ffffff"),
            marker_color=[th["c1"],th["c2"],th["c3"],th["c4"],th["c5"]]*2))
    themed_layout(fig_mkt,ASEC,False); fig_mkt.update_xaxes(title=dict(text="Export Value (EUR Billion)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_mkt.update_yaxes(title=dict(text="Market",font=dict(color=th["muted"],size=11),standoff=12)); fig_mkt
    # Scenario bars show the selected market's losses at 0%/5%/10%/25%/50%,
    # scaled by the chosen pass-through rate.
    rates=[0.0,0.05,0.10,0.25,0.50]; base_M=base_for_tariff/1e6
    fig_tar=go.Figure(go.Bar(x=[f"{int(r*100)}%" for r in rates],y=[base_M*r*pt for r in rates],
        marker_color=[th["c4"],th["c3"],th["c2"],th["amber"],th["red"]],
        text=[f"EUR {base_M*r*pt:.0f}M" for r in rates],textposition="inside",insidetextanchor="middle",textfont=dict(size=11,color="#ffffff")))
    themed_layout(fig_tar,ASEC,False); fig_tar.update_xaxes(title=dict(text=f"Tariff Rate on {mkt_label}",font=dict(color=th["muted"],size=11),standoff=12))
    fig_tar.update_yaxes(title=dict(text=f"Annual Revenue Loss at {int(pt*100)}% pass-through (EUR M)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_matrix=go.Figure()
    if not exp.empty:
        top_comms=exp.groupby("cmddesc")["primaryvalue"].sum().sort_values(ascending=False).head(8).index
        pivot=exp.groupby(["cmddesc","partnerdesc"])["primaryvalue"].sum().unstack(fill_value=0)
        pivot=pivot.loc[[c for c in top_comms if c in pivot.index]]
        top_markets=exp.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(6).index
        pivot=pivot[[c for c in top_markets if c in pivot.columns]]
        if not pivot.empty:
            fig_matrix=go.Figure(go.Heatmap(z=pivot.values/1e9,x=list(pivot.columns),y=list(pivot.index),
                colorscale=[[0,th["bg"]],[0.5,th["c3"]],[1,th["c1"]]],
                text=[[f"EUR {v:.2f}bn" for v in row_v] for row_v in pivot.values/1e9],
                texttemplate="%{text}",hovertemplate="Commodity: %{y}<br>Market: %{x}<br>Value: EUR %{z:.3f}bn<extra></extra>"))
    themed_layout(fig_matrix,ASEC,False)
    fig_matrix.update_xaxes(title=dict(text="Export Market",font=dict(color=th["muted"],size=11),standoff=12))
    fig_matrix.update_yaxes(title=dict(text="Commodity Group",font=dict(color=th["muted"],size=11),standoff=12))
    return cards,fig_mkt,fig_tar,fig_matrix

make_raw_toggle("a-d3-raw-toggle-btn","a-d3-raw-panel","a-d3-raw-content","a-d3-raw-source-tabs",[
    ("Comtrade Exports",agri_ct[agri_ct["flowcode"]=="X"].head(500) if not agri_ct.empty else None,["refyear","partnerdesc","cmddesc","primaryvalue"]),
])


def a_d4_layout():
    return html.Div([
        a_header("Agriculture -- Supply Flow","Sankey of agri commodity flows by origin and destination",[yr_dropdown_generic("a-d4-year",AGRI_DD_YEARS,multi=True,default=2023)]),
        a_card("Agriculture Supply Flow Sankey",[a_graph("a-d4-sankey",420)], info_id="Agriculture Supply Flow Sankey"),
        html.Div([html.Div(id="a-d4-imp-tbl"),html.Div(id="a-d4-exp-tbl"),html.Div(id="a-d4-summary")],style={"display":"flex","gap":"12px","marginTop":"12px"}),
        raw_data_section("a-d4",[("Comtrade All Flows",agri_ct.head(500) if not agri_ct.empty else None,["refyear","flowcode","partnerdesc","cmddesc","primaryvalue","continent"])]),
    ])

@app.callback(Output("a-d4-sankey","figure"),Output("a-d4-imp-tbl","children"),Output("a-d4-exp-tbl","children"),Output("a-d4-summary","children"),
              Input("url","pathname"),Input("a-d4-year","value"))
def cb_a_d4(pathname,year_val):
    th=THEMES[ASEC]; year=resolve_year(year_val,AGRI_CT_YMAX); ct_yr=ct_nearest(year,AGRI_CT_YEARS)
    if agri_ct.empty: return go.Figure(),html.P("No data"),html.P("No data"),html.P("No data")
    ct_y=agri_ct[agri_ct["refyear"]==ct_yr]; imp=ct_y[ct_y["flowcode"]=="M"]; exp=ct_y[ct_y["flowcode"]=="X"]
    PAL=[th["c1"],th["c2"],th["c3"],th["c4"],th["c5"],th["accent"],th["accent2"],th["muted"]]
    # Country-level Sankey -- top 8 import sources, top 6 export markets
    JUNK_S = {"world","areas, nes","other","unspecified","not specified","total"}
    imp_c = imp[~imp["partnerdesc"].str.strip().str.lower().isin(JUNK_S)]
    exp_c = exp[~exp["partnerdesc"].str.strip().str.lower().isin(JUNK_S)] if not exp.empty else exp
    ic=imp_c.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(8)
    ec=exp_c.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(6) if not exp_c.empty else pd.Series(dtype=float)
    nodes=[f">> {r}" for r in ic.index]+["Ireland"]+[f"<< {r}" for r in ec.index]
    ni={n:i for i,n in enumerate(nodes)}; src,tgt,vals,cols=[],[],[],[]
    for i,(region,val) in enumerate(ic.items()):
        src.append(ni[f">> {region}"]); tgt.append(ni["Ireland"]); vals.append(val/1e9); cols.append(hex_rgba(PAL[i%len(PAL)],0.6))
    for region,val in ec.items():
        if f"<< {region}" in ni:
            src.append(ni["Ireland"]); tgt.append(ni[f"<< {region}"]); vals.append(val/1e9); cols.append(hex_rgba(th["c1"],0.5))
    fig=go.Figure(go.Sankey(arrangement="snap",
        node=dict(pad=15,thickness=18,label=nodes,color=[hex_rgba(PAL[i%len(PAL)],0.9) if i<len(ic) else hex_rgba(th["accent"],0.9) if i==len(ic) else hex_rgba(th["c1"],0.9) for i in range(len(nodes))],line=dict(color=th["border"],width=0.5)),
        link=dict(source=src,target=tgt,value=vals,color=cols,hovertemplate="%{source.label} >> %{target.label}<br>EUR %{value:.2f}bn<extra></extra>")))
    sankey_layout(fig,ASEC)
    def make_tbl(df,title):
        if df is None or df.empty: return a_card(title,[html.P("No data",style={"color":th["muted"],"fontSize":"12px"})],flex="1", info_id=title)
        g=df.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False).head(8)
        rows=[html.Div([html.Span(p,style={"fontSize":"12px","color":th["muted"],"flex":"1"}),html.Span(fmt_val(v),style={"fontSize":"12px","color":th["text"],"fontWeight":"600"})],style={"display":"flex","justifyContent":"space-between","padding":"6px 0","borderBottom":f"1px solid {th['border']}"}) for p,v in g.items()]
        return a_card(title,rows,flex="1", info_id=title)
    return fig,make_tbl(imp,"Top Import Sources"),make_tbl(exp,"Top Export Destinations"),a_card("Flow Summary",[
        html.Div([html.Span("Total Imports",style={"color":th["muted"],"fontSize":"11px"}),html.Span(fmt_val(imp["primaryvalue"].sum()),style={"color":th["red"],"fontWeight":"700","fontSize":"16px"})],style={"marginBottom":"8px"}),
        html.Div([html.Span("Total Exports",style={"color":th["muted"],"fontSize":"11px"}),html.Span(fmt_val(exp["primaryvalue"].sum()) if not exp.empty else "N/A",style={"color":th["accent"],"fontWeight":"700","fontSize":"16px"})]),
    ],flex="1", info_id="Flow Summary")

make_raw_toggle("a-d4-raw-toggle-btn","a-d4-raw-panel","a-d4-raw-content","a-d4-raw-source-tabs",[
    ("Comtrade All Flows",agri_ct.head(500) if not agri_ct.empty else None,["refyear","flowcode","partnerdesc","cmddesc","primaryvalue","continent"]),
])

def a_d5_layout():
    th = THEMES[ASEC]
    return html.Div([
        a_header("Agriculture & Food -- Trade Map",
                 "Geographic agri trade corridors -- arc width = trade value",
                 [yr_dropdown_generic("a-d5-year",AGRI_DD_YEARS,multi=True,default=2023),
                  html.Div([
                      html.Label("Flow",style={"fontSize":"11px","color":th["muted"],"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="a-d5-flow",
                          options=[{"label":"Both","value":"Both"},{"label":"Imports only","value":"M"},{"label":"Exports only","value":"X"}],
                          value="Both",clearable=False,
                          style={"fontSize":"13px","width":"150px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}"}),
                  ])]),
        html.Div([
            html.Div([a_card("Agri Trade Map",[a_graph("a-d5-map",460)], info_id="Agri Trade Map")],style={"flex":"1","minWidth":"0"}),
            html.Div(id="a-d5-country-panel",style={"width":"230px","flexShrink":"0",
                "background":th["card"],"border":f"1px solid {th['border']}",
                "borderRadius":"8px","padding":"14px","overflowY":"auto","maxHeight":"500px"}),
        ],style={"display":"flex","gap":"14px","marginBottom":"14px"}),
        html.Div(id="a-d5-kpi",style={"display":"flex","gap":"12px"}),
        raw_data_section("a-d5",[
            ("Comtrade",agri_ct.head(500) if not agri_ct.empty else None,
             ["refyear","flowcode","partnerdesc","primaryvalue","partner_lat","partner_lon"]),
        ]),
    ])

@app.callback(
    Output("a-d5-map","figure"),Output("a-d5-kpi","children"),Output("a-d5-country-panel","children"),
    Input("url","pathname"),Input("a-d5-year","value"),Input("a-d5-flow","value"))
def cb_a_d5(pathname,year_val,flow):
    th=THEMES[ASEC]; year=resolve_year(year_val,AGRI_CT_YMAX); ct_yr=ct_nearest(year,AGRI_CT_YEARS)
    flow=flow or "Both"
    IRELAND_LAT,IRELAND_LON=53.35,-6.26
    fig=go.Figure()
    fig.update_layout(geo=dict(showframe=False,showcoastlines=True,showland=True,showocean=True,landcolor="#0d2310",
        oceancolor="#071a0a",coastlinecolor=th["border"],coastlinewidth=0.5,projection_type="natural earth",bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=0,b=0))
    if agri_ct.empty:
        return fig,[a_kri("No Data","N/A","",th["muted"])],html.Div("No data",style={"color":th["muted"],"fontSize":"12px"})
    ct_y=agri_ct[agri_ct["refyear"]==ct_yr]; imp=ct_y[ct_y["flowcode"]=="M"]; exp=ct_y[ct_y["flowcode"]=="X"]
    def add_arcs(df_flow,rgb,name):
        if df_flow.empty or "partner_lat" not in df_flow.columns: return
        g=df_flow.groupby(["partnerdesc","partner_lat","partner_lon"])["primaryvalue"].sum().reset_index()
        if g.empty: return
        max_v=g["primaryvalue"].max() or 1
        for _,r in g.iterrows():
            if r["partner_lat"]==0 and r["partner_lon"]==0: continue
            raw_w=r["primaryvalue"]/max_v; w=max(1.5,min(10,raw_w*10)); op=max(0.35,min(1.0,raw_w))
            rv,gv,bv=int(rgb[0]),int(rgb[1]),int(rgb[2])
            fig.add_trace(go.Scattergeo(lat=[r["partner_lat"],IRELAND_LAT],lon=[r["partner_lon"],IRELAND_LON],
                mode="lines",showlegend=False,line=dict(width=w,color=f"rgba({rv},{gv},{bv},{op:.2f})"),
                hovertemplate=f"<b>{r['partnerdesc']}</b><br>{name}: {fmt_val(r['primaryvalue'])}<extra></extra>"))
            fig.add_trace(go.Scattergeo(lat=[r["partner_lat"]],lon=[r["partner_lon"]],mode="markers",showlegend=False,
                marker=dict(size=max(5,w*1.2),color=f"rgba({rv},{gv},{bv},0.9)"),
                text=f"{r['partnerdesc']}: {fmt_val(r['primaryvalue'])}",hoverinfo="text"))
    if flow in ("Both","M"): add_arcs(imp,(247,131,35),"Imports")
    if flow in ("Both","X"): add_arcs(exp,(76,175,80),"Exports")
    fig.add_trace(go.Scattergeo(lat=[IRELAND_LAT],lon=[IRELAND_LON],mode="markers+text",
        marker=dict(size=14,color=th["accent"],line=dict(color=WHITE,width=2)),
        text=["Ireland"],textposition="top center",textfont=dict(color=th["text"],size=12),showlegend=False))
    if flow in ("Both","M"):
        fig.add_trace(go.Scattergeo(lat=[None],lon=[None],mode="lines",name="Imports",line=dict(color="rgba(247,131,35,1)",width=3),showlegend=True))
    if flow in ("Both","X"):
        fig.add_trace(go.Scattergeo(lat=[None],lon=[None],mode="lines",name="Exports",line=dict(color="rgba(76,175,80,1)",width=3),showlegend=True))
    fig.update_layout(legend=dict(bgcolor=f"rgba(13,35,16,0.9)",bordercolor=th["border"],borderwidth=1,
        font=dict(color=th["text"],size=12),x=0.01,y=0.99),font=dict(color=th["text"]))
    kpi=html.Div([
        a_kri("Agri imports",fmt_val(imp["primaryvalue"].sum()) if not imp.empty else "N/A","",th["c3"],f"Year {ct_yr}"),
        a_kri("Agri exports",fmt_val(exp["primaryvalue"].sum()) if not exp.empty else "N/A","",th["accent"],f"Year {ct_yr}"),
        a_kri("Import partners",str(imp["partnerdesc"].nunique()) if not imp.empty else "0","",th["c2"],"Unique countries"),
        a_kri("Export partners",str(exp["partnerdesc"].nunique()) if not exp.empty else "0","",th["c1"],"Unique markets"),
    ],style={"display":"flex","gap":"12px"})
    def country_rows(df,colour,label):
        if df.empty: return []
        g=df.groupby("partnerdesc")["primaryvalue"].sum().sort_values(ascending=False); total=g.sum() or 1
        rows=[html.Div([html.Span(label,style={"fontSize":"9px","color":colour,"letterSpacing":"0.5px","fontWeight":"600"}),
                        html.Span(f"({fmt_val(g.sum())} total)",style={"fontSize":"9px","color":th["muted"],"marginLeft":"6px"})],
                       style={"padding":"6px 0 4px","borderBottom":f"1px solid {th['border']}","marginBottom":"4px"})]
        for partner,val in g.head(10).items():
            bar_w=min(100,val/g.iloc[0]*100) if g.iloc[0]>0 else 0
            rows.append(html.Div([
                html.Div([html.Span("*",style={"color":colour,"fontSize":"8px","marginRight":"4px"}),
                          html.Span(partner,style={"fontSize":"11px","color":th["text"],"flex":"1","overflow":"hidden","textOverflow":"ellipsis","whiteSpace":"nowrap"}),
                          html.Span(fmt_val(val),style={"fontSize":"11px","color":colour,"fontWeight":"600","flexShrink":"0","marginLeft":"6px"})],
                         style={"display":"flex","alignItems":"center","marginBottom":"3px"}),
                html.Div(style={"background":BG_INPUT,"borderRadius":"2px","height":"4px","overflow":"hidden","marginBottom":"6px"},
                         children=[html.Div(style={"width":f"{bar_w:.0f}%","height":"100%","background":colour,"borderRadius":"2px"})]),
            ]))
        return rows
    imp_rows=country_rows(imp,th["c3"],"IMPORTS") if flow in ("Both","M") else []
    exp_rows=country_rows(exp,th["accent"],"EXPORTS") if flow in ("Both","X") else []
    panel=html.Div([
        html.Div([html.Span(f"Year {ct_yr}",style={"fontSize":"11px","color":th["text"],"fontWeight":"600"}),
                  html.Span(" -- Trade Partners",style={"fontSize":"10px","color":th["muted"]})],
                 style={"marginBottom":"10px","paddingBottom":"8px","borderBottom":f"1px solid {th['border']}"}),
        *imp_rows,*exp_rows,
    ])
    return fig,kpi,panel


make_raw_toggle("a-d5-raw-toggle-btn","a-d5-raw-panel","a-d5-raw-content","a-d5-raw-source-tabs",[
    ("Comtrade",agri_ct.head(500) if not agri_ct.empty else None,["refyear","flowcode","partnerdesc","primaryvalue","partner_lat","partner_lon"]),
])

def a_d6_layout():
    return html.Div([
        a_header("Agriculture -- Stress Test","Supply disruption scenario -- impact by commodity and alternative sourcing",
                 [yr_dropdown_generic("a-d6-year",AGRI_DD_YEARS,multi=True,default=2023),
                  html.Div([html.Label("Remove Country",style={"fontSize":"11px","color":TEXT_MUTED,"display":"block","marginBottom":"4px"}),
                             dcc.Dropdown(id="a-d6-country",options=[{"label":p,"value":p} for p in (sorted(agri_ct["partnerdesc"].dropna().unique()) if not agri_ct.empty else [])],
                                          value="United Kingdom" if not agri_ct.empty else None,clearable=False,
                                          style={"fontSize":"13px","width":"200px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {BORDER}"})])]),
        html.Div("Scenario assumes no short-term substitution from alternative suppliers. Real-world impact is typically lower as reshoring and rerouting mitigate some losses over 3-12 months.",
                 style={"fontSize":"11px","color":TEXT_MUTED,"marginBottom":"8px","fontStyle":"italic"}),
        html.Div(id="a-d6-cards",style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(a_card("Supply lost by commodity (%)",[a_graph("a-d6-bar",280)],flex="2", info_id="Supply lost by commodity (%)"),
            a_card("Top alternative suppliers",[html.Div(id="a-d6-alts")],flex="1", info_id="Top alternative suppliers")),
        raw_data_section("a-d6",[("Comtrade Imports",agri_ct[agri_ct["flowcode"]=="M"].head(500) if not agri_ct.empty else None,["refyear","partnerdesc","cmddesc","primaryvalue"])]),
    ])

@app.callback(Output("a-d6-cards","children"),Output("a-d6-bar","figure"),Output("a-d6-alts","children"),
              Input("url","pathname"),Input("a-d6-year","value"),Input("a-d6-country","value"))
def cb_a_d6(pathname,year_val,country):
    th=THEMES[ASEC]; year=resolve_year(year_val,AGRI_CT_YMAX); ct_yr=ct_nearest(year,AGRI_CT_YEARS)
    if agri_ct.empty or not country: return [],go.Figure(),html.P("No data")
    imp=agri_ct[(agri_ct["refyear"]==ct_yr)&(agri_ct["flowcode"]=="M")]
    if imp.empty: return [],go.Figure(),html.P("No data")
    results=[]
    for prod in imp["cmddesc"].unique():
        pdf=imp[imp["cmddesc"]==prod]; tv=pdf["primaryvalue"].sum()
        if tv==0: continue
        cv=pdf[pdf["partnerdesc"]==country]["primaryvalue"].sum(); ls=cv/tv*100
        results.append({"product":prod[:35],"lost_share":round(ls,1),"impact":stress_label(ls),"colour":stress_colour(ls)})
    results=sorted(results,key=lambda x:x["lost_share"],reverse=True)
    crit=sum(1 for r in results if r["impact"]=="CRITICAL")
    cards=html.Div([a_kri("Products affected",str(sum(1 for r in results if r.get("lost_share",0)>0)),"",th["green"] if sum(1 for r in results if r.get("lost_share",0)>0)==0 else th["accent2"],"Products with >0% supply from this country"),a_kri("Critical impact",str(crit),"",th["red"],">70% supply lost"),a_kri("Country",country,"",th["accent"],"Being removed")],style={"display":"flex","gap":"12px","flexWrap":"wrap"})
    fig=go.Figure(go.Bar(y=[r["product"] for r in results],x=[r["lost_share"] for r in results],orientation="h",
        marker_color=[r["colour"] for r in results],text=[f"{r['lost_share']}%" for r in results],textposition="inside",insidetextanchor="middle",textfont=dict(size=11,color="#ffffff")))
    themed_layout(fig,ASEC,False); fig.update_xaxes(title=dict(text="Supply Lost (%)",font=dict(color=th["muted"],size=11),standoff=12),range=[0,120],ticksuffix="%")
    fig
    alts=imp[imp["partnerdesc"]!=country].groupby("partnerdesc")["primaryvalue"].sum(); tot_imp=imp["primaryvalue"].sum() or 1
    alt_els=[html.Div([html.Div([html.Span(p,style={"color":th["text"],"fontSize":"12px","flex":"1"}),html.Span(f"{v/tot_imp*100:.1f}%",style={"color":th["accent"],"fontSize":"12px","fontWeight":"600"})],style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"}),
        html.Div(style={"background":BG_INPUT,"borderRadius":"3px","height":"8px","overflow":"hidden","marginBottom":"8px"},children=[html.Div(style={"width":f"{min(v/tot_imp*100/40*100,100):.0f}%","height":"100%","background":th["accent"],"borderRadius":"3px"})])]) for p,v in alts.sort_values(ascending=False).head(6).items()]
    return cards,fig,alt_els

make_raw_toggle("a-d6-raw-toggle-btn","a-d6-raw-panel","a-d6-raw-content","a-d6-raw-source-tabs",[
    ("Comtrade Imports",agri_ct[agri_ct["flowcode"]=="M"].head(500) if not agri_ct.empty else None,["refyear","partnerdesc","cmddesc","primaryvalue"]),
])

def a_d7_layout():
    return html.Div([
        a_header("Agriculture -- Seasonal Risk","Monthly trade patterns, seasonal commodity cycles, CSO trade trends",[yr_dropdown_generic("a-d7-year",AGRI_DD_YEARS,multi=True,default=2023)]),
        html.Div(id="a-d7-kpi",style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(a_card("Monthly trade pattern (CSO)",[a_graph("a-d7-monthly",260)],flex="2", info_id="Monthly trade pattern (CSO)"),
            a_card("Seasonal risk signal",[a_graph("a-d7-risk",260)],flex="1", info_id="Seasonal risk signal")),
        a_card("CSO agricultural output trend",[a_graph("a-d7-output",240)], info_id="CSO agricultural output trend"),
        raw_data_section("a-d7",[
            ("CSO Monthly Trade",agri_trade.head(500) if not agri_trade.empty else None,["year","month","flow","commodity","value","unit"]),
            ("CSO Output",agri_out.head(500) if not agri_out.empty else None,["year","metric","value","unit"]),
        ]),
    ])

@app.callback(Output("a-d7-kpi","children"),Output("a-d7-monthly","figure"),Output("a-d7-risk","figure"),Output("a-d7-output","figure"),
              Input("url","pathname"),Input("a-d7-year","value"))
def cb_a_d7(pathname,year_val):
    th=THEMES[ASEC]; year=resolve_year(year_val,AGRI_CT_YMAX)
    # Month name → calendar number for chronological sort (Fix #16)
    MONTH_ORDER = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                   'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
    def _month_num(label):
        """Parse '2023 April' → 4; fallback to 0 if unparseable."""
        if pd.isna(label): return 0
        parts = str(label).strip().split()
        for p in parts:
            if p in MONTH_ORDER: return MONTH_ORDER[p]
        return 0

    # Monthly pattern from CSO
    fig_monthly=go.Figure()
    if not agri_trade.empty:
        yr_data=agri_trade[agri_trade["year"]==year]
        if not yr_data.empty:
            imp_m=yr_data[yr_data["flow"].str.contains("Import",case=False,na=False)].groupby("month")["value"].sum()
            exp_m=yr_data[yr_data["flow"].str.contains("Export",case=False,na=False)].groupby("month")["value"].sum()
            # Fix #16: sort by calendar month, not alphabetically
            months_sorted = sorted(imp_m.index.tolist(), key=_month_num)
            if months_sorted:
                fig_monthly.add_trace(go.Bar(x=months_sorted,y=[imp_m.get(m,0) for m in months_sorted],name="Imports",
        text=[f"EUR {imp_m.get(m,0)/1e9:.1f}bn" for m in months_sorted],textposition="inside",insidetextanchor="middle",textfont=dict(size=9,color="#ffffff"),marker_color=th["c3"],opacity=0.85))
                fig_monthly.add_trace(go.Bar(x=months_sorted,y=[exp_m.get(m,0) for m in months_sorted],name="Exports",marker_color=th["c1"],opacity=0.85))
    themed_layout(fig_monthly,ASEC,True); fig_monthly.update_layout(barmode="group")
    fig_monthly.update_xaxes(title=dict(text="Month (chronological)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_monthly.update_yaxes(title=dict(text="Value (EUR Thousand)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_monthly.update_layout(legend=dict(orientation="h",y=-0.25,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=11)))

    # Fix #8: Seasonal Risk signal computed from actual CSO monthly data.
    # Method: for the selected year, compute monthly total trade (imports+exports),
    # z-score vs the 12-month mean, aggregate to 6 bimonthly periods, map |z| → 0-100.
    categories=["Jan-Feb","Mar-Apr","May-Jun","Jul-Aug","Sep-Oct","Nov-Dec"]
    values = [0, 0, 0, 0, 0, 0]
    if not agri_trade.empty:
        yr_data = agri_trade[agri_trade["year"]==year]
        if not yr_data.empty:
            # Combined monthly volume (imp + exp)
            combined = yr_data.groupby("month")["value"].sum()
            if len(combined) >= 3:
                # Attach calendar month number, build 12-slot series
                monthly_vals = {}
                for month_label, val in combined.items():
                    mnum = _month_num(month_label)
                    if mnum > 0:
                        monthly_vals[mnum] = monthly_vals.get(mnum, 0) + val
                if monthly_vals:
                    mean_v = np.mean(list(monthly_vals.values()))
                    std_v = np.std(list(monthly_vals.values())) or 1.0
                    # Bimonthly pairs (1-2, 3-4, ..., 11-12)
                    for idx, (m1, m2) in enumerate([(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)]):
                        bi_sum = monthly_vals.get(m1, 0) + monthly_vals.get(m2, 0)
                        bi_mean = mean_v * 2  # two months' worth of mean
                        # |z-score| scaled to 0-100 (cap at 100)
                        z = abs(bi_sum - bi_mean) / (std_v * 2) if std_v > 0 else 0
                        values[idx] = min(100, round(z * 50, 1))  # z=1 → 50, z=2 → 100

    fig_risk=go.Figure(go.Bar(x=categories,y=values,marker_color=[stress_colour(v) for v in values],
        text=[f"{v:.0f}  {stress_label(v)}" for v in values],textposition="inside",insidetextanchor="middle",textfont=dict(size=10,color="#ffffff")))
    themed_layout(fig_risk,ASEC,False)
    fig_risk.update_xaxes(title=dict(text="Bimonth Period",font=dict(color=th["muted"],size=11),standoff=12))
    fig_risk.update_yaxes(title=dict(text="Seasonal Deviation Score (|z|-based)",font=dict(color=th["muted"],size=11),standoff=12))
    # Output trend
    fig_out=go.Figure()
    if not agri_out.empty:
        livestock=agri_out[agri_out["metric"].str.contains("livestock",case=False,na=False)].sort_values("year")
        crops=agri_out[agri_out["metric"].str.contains("crop",case=False,na=False)].sort_values("year")
        if not livestock.empty:
            fig_out.add_trace(go.Scatter(x=livestock["year"],y=livestock["value"],mode="lines+markers",name="All Livestock",line=dict(color=th["c1"],width=2),marker=dict(size=4)))
        if not crops.empty:
            fig_out.add_trace(go.Scatter(x=crops["year"],y=crops["value"],mode="lines+markers",name="All Crops",line=dict(color=th["c3"],width=2),marker=dict(size=4)))
    themed_layout(fig_out,ASEC,True)
    fig_out.update_xaxes(title=dict(text="Year",font=dict(color=th["muted"],size=11),standoff=12))
    fig_out.update_yaxes(title=dict(text="Value (EUR Million)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_out.update_layout(legend=dict(orientation="h",y=-0.25,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=11)))
    total_exp=agri_ct[agri_ct["flowcode"]=="X"]["primaryvalue"].sum() if not agri_ct.empty else 0
    cards=html.Div([a_kri("Annual agri exports",fmt_val(total_exp),"",th["accent"],"All years"),a_kri("CSO trade records",str(len(agri_trade)),"",th["c2"],"Monthly rows"),a_kri("Output metrics",str(agri_out["metric"].nunique()) if not agri_out.empty else "0","",th["c3"],"Unique metrics")],style={"display":"flex","gap":"12px","flexWrap":"wrap"})
    return cards,fig_monthly,fig_risk,fig_out

make_raw_toggle("a-d7-raw-toggle-btn","a-d7-raw-panel","a-d7-raw-content","a-d7-raw-source-tabs",[
    ("CSO Monthly Trade",agri_trade.head(500) if not agri_trade.empty else None,["year","month","flow","commodity","value","unit"]),
    ("CSO Output",agri_out.head(500) if not agri_out.empty else None,["year","metric","value","unit"]),
])

def a_d8_layout():
    return html.Div([
        a_header("Agriculture -- Food Security","FAOSTAT food balance analysis -- production, supply, demand per capita",[yr_dropdown_generic("a-d8-year",sorted(faostat["year"].dropna().astype(int).unique().tolist(),reverse=True) if not faostat.empty else [2023],multi=True,default=2023)]),
        html.Div(id="a-d8-kpi",style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(a_card("Production vs domestic supply",[a_graph("a-d8-balance",280)],flex="2", info_id="Production vs domestic supply"),
            a_card("Food categories breakdown",[a_graph("a-d8-items",280)],flex="1", info_id="Food categories breakdown")),
        a_card("Per capita food supply trend (kg/capita/yr)",[a_graph("a-d8-percapita",240)], info_id="Per capita food supply trend (kg/capita/yr)"),
        raw_data_section("a-d8",[("FAOSTAT",faostat.head(500) if not faostat.empty else None,["year","element","item","value","unit"])]),
    ])

@app.callback(Output("a-d8-kpi","children"),Output("a-d8-balance","figure"),Output("a-d8-items","figure"),Output("a-d8-percapita","figure"),
              Input("url","pathname"),Input("a-d8-year","value"))
def cb_a_d8(pathname,year_val):
    th=THEMES[ASEC]; year=resolve_year(year_val,max(faostat["year"]) if not faostat.empty else 2023)
    if not faostat.empty:
        fao_years_d8 = sorted(faostat["year"].dropna().unique().astype(int).tolist())
        fao_year_d8 = year if year in fao_years_d8 else max((y for y in fao_years_d8 if y<=year),default=max(fao_years_d8))
    else:
        fao_year_d8 = year
    fao_fallback_d8 = fao_year_d8 != year
    fy=faostat[faostat["year"]==fao_year_d8] if not faostat.empty else pd.DataFrame()
    prod  = fy[fy["element"].str.contains("production",      case=False,na=False)]["value"].sum() if not fy.empty else 0
    dom   = fy[fy["element"].str.contains("domestic supply",  case=False,na=False)]["value"].sum() if not fy.empty else 0
    imp_q = fy[fy["element"].str.contains("import quantity",  case=False,na=False)]["value"].sum() if not fy.empty else 0
    exp_q = fy[fy["element"].str.contains("export quantity",  case=False,na=False)]["value"].sum() if not fy.empty else 0
    # Fix #9: item-matched SSR (inner join on 'item') instead of sum of all rows
    if not fy.empty:
        prod_rows = fy[fy["element"].str.contains("production", case=False, na=False)][["item","value"]]
        dom_rows  = fy[fy["element"].str.contains("domestic supply", case=False, na=False)][["item","value"]]
        merged = prod_rows.merge(dom_rows, on="item", how="inner", suffixes=("_p","_d"))
        if not merged.empty and merged["value_d"].sum() > 0:
            ss = round(merged["value_p"].sum() / merged["value_d"].sum() * 100, 1)
        else:
            ss = 0
    else:
        ss = 0
    fao_note_d8 = f"FAOSTAT {fao_year_d8}" + (" (latest available)" if fao_fallback_d8 else "")
    cards=html.Div([
        a_kri("Food self-sufficiency",f"{ss:.0f}","%",th["accent"] if ss>100 else th["red"],f"Item-matched, {fao_note_d8}"),
        a_kri("Production",f"{prod/1000:.0f}","k tonnes","",fao_note_d8),
        a_kri("Food import quantity",f"{imp_q/1000:.0f}","k tonnes",th["c3"],fao_note_d8),
        a_kri("Food export quantity",f"{exp_q/1000:.0f}","k tonnes",th["c1"],fao_note_d8),
    ],style={"display":"flex","gap":"12px","flexWrap":"wrap"})
    fig_bal=go.Figure()
    if not fy.empty:
        elements=["Production","Import quantity","Domestic supply quantity","Export quantity"]
        vals=[fy[fy["element"]==e]["value"].sum()/1000 for e in elements]
        cols=[th["c1"],th["c3"],th["c2"],th["accent"]]
        fig_bal.add_trace(go.Bar(x=elements,y=vals,marker_color=cols,text=[f"{v:.0f}k t" for v in vals],textposition="inside",insidetextanchor="middle",textfont=dict(size=11,color="#ffffff")))
    themed_layout(fig_bal,ASEC,False); fig_bal.update_xaxes(title=dict(text="Food Balance Element",font=dict(color=th["muted"],size=11),standoff=12))
    fig_bal.update_yaxes(title=dict(text="Quantity (thousand tonnes)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_items=go.Figure()
    if not fy.empty:
        top_items=fy[fy["element"]=="Production"].groupby("item")["value"].sum().sort_values(ascending=False).head(8)
        if not top_items.empty:
            fig_items.add_trace(go.Pie(labels=list(top_items.index),values=list(top_items.values),hole=0.4,
                marker=dict(colors=[th["c1"],th["c2"],th["c3"],th["c4"],th["c5"],th["accent"],th["c1"],th["c2"]]),
                textfont=dict(size=10,color=th["text"])))
    themed_layout(fig_items,ASEC,True); fig_items.update_layout(paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=10),legend=dict(orientation="h",y=-0.3,x=0,font=dict(size=10,color=th["muted"]),bgcolor="rgba(0,0,0,0)"))
    fig_pc=go.Figure()
    if not faostat.empty:
        pc=faostat[faostat["element"].str.contains("food supply",case=False,na=False)].groupby("year")["value"].sum().reset_index()
        if not pc.empty:
            fig_pc.add_trace(go.Scatter(x=pc["year"],y=pc["value"],mode="lines+markers",line=dict(color=th["accent"],width=2.5),marker=dict(size=5,color=th["accent"]),fill="tozeroy",fillcolor=hex_rgba(th["accent"],0.08)))
    themed_layout(fig_pc,ASEC,False); fig_pc.update_xaxes(title=dict(text="Year",font=dict(color=th["muted"],size=11),standoff=12))
    fig_pc.update_yaxes(title=dict(text="kg / capita / year",font=dict(color=th["muted"],size=11),standoff=12))
    return cards,fig_bal,fig_items,fig_pc

make_raw_toggle("a-d8-raw-toggle-btn","a-d8-raw-panel","a-d8-raw-content","a-d8-raw-source-tabs",[
    ("FAOSTAT",faostat.head(500) if not faostat.empty else None,["year","element","item","value","unit"]),
])


# ???????????????????????????????????????????????????????
# =============================================================
# MEDTECH DASHBOARDS (8 dashboards - new Comtrade data)
# HS parent codes: 9018 Medical Instruments, 9021 Implants & Stents, 9022 X-Ray Apparatus
# HS sub-codes:  901839 Catheters, 901890 Other Instruments, 902190 Cardiovascular Implants
# RULE: parent codes for overview/totals. Sub-codes for product analysis. Never mixed.
# =============================================================
MSEC = "medtech"

# ── HS code constants ─────────────────────────────────────────────────────────
MT_PARENT_CODES  = ['9018','9021','9022']
MT_SUB_CODES     = ['901839','901890','902190']
MT_PARENT_LABELS = {'9018':'Medical Instruments','9021':'Implants and Stents','9022':'X-Ray Apparatus'}
MT_SUB_LABELS    = {'901839':'Catheters and Cannulae','901890':'Other Medical Instruments','902190':'Cardiovascular Implants and Stents'}

def load_medtech_new():
    import os, pandas as pd
    path = os.path.join(PROC_DIR, "medtech_new_comtrade.csv")
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df['refYear']      = pd.to_numeric(df['refYear'],      errors='coerce').fillna(0).astype(int)
    df['primaryValue'] = pd.to_numeric(df['primaryValue'], errors='coerce').fillna(0)
    # cmdCode is saved as int64 in CSV (9018 not '9018') - force to string for all comparisons
    df['cmdCode']      = df['cmdCode'].astype(str).str.strip()
    # Recompute codeType from string cmdCode to ensure consistency
    _parent = ['9018','9021','9022']
    _sub    = ['901839','901890','902190']
    df['codeType']  = df['cmdCode'].apply(lambda x: 'parent' if x in _parent else ('sub' if x in _sub else 'other'))
    df['isParent']  = df['cmdCode'].isin(_parent)
    df['isSub']     = df['cmdCode'].isin(_sub)
    return df

mt_ct = load_medtech_new()

MT_CT_YEARS = sorted(mt_ct['refYear'].unique().astype(int).tolist()) if not mt_ct.empty else []
MT_CT_YMAX  = max(MT_CT_YEARS) if MT_CT_YEARS else 2024
MT_DD_YEARS = sorted([y for y in MT_CT_YEARS if y >= MT_CT_YMAX-10], reverse=True) or [2024]
MT_PARTNERS = sorted(mt_ct['partnerISO'].dropna().unique().tolist()) if not mt_ct.empty else []

def mt_nearest(year):
    if not MT_CT_YEARS: return MT_CT_YMAX
    year = int(year)
    if year in MT_CT_YEARS: return year
    below = [y for y in MT_CT_YEARS if y <= year]
    return max(below) if below else MT_CT_YMAX

def mt_filter(year, flow=None, code_type=None, codes=None):
    """Filter mt_ct by year, flow, code_type ('parent'/'sub') or specific codes list.
    All comparisons use string dtype to avoid int/str mismatch after CSV reload."""
    if mt_ct.empty: return pd.DataFrame()
    d = mt_ct[mt_ct['refYear']==mt_nearest(year)].copy()
    if flow:      d = d[d['flowCode']==flow]
    if code_type: d = d[d['codeType']==code_type]
    if codes:
        # Force both sides to string for safe comparison
        str_codes = [str(c) for c in codes]
        d = d[d['cmdCode'].astype(str).isin(str_codes)]
    return d

def m_header(title, subtitle, controls):
    th = THEMES[MSEC]
    return html.Div([
        html.Div([
            html.H2(title, style={"color":th["text"],"fontWeight":"600","margin":"0 0 2px","fontSize":"20px"}),
            html.P(subtitle, style={"color":th["muted"],"margin":0,"fontSize":"12px"}),
        ], style={"flex":"1"}),
        *controls,
    ], style={"display":"flex","gap":"24px","alignItems":"flex-end","marginBottom":"16px",
              "background":th["card"],"border":f"1px solid {th['border']}","borderRadius":"8px","padding":"16px"})

def m_card(title, children, badge=None, flex=None, info_id=None): return dark_card(title, children, badge=badge, flex=flex, sector=MSEC, info_id=info_id)
def m_kri(title, value, unit="", colour=None, subtitle=""): return kri_card(title, value, unit, colour, subtitle, MSEC)
def m_graph(gid, height=280): return dark_graph(gid, height)

# ── M-D1: Strategic Overview ──────────────────────────────────────────────────
def m_d1_layout():
    th = THEMES[MSEC]
    return html.Div([
        m_header("MedTech -- Strategic Overview",
                 "Parent codes only: 9018 Medical Instruments . 9021 Implants & Stents . 9022 X-Ray Apparatus",
                 [yr_dropdown_generic("m-d1-year", MT_DD_YEARS, multi=True, default=2023),
                  html.Div([
                      html.Label("View", style={"fontSize":"11px","color":th["muted"],"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="m-d1-view",
                          options=[{"label":"Sector View","value":"sector"},{"label":"Product View","value":"product"}],
                          value="sector", clearable=False,
                          style={"fontSize":"13px","width":"160px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}"}),
                  ])]),
        html.Div(id="m-d1-kri", style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(m_card("Export value by sector",[m_graph("m-d1-exp-bar",240)], flex="2", info_id="Export value by sector"),
            m_card("Key risk signals",[html.Div(id="m-d1-insights",style={"overflowY":"auto","maxHeight":"260px"})], flex="1", info_id="Key risk signals")),
        row(m_card("Import value by sector",[m_graph("m-d1-imp-bar",200)], flex="2", info_id="Import value by sector"),
            m_card("Import vs export balance",[m_graph("m-d1-balance",200)], flex="1", info_id="Import vs export balance")),
        raw_data_section("m-d1",[
            ("Parent Codes",mt_ct[mt_ct['codeType']=='parent'].head(500) if not mt_ct.empty else None,
             ["refYear","flowCode","partnerISO","cmdCode","hsLabel","primaryValue","continent"]),
        ]),
    ])

@app.callback(
    Output("m-d1-kri","children"), Output("m-d1-exp-bar","figure"),
    Output("m-d1-insights","children"), Output("m-d1-imp-bar","figure"),
    Output("m-d1-balance","figure"),
    Input("url","pathname"), Input("m-d1-year","value"), Input("m-d1-view","value"))
def cb_m_d1(pathname, year_val, view):
    th = THEMES[MSEC]
    year = resolve_year(year_val, MT_CT_YMAX)
    exp_p = mt_filter(year,'X','parent'); imp_p = mt_filter(year,'M','parent')
    exp_s = mt_filter(year,'X','sub');   imp_s = mt_filter(year,'M','sub')

    exp_total = exp_p['primaryValue'].sum(); imp_total = imp_p['primaryValue'].sum()
    surplus   = exp_total - imp_total
    n_exp_mkts = exp_p['partnerISO'].nunique(); n_imp_src = imp_p['partnerISO'].nunique()
    top_exp_mkt = "N/A"; top_exp_sh = 0.0
    if not exp_p.empty:
        ge = exp_p.groupby('partnerISO')['primaryValue'].sum()
        if len(ge)>0: top_exp_mkt = ge.idxmax(); top_exp_sh = round(ge.max()/ge.sum()*100,1)
    top_imp_src = "N/A"; top_imp_sh = 0.0
    if not imp_p.empty:
        gi = imp_p.groupby('partnerISO')['primaryValue'].sum()
        hhi_v = round(sum((v/gi.sum())**2 for v in gi.values),4) if gi.sum()>0 else 0
        if len(gi)>0: top_imp_src = gi.idxmax(); top_imp_sh = round(gi.max()/gi.sum()*100,1)
    else: hhi_v = 0

    cards = html.Div([
        m_kri("MedTech exports", fmt_val(exp_total),"", th["accent"], f"Parent codes {year}"),
        m_kri("MedTech imports", fmt_val(imp_total),"", th["c3"],    f"Parent codes {year}"),
        m_kri("Trade surplus",   fmt_val(surplus),  "", th["green"] if surplus>0 else th["red"], "Net trade balance"),
        m_kri("Import HHI",      f"{hhi_v:.3f}",    "", hhi_colour(hhi_v), hhi_label(hhi_v)),
        m_kri("Top export market", top_exp_mkt,     "", th["c2"], f"{top_exp_sh:.1f}% of exports"),
        m_kri("Top import source", top_imp_src,     "", th["c3"], f"{top_imp_sh:.1f}% of imports"),
        m_kri("Export markets",  str(n_exp_mkts),   "", th["accent2"], f"Countries {year}"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap"})

    PAL = [th["c1"],th["c2"],th["c3"],th["c4"],th["c5"]]

    # Export bar
    fig_exp = go.Figure()
    if view == "sector":
        data_exp = [(MT_PARENT_LABELS.get(c,c), exp_p[exp_p['cmdCode'].astype(str)==str(c)]['primaryValue'].sum()) for c in MT_PARENT_CODES]
    else:
        data_exp = [(MT_SUB_LABELS.get(c,c), exp_s[exp_s['cmdCode'].astype(str)==str(c)]['primaryValue'].sum()) for c in MT_SUB_CODES]
    data_exp = sorted(data_exp, key=lambda x: x[1])
    labels_e, vals_e = zip(*data_exp) if data_exp else ([],[])
    fig_exp.add_trace(go.Bar(y=list(labels_e), x=[v/1e9 for v in vals_e], orientation="h",
        marker_color=PAL[:len(labels_e)],
        text=[f"EUR {v/1e9:.2f}bn" for v in vals_e], textposition="outside",
        textfont=dict(size=10, color=th["text"])))
    themed_layout(fig_exp, MSEC, False)
    fig_exp.update_xaxes(title=dict(text="Export Value (EUR Billion)", font=dict(color=th["muted"],size=11),standoff=12))
    fig_exp.update_yaxes(title=dict(text="Sector" if view=="sector" else "Product", font=dict(color=th["muted"],size=11),standoff=12))

    # Insights
    insights = []
    if exp_total > 10e9: insights.append(("STRONG",th["accent"],"Export Leader", f"EUR {exp_total/1e9:.1f}bn exports -- Ireland is top EU MedTech hub"))
    if hhi_v > 0.25:     insights.append(("RISK",th["red"],"Concentration Risk", f"Import HHI {hhi_v:.3f} -- above moderate threshold"))
    elif hhi_v > 0.15:   insights.append(("WATCH",th["amber"],"Moderate HHI", f"Import HHI {hhi_v:.3f} -- monitor for further concentration"))
    if top_imp_sh > 30:  insights.append(("WATCH",th["amber"],"Import Dependency", f"{top_imp_src} = {top_imp_sh:.1f}% of imports -- single market risk"))
    if surplus > 10e9:   insights.append(("STRONG",th["c2"],"Trade Surplus", f"EUR {surplus/1e9:.1f}bn surplus -- strong net exporter position"))
    if not insights:     insights.append(("OK",th["accent"],"No Alerts","All MedTech indicators within acceptable ranges"))
    ins_els = [insight_item(s,c,t,m) for s,c,t,m in insights]

    # Import bar
    fig_imp = go.Figure()
    if view == "sector":
        data_imp = [(MT_PARENT_LABELS.get(c,c), imp_p[imp_p['cmdCode'].astype(str)==str(c)]['primaryValue'].sum()) for c in MT_PARENT_CODES]
    else:
        data_imp = [(MT_SUB_LABELS.get(c,c), imp_s[imp_s['cmdCode'].astype(str)==str(c)]['primaryValue'].sum()) for c in MT_SUB_CODES]
    data_imp = sorted(data_imp, key=lambda x: x[1])
    labels_i, vals_i = zip(*data_imp) if data_imp else ([],[])
    fig_imp.add_trace(go.Bar(y=list(labels_i), x=[v/1e9 for v in vals_i], orientation="h",
        marker_color=[th["c3"],th["c4"],th["c5"]][:len(labels_i)],
        text=[f"EUR {v/1e9:.2f}bn" for v in vals_i], textposition="outside",
        textfont=dict(size=10, color=th["text"])))
    themed_layout(fig_imp, MSEC, False)
    fig_imp.update_xaxes(title=dict(text="Import Value (EUR Billion)", font=dict(color=th["muted"],size=11),standoff=12))
    fig_imp.update_yaxes(title=dict(text="Sector" if view=="sector" else "Product", font=dict(color=th["muted"],size=11),standoff=12))

    # Balance bar
    fig_bal = go.Figure()
    codes_b = MT_PARENT_CODES if view=="sector" else MT_SUB_CODES
    labels_b = MT_PARENT_LABELS if view=="sector" else MT_SUB_LABELS
    data_src = (exp_p, imp_p) if view=="sector" else (exp_s, imp_s)
    bal_labels = [labels_b.get(c,c) for c in codes_b]
    bal_vals   = [(data_src[0][data_src[0]['cmdCode']==c]['primaryValue'].sum() -
                   data_src[1][data_src[1]['cmdCode']==c]['primaryValue'].sum())/1e9 for c in codes_b]
    fig_bal.add_trace(go.Bar(x=bal_labels, y=bal_vals,
        marker_color=[th["green"] if v>=0 else th["red"] for v in bal_vals],
        text=[f"EUR {v:.2f}bn" for v in bal_vals],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=11,color="#ffffff")))
    themed_layout(fig_bal, MSEC, False)
    fig_bal.update_xaxes(title=dict(text="Sector", font=dict(color=th["muted"],size=11),standoff=12))
    fig_bal.update_yaxes(title=dict(text="Trade Balance (EUR Billion)", font=dict(color=th["muted"],size=11),standoff=12))
    return cards, fig_exp, ins_els, fig_imp, fig_bal

make_raw_toggle("m-d1-raw-toggle-btn","m-d1-raw-panel","m-d1-raw-content","m-d1-raw-source-tabs",[
    ("Parent Codes", mt_ct[mt_ct['codeType']=='parent'].head(500) if not mt_ct.empty else None,
     ["refYear","flowCode","partnerISO","cmdCode","hsLabel","primaryValue"]),
])


# ── M-D2: Import Dependency ───────────────────────────────────────────────────
def m_d2_layout():
    th = THEMES[MSEC]
    return html.Div([
        m_header("MedTech -- Import Dependency",
                 "HHI per sector (parent codes). Sector View = per HS parent. Product View = per sub-code.",
                 [yr_dropdown_generic("m-d2-year", MT_DD_YEARS, multi=True, default=2023),
                  html.Div([
                      html.Label("View", style={"fontSize":"11px","color":th["muted"],"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="m-d2-view",
                          options=[{"label":"Sector View","value":"sector"},{"label":"Product View","value":"product"}],
                          value="sector", clearable=False,
                          style={"fontSize":"13px","width":"160px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}"}),
                  ])]),
        html.Div(id="m-d2-kri", style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(m_card("HHI by sector",[m_graph("m-d2-hhi",260)], flex="1",
                   badge="Sector View=parent codes | Product View=sub-codes", info_id="HHI by sector"),
            m_card("Top import sources",[m_graph("m-d2-sources",260)], flex="1", info_id="Top import sources")),
        row(m_card("Continental breakdown",[m_graph("m-d2-cont",220)], flex="1", info_id="Continental breakdown"),
            m_card("Import trend 2015-2024",[m_graph("m-d2-trend",220)], flex="1", info_id="Import trend 2015-2024")),
        raw_data_section("m-d2",[
            ("Parent Code Imports", mt_ct[(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
             ["refYear","partnerISO","cmdCode","hsLabel","primaryValue","continent"]),
            ("Sub-Code Imports", mt_ct[(mt_ct['codeType']=='sub')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
             ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
        ]),
    ])

@app.callback(
    Output("m-d2-kri","children"), Output("m-d2-hhi","figure"),
    Output("m-d2-sources","figure"), Output("m-d2-cont","figure"), Output("m-d2-trend","figure"),
    Input("url","pathname"), Input("m-d2-year","value"), Input("m-d2-view","value"))
def cb_m_d2(pathname, year_val, view):
    th = THEMES[MSEC]
    year = resolve_year(year_val, MT_CT_YMAX)
    code_type = "parent" if view=="sector" else "sub"
    codes = MT_PARENT_CODES if view=="sector" else MT_SUB_CODES
    labels_map = MT_PARENT_LABELS if view=="sector" else MT_SUB_LABELS
    imp = mt_filter(year,'M',code_type)

    imp_total = imp['primaryValue'].sum() or 1
    n_partners = imp['partnerISO'].nunique()
    top_src = "N/A"; top_sh = 0.0
    if not imp.empty:
        gi = imp.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False)
        if len(gi)>0: top_src=gi.index[0]; top_sh=round(gi.iloc[0]/gi.sum()*100,1)
    overall_hhi = round(sum((v/imp_total)**2 for v in imp.groupby('partnerISO')['primaryValue'].sum().values),4) if not imp.empty else 0

    cards = html.Div([
        m_kri("Total imports", fmt_val(imp_total),"", th["c3"], f"{code_type.title()} codes {year}"),
        m_kri("Top import source", top_src,"", th["accent"], f"{top_sh:.1f}% share"),
        m_kri("Overall HHI", f"{overall_hhi:.3f}","", hhi_colour(overall_hhi), hhi_label(overall_hhi)),
        m_kri("Import partners", str(n_partners),"", th["accent2"], "Unique countries"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap"})

    # HHI by code
    fig_hhi = go.Figure()
    hhi_data = []
    for code in codes:
        sub = imp[imp['cmdCode'].astype(str)==str(code)]
        tot = sub['primaryValue'].sum() or 1
        h = round(sum((v/tot)**2 for v in sub.groupby('partnerISO')['primaryValue'].sum().values),4) if not sub.empty else 0
        hhi_data.append((labels_map.get(code,code), h))
    hhi_data.sort(key=lambda x: x[1])
    if hhi_data:
        yl, xv = zip(*hhi_data)
        fig_hhi.add_trace(go.Bar(y=list(yl), x=list(xv), orientation="h",
            marker_color=[hhi_colour(h) for h in xv],
            text=[f"{h:.4f}  {hhi_label(h)}" for h in xv], textposition="outside",
            textfont=dict(size=10, color=th["text"])))
    themed_layout(fig_hhi, MSEC, False)
    fig_hhi.update_xaxes(title=dict(text="HHI Score (0=diversified, 1=single source)", font=dict(color=th["muted"],size=11),standoff=12))
    fig_hhi.update_yaxes(title=dict(text="Sector / Product", font=dict(color=th["muted"],size=11),standoff=12))

    # Top sources
    fig_src = go.Figure()
    if not imp.empty:
        gs = imp.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False).head(10)
        fig_src.add_trace(go.Bar(y=list(gs.index), x=list(gs.values/1e9), orientation="h",
            marker_color=[th["c1"],th["c2"],th["c3"],th["c4"],th["c5"]]*2,
            text=[f"EUR {v/1e9:.2f}bn" for v in gs.values], textposition="outside",
            textfont=dict(size=10, color=th["text"])))
    themed_layout(fig_src, MSEC, False)
    fig_src.update_xaxes(title=dict(text="Import Value (EUR Billion)", font=dict(color=th["muted"],size=11),standoff=12))
    fig_src.update_yaxes(title=dict(text="Country", font=dict(color=th["muted"],size=11),standoff=12))

    # Continental breakdown
    fig_cont = go.Figure()
    if not imp.empty and 'continent' in imp.columns:
        gc = imp.groupby('continent')['primaryValue'].sum().sort_values(ascending=False)
        fig_cont.add_trace(go.Bar(x=list(gc.index), y=list(gc.values/1e9),
            text=[f"EUR {v/1e9:.2f}bn" for v in gc.values],textposition="inside",insidetextanchor="middle",textfont=dict(size=10,color="#ffffff"),
            marker_color=th["c1"]))
    themed_layout(fig_cont, MSEC, False)
    fig_cont.update_xaxes(title=dict(text="Region", font=dict(color=th["muted"],size=11),standoff=12))
    fig_cont.update_yaxes(title=dict(text="Import Value (EUR Billion)", font=dict(color=th["muted"],size=11),standoff=12))

    # Trend
    fig_trend = go.Figure()
    if not mt_ct.empty:
        trend = [{"year":y, "imp": mt_ct[(mt_ct['refYear']==y)&(mt_ct['codeType']==code_type)&(mt_ct['flowCode']=='M')]['primaryValue'].sum()/1e9}
                 for y in sorted(mt_ct['refYear'].unique())]
        df_t = pd.DataFrame(trend)
        if not df_t.empty:
            fig_trend.add_trace(go.Scatter(x=df_t["year"], y=df_t["imp"], mode="lines+markers",
                name="Imports (EUR bn)", line=dict(color=th["c3"],width=2.5),
                marker=dict(size=5, color=th["c3"])))
    themed_layout(fig_trend, MSEC, False)
    fig_trend.update_xaxes(title=dict(text="Year", font=dict(color=th["muted"],size=11),standoff=12))
    fig_trend.update_yaxes(title=dict(text="Import Value (EUR Billion)", font=dict(color=th["muted"],size=11),standoff=12))
    return cards, fig_hhi, fig_src, fig_cont, fig_trend

make_raw_toggle("m-d2-raw-toggle-btn","m-d2-raw-panel","m-d2-raw-content","m-d2-raw-source-tabs",[
    ("Parent Code Imports", mt_ct[(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
     ["refYear","partnerISO","cmdCode","hsLabel","primaryValue","continent"]),
    ("Sub-Code Imports", mt_ct[(mt_ct['codeType']=='sub')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
     ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
])

# ── M-D3: Export Dependency ───────────────────────────────────────────────────
def m_d3_layout():
    th = THEMES[MSEC]
    # Build market options
    mkt_opts = [{"label":"All markets (universal shock)","value":"__ALL__"}]
    if not mt_ct.empty:
        top_mkts = (mt_ct[(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='X')]
                    .groupby("partnerISO")["primaryValue"].sum().sort_values(ascending=False)
                    .head(20).index.tolist())
        mkt_opts += [{"label":m,"value":m} for m in top_mkts]
    return html.Div([
        m_header("MedTech -- Export Dependency",
                 "Export market concentration, tariff scenario impact (parent codes)",
                 [yr_dropdown_generic("m-d3-year", MT_DD_YEARS, multi=True, default=2023),
                  html.Div([
                      html.Label("Tariff Market", style={"fontSize":"11px","color":th["muted"],"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="m-d3-market",options=mkt_opts,
                                   value="USA" if any(o["value"]=="USA" for o in mkt_opts) else "__ALL__",
                                   clearable=False,style={"fontSize":"13px","width":"220px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}"})]),
                  html.Div([
                      html.Label("Tariff Rate", style={"fontSize":"11px","color":th["muted"],"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="m-d3-tariff",
                          options=[{"label":"0%","value":0.0},{"label":"10%","value":0.10},{"label":"25%","value":0.25},{"label":"50%","value":0.50}],
                          value=0.10, clearable=False,
                          style={"fontSize":"13px","width":"130px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}"}),
                  ]),
                  html.Div([
                      html.Label("Pass-through %",style={"fontSize":"11px","color":th["muted"],"display":"block","marginBottom":"4px"}),
                      dcc.Slider(id="m-d3-pt",min=0.50,max=1.00,step=0.05,value=0.75,
                                 marks={0.50:{"label":"50%","style":{"color":th["muted"],"fontSize":"10px"}},
                                        0.75:{"label":"75%","style":{"color":th["muted"],"fontSize":"10px"}},
                                        1.00:{"label":"100%","style":{"color":th["muted"],"fontSize":"10px"}}},
                                 tooltip={"placement":"bottom","always_visible":False})
                  ],style={"width":"170px","marginTop":"2px"})]),
        html.Div("Tariff impact = exports to selected market × tariff rate × pass-through. Default 75% per WTO research.",
                 style={"fontSize":"11px","color":th["muted"],"marginBottom":"8px","fontStyle":"italic"}),
        html.Div(id="m-d3-kri", style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(m_card("Top export markets (parent codes)",[m_graph("m-d3-markets",280)], flex="1", info_id="Top export markets (parent codes)"),
            m_card("Export value by sector",[m_graph("m-d3-sectors",280)], flex="1", info_id="Export value by sector")),
        m_card("Tariff scenario -- revenue at risk (EUR M)",[m_graph("m-d3-tariff-fig",240)], info_id="Tariff scenario -- revenue at risk (EUR M)"),
        raw_data_section("m-d3",[
            ("Parent Code Exports", mt_ct[(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='X')].head(500) if not mt_ct.empty else None,
             ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
        ]),
    ])

@app.callback(
    Output("m-d3-kri","children"), Output("m-d3-markets","figure"),
    Output("m-d3-sectors","figure"), Output("m-d3-tariff-fig","figure"),
    Input("url","pathname"), Input("m-d3-year","value"), Input("m-d3-tariff","value"),
    Input("m-d3-market","value"), Input("m-d3-pt","value"))
def cb_m_d3(pathname, year_val, tariff, market, pt):
    th = THEMES[MSEC]; year = resolve_year(year_val, MT_CT_YMAX); tariff = 0.10 if tariff is None else tariff
    pt = 0.75 if pt is None else float(pt)
    market = market or "__ALL__"
    exp = mt_filter(year,'X','parent')
    exp_total = exp['primaryValue'].sum() or 1
    n_markets = exp['partnerISO'].nunique()
    top_mkt = "N/A"; top_sh = 0.0
    if not exp.empty:
        gm = exp.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False)
        if len(gm)>0: top_mkt=gm.index[0]; top_sh=round(gm.iloc[0]/gm.sum()*100,1)
    # Base exposed to tariff depends on market selection
    if market == "__ALL__":
        base_for_tariff = exp_total
        mkt_label = "all markets"
    else:
        base_for_tariff = exp[exp['partnerISO']==market]['primaryValue'].sum()
        mkt_label = market
    rev_loss = (base_for_tariff/1e6)*tariff*pt

    cards = html.Div([
        m_kri("Total MedTech exports", fmt_val(exp_total),"", th["accent"], f"Parent codes {year}"),
        m_kri("Top export market", top_mkt,"", th["c2"], f"{top_sh:.1f}% of exports"),
        m_kri(f"Revenue at risk ({mkt_label})",
          f"EUR {rev_loss:.0f}M" if tariff > 0 else "EUR 0",
          "",
          th["green"] if tariff==0 else th["red"] if tariff>0.2 else th["amber"],
          f"{int(tariff*100)}% × {int(pt*100)}% pass-through"),
        m_kri("Export markets", str(n_markets),"", th["accent2"], "Unique destinations"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap"})

    fig_mkt = go.Figure()
    if not exp.empty:
        gm = exp.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False).head(10)
        fig_mkt.add_trace(go.Bar(y=list(gm.index), x=list(gm.values/1e9), orientation="h",
            marker_color=[th["c1"],th["c2"],th["c3"],th["c4"],th["c5"]]*2,
            text=[f"EUR {v/1e9:.2f}bn" for v in gm.values], textposition="outside",
            textfont=dict(size=10, color=th["text"])))
    themed_layout(fig_mkt, MSEC, False)
    fig_mkt.update_xaxes(title=dict(text="Export Value (EUR Billion)", font=dict(color=th["muted"],size=11),standoff=12))
    fig_mkt.update_yaxes(title=dict(text="Market", font=dict(color=th["muted"],size=11),standoff=12))

    fig_sec = go.Figure()
    sec_data = [(MT_PARENT_LABELS.get(c,c), exp[exp['cmdCode'].astype(str)==str(c)]['primaryValue'].sum()) for c in MT_PARENT_CODES]
    sec_data.sort(key=lambda x: x[1])
    if sec_data:
        yl, xv = zip(*sec_data)
        fig_sec.add_trace(go.Bar(y=list(yl), x=[v/1e9 for v in xv], orientation="h",
            marker_color=[th["c1"],th["c2"],th["c3"]],
            text=[f"EUR {v/1e9:.2f}bn" for v in xv], textposition="outside",
            textfont=dict(size=10, color=th["text"])))
    themed_layout(fig_sec, MSEC, False)
    fig_sec.update_xaxes(title=dict(text="Export Value (EUR Billion)", font=dict(color=th["muted"],size=11),standoff=12))
    fig_sec.update_yaxes(title=dict(text="Sector", font=dict(color=th["muted"],size=11),standoff=12))

    rates = [0.0, 0.05, 0.10, 0.25, 0.50]; base_M = base_for_tariff/1e6
    # Apply tariff × pass-through to the selected market's exports
    fig_tar = go.Figure(go.Bar(x=[f"{int(r*100)}%" for r in rates], y=[base_M*r*pt for r in rates],
        marker_color=[th["c4"],th["c3"],th["c2"],th["amber"],th["red"]],
        text=[f"EUR {base_M*r*pt:.0f}M" for r in rates], textposition="inside",
        insidetextanchor="middle",textfont=dict(size=10, color="#ffffff")))
    themed_layout(fig_tar, MSEC, False)
    fig_tar.update_xaxes(title=dict(text=f"Tariff rate applied to {mkt_label}",
                          font=dict(color=th["muted"],size=11),standoff=12))
    fig_tar.update_yaxes(title=dict(text=f"Annual Revenue Loss at {int(pt*100)}% p-t (EUR M)",
                          font=dict(color=th["muted"],size=11),standoff=12))
    # Add annotation explaining methodology
    if tariff > 0:
        fig_tar.add_annotation(
            text=f"{int(tariff*100)}% tariff × {int(pt*100)}% pass-through on {mkt_label}: EUR {rev_loss:.0f}M",
            xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False,
            font=dict(size=10, color=th["muted"]), xanchor="center")
    return cards, fig_mkt, fig_sec, fig_tar

make_raw_toggle("m-d3-raw-toggle-btn","m-d3-raw-panel","m-d3-raw-content","m-d3-raw-source-tabs",[
    ("Parent Code Exports", mt_ct[(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='X')].head(500) if not mt_ct.empty else None,
     ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
])


# ── M-D4: Supply Flow + Trend ─────────────────────────────────────────────────
def m_d4_layout():
    th = THEMES[MSEC]
    return html.Div([
        m_header("MedTech -- Supply Flow",
                 "Sankey + trade trend analysis (2015-2024). Sector View = parent codes. Product View = sub-codes.",
                 [yr_dropdown_generic("m-d4-year", MT_DD_YEARS, multi=True, default=2023),
                  html.Div([
                      html.Label("View", style={"fontSize":"11px","color":th["muted"],"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="m-d4-view",
                          options=[{"label":"Sector View","value":"sector"},{"label":"Product View","value":"product"}],
                          value="sector", clearable=False,
                          style={"fontSize":"13px","width":"160px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}"}),
                  ])]),
        m_card("Supply Flow Sankey",[m_graph("m-d4-sankey",380)], info_id="Supply Flow Sankey"),
        html.Div([html.Div(id="m-d4-imp-tbl"),html.Div(id="m-d4-exp-tbl"),html.Div(id="m-d4-summary")],
                 style={"display":"flex","gap":"12px","marginTop":"8px","marginBottom":"8px"}),
        html.Div([
            html.Div("TRADE TREND ANALYSIS (2015-2024)", style={
                "fontSize":"9px","fontWeight":"500","letterSpacing":"0.8px","color":th["muted"],
                "borderTop":f"1px solid {th['border']}","paddingTop":"10px","marginBottom":"8px","marginTop":"4px"}),
        ]),
        row(m_card("Import / export trend (EUR bn)",[m_graph("m-d4-trend",220)], flex="2", info_id="Import / export trend (EUR bn)"),
            m_card("CAGR summary",[html.Div(id="m-d4-cagr")], flex="1", info_id="CAGR summary")),
        raw_data_section("m-d4",[
            ("All Flows", mt_ct.head(500) if not mt_ct.empty else None,
             ["refYear","flowCode","partnerISO","cmdCode","hsLabel","primaryValue","continent"]),
        ]),
    ])

@app.callback(
    Output("m-d4-sankey","figure"), Output("m-d4-imp-tbl","children"),
    Output("m-d4-exp-tbl","children"), Output("m-d4-summary","children"),
    Output("m-d4-trend","figure"), Output("m-d4-cagr","children"),
    Input("url","pathname"), Input("m-d4-year","value"), Input("m-d4-view","value"))
def cb_m_d4(pathname, year_val, view):
    th = THEMES[MSEC]; year = resolve_year(year_val, MT_CT_YMAX)
    code_type = "parent" if view=="sector" else "sub"
    labels_map = MT_PARENT_LABELS if view=="sector" else MT_SUB_LABELS
    imp = mt_filter(year,'M',code_type); exp = mt_filter(year,'X',code_type)
    PAL = [th["c1"],th["c2"],th["c3"],th["c4"],th["c5"],th["accent"],th["accent2"],th["muted"]]

    # Country-level Sankey -- top 8 import sources, top 6 export markets
    JUNK_S = {"world","areas, nes","other","unspecified","not specified","total"}
    imp_c = imp[~imp['partnerISO'].str.strip().str.lower().isin(JUNK_S)]
    exp_c = exp[~exp['partnerISO'].str.strip().str.lower().isin(JUNK_S)] if not exp.empty else exp
    ic = imp_c.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False).head(8)
    ec = exp_c.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False).head(6) if not exp_c.empty else pd.Series(dtype=float)
    nodes = [f">> {r}" for r in ic.index]+["Ireland"]+[f"<< {r}" for r in ec.index]
    ni = {n:i for i,n in enumerate(nodes)}; src,tgt,vals,cols=[],[],[],[]
    for i,(region,val) in enumerate(ic.items()):
        src.append(ni[f">> {region}"]); tgt.append(ni["Ireland"])
        vals.append(val/1e9); cols.append(hex_rgba(PAL[i%len(PAL)],0.6))
    for region,val in ec.items():
        if f"<< {region}" in ni:
            src.append(ni["Ireland"]); tgt.append(ni[f"<< {region}"])
            vals.append(val/1e9); cols.append(hex_rgba(th["c1"],0.5))
    fig = go.Figure(go.Sankey(arrangement="snap",
        node=dict(pad=15,thickness=18,label=nodes,
                  color=[hex_rgba(PAL[i%len(PAL)],0.9) if i<len(ic) else
                         hex_rgba(th["accent"],0.9) if i==len(ic) else hex_rgba(th["c1"],0.9)
                         for i in range(len(nodes))],
                  line=dict(color=th["border"],width=0.5)),
        link=dict(source=src,target=tgt,value=vals,color=cols,
                  hovertemplate="%{source.label} >> %{target.label}<br>EUR %{value:.2f}bn<extra></extra>")))
    sankey_layout(fig,MSEC)

    def make_tbl(df, title):
        if df is None or df.empty: return m_card(title,[html.P("No data",style={"color":th["muted"],"fontSize":"12px"})],flex="1", info_id=title)
        g = df.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False).head(8)
        rows = [html.Div([html.Span(p,style={"fontSize":"11px","color":th["muted"],"flex":"1"}),
                          html.Span(fmt_val(v),style={"fontSize":"11px","color":th["text"],"fontWeight":"600"})],
                         style={"display":"flex","justifyContent":"space-between","padding":"5px 0","borderBottom":f"1px solid {th['border']}"})
                for p,v in g.items()]
        return m_card(title, rows, flex="1", info_id=title)
    summary = m_card("Flow Summary",[
        html.Div([html.Span("Total Imports",style={"color":th["muted"],"fontSize":"10px"}),
                  html.Span(fmt_val(imp["primaryValue"].sum()),style={"color":th["red"],"fontWeight":"700","fontSize":"15px"})],style={"marginBottom":"7px"}),
        html.Div([html.Span("Total Exports",style={"color":th["muted"],"fontSize":"10px"}),
                  html.Span(fmt_val(exp["primaryValue"].sum()) if not exp.empty else "N/A",style={"color":th["accent"],"fontWeight":"700","fontSize":"15px"})],style={"marginBottom":"7px"}),
        html.Div([html.Span("Surplus",style={"color":th["muted"],"fontSize":"10px"}),
                  html.Span(fmt_val(exp["primaryValue"].sum()-imp["primaryValue"].sum()),style={"color":th["green"],"fontWeight":"700","fontSize":"15px"})]),
    ], flex="1", info_id="Flow Summary")

    # Trend chart
    fig_trend = go.Figure()
    if not mt_ct.empty:
        yrs = sorted(mt_ct['refYear'].unique())
        exp_t=[mt_ct[(mt_ct['refYear']==y)&(mt_ct['codeType']==code_type)&(mt_ct['flowCode']=='X')]['primaryValue'].sum()/1e9 for y in yrs]
        imp_t=[mt_ct[(mt_ct['refYear']==y)&(mt_ct['codeType']==code_type)&(mt_ct['flowCode']=='M')]['primaryValue'].sum()/1e9 for y in yrs]
        fig_trend.add_trace(go.Scatter(x=yrs,y=exp_t,mode="lines+markers",name="Exports",
            line=dict(color=th["accent"],width=2.5),marker=dict(size=5,color=th["accent"])))
        fig_trend.add_trace(go.Scatter(x=yrs,y=imp_t,mode="lines+markers",name="Imports",
            line=dict(color=th["c3"],width=2,dash="dash"),marker=dict(size=5,color=th["c3"])))
    themed_layout(fig_trend,MSEC,True)
    fig_trend.update_xaxes(title=dict(text="Year",font=dict(color=th["muted"],size=11),standoff=12))
    fig_trend.update_yaxes(title=dict(text="Value (EUR Billion)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_trend.update_layout(legend=dict(orientation="h",y=-0.25,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=11)))

    # CAGR
    cagr_els = html.Div()
    if not mt_ct.empty and len(MT_CT_YEARS)>=2:
        y0,y1 = min(MT_CT_YEARS),max(MT_CT_YEARS); n=y1-y0
        e0=mt_ct[(mt_ct['refYear']==y0)&(mt_ct['codeType']==code_type)&(mt_ct['flowCode']=='X')]['primaryValue'].sum()
        e1=mt_ct[(mt_ct['refYear']==y1)&(mt_ct['codeType']==code_type)&(mt_ct['flowCode']=='X')]['primaryValue'].sum()
        i0=mt_ct[(mt_ct['refYear']==y0)&(mt_ct['codeType']==code_type)&(mt_ct['flowCode']=='M')]['primaryValue'].sum()
        i1=mt_ct[(mt_ct['refYear']==y1)&(mt_ct['codeType']==code_type)&(mt_ct['flowCode']=='M')]['primaryValue'].sum()
        exp_cagr=round(((e1/e0)**(1/n)-1)*100,1) if e0>0 else 0
        imp_cagr=round(((i1/i0)**(1/n)-1)*100,1) if i0>0 else 0
        # Fix #12: compute actual 2020-vs-2019 delta rather than hardcoded "COVID dip"
        if 2019 in MT_CT_YEARS and 2020 in MT_CT_YEARS:
            e_2019 = mt_ct[(mt_ct['refYear']==2019)&(mt_ct['codeType']==code_type)&(mt_ct['flowCode']=='X')]['primaryValue'].sum()
            e_2020 = mt_ct[(mt_ct['refYear']==2020)&(mt_ct['codeType']==code_type)&(mt_ct['flowCode']=='X')]['primaryValue'].sum()
            if e_2019 > 0:
                delta_2020 = (e_2020 - e_2019) / e_2019 * 100
                impact_label = f"{delta_2020:+.1f}% YoY"
                impact_color = th["red"] if delta_2020 < -5 else (th["amber"] if delta_2020 < 5 else th["green"])
            else:
                impact_label = "No data"; impact_color = th["muted"]
        else:
            impact_label = "No data"; impact_color = th["muted"]
        # Fix #11: signed format for CAGR (was hardcoded "+")
        exp_color = th["green"] if exp_cagr >= 0 else th["red"]
        imp_color = th["red"] if imp_cagr > 0 else th["green"]
        rows=[
            html.Div([html.Span(f"Export CAGR {y0}-{y1}",style={"fontSize":"10px","color":th["muted"],"flex":"1"}),
                      html.Span(f"{exp_cagr:+.1f}%",style={"fontSize":"11px","color":exp_color,"fontWeight":"600"})],
                     style={"display":"flex","padding":"5px 0","borderBottom":f"1px solid {th['border']}"}),
            html.Div([html.Span(f"Import CAGR {y0}-{y1}",style={"fontSize":"10px","color":th["muted"],"flex":"1"}),
                      html.Span(f"{imp_cagr:+.1f}%",style={"fontSize":"11px","color":imp_color,"fontWeight":"600"})],
                     style={"display":"flex","padding":"5px 0","borderBottom":f"1px solid {th['border']}"}),
            html.Div([html.Span(f"Peak exports",style={"fontSize":"10px","color":th["muted"],"flex":"1"}),
                      html.Span(f"{y1}: {fmt_val(e1)}",style={"fontSize":"10px","color":th["accent"]})],
                     style={"display":"flex","padding":"5px 0","borderBottom":f"1px solid {th['border']}"}),
            html.Div([html.Span("2020 vs 2019",style={"fontSize":"10px","color":th["muted"],"flex":"1"}),
                      html.Span(impact_label,style={"fontSize":"10px","color":impact_color})],
                     style={"display":"flex","padding":"5px 0"}),
        ]
        cagr_els = html.Div(rows)

    return fig, make_tbl(imp,"Top Import Sources"), make_tbl(exp,"Top Export Destinations"), summary, fig_trend, cagr_els

make_raw_toggle("m-d4-raw-toggle-btn","m-d4-raw-panel","m-d4-raw-content","m-d4-raw-source-tabs",[
    ("All Flows", mt_ct.head(500) if not mt_ct.empty else None,
     ["refYear","flowCode","partnerISO","cmdCode","hsLabel","primaryValue","continent"]),
])


# ── M-D5: Trade Map ───────────────────────────────────────────────────────────
def m_d5_layout():
    th = THEMES[MSEC]
    return html.Div([
        m_header("MedTech -- Trade Map",
                 "Geographic MedTech trade corridors -- arc width = trade value (parent codes)",
                 [yr_dropdown_generic("m-d5-year", MT_DD_YEARS, multi=True, default=2023),
                  html.Div([
                      html.Label("Flow", style={"fontSize":"11px","color":th["muted"],"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="m-d5-flow",
                          options=[{"label":"Both","value":"Both"},{"label":"Imports only","value":"M"},{"label":"Exports only","value":"X"}],
                          value="Both", clearable=False,
                          style={"fontSize":"13px","width":"150px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}"}),
                  ])]),
        html.Div([
            html.Div([m_card("MedTech Trade Map",[m_graph("m-d5-map",460)], info_id="MedTech Trade Map")],style={"flex":"1","minWidth":"0"}),
            html.Div(id="m-d5-country-panel",style={"width":"230px","flexShrink":"0",
                "background":th["card"],"border":f"1px solid {th['border']}",
                "borderRadius":"8px","padding":"14px","overflowY":"auto","maxHeight":"500px"}),
        ], style={"display":"flex","gap":"14px","marginBottom":"14px"}),
        html.Div(id="m-d5-kpi", style={"display":"flex","gap":"12px"}),
        raw_data_section("m-d5",[
            ("Parent Code Flows", mt_ct[mt_ct['codeType']=='parent'].head(500) if not mt_ct.empty else None,
             ["refYear","flowCode","partnerISO","cmdCode","hsLabel","primaryValue","partner_lat","partner_lon"]),
        ]),
    ])

@app.callback(
    Output("m-d5-map","figure"), Output("m-d5-kpi","children"), Output("m-d5-country-panel","children"),
    Input("url","pathname"), Input("m-d5-year","value"), Input("m-d5-flow","value"))
def cb_m_d5(pathname, year_val, flow):
    th = THEMES[MSEC]; year = resolve_year(year_val, MT_CT_YMAX); flow = flow or "Both"
    imp = mt_filter(year,'M','parent'); exp = mt_filter(year,'X','parent')
    IRELAND_LAT, IRELAND_LON = 53.35, -6.26
    fig = go.Figure()
    fig.update_layout(
        geo=dict(showframe=False,showcoastlines=True,showland=True,showocean=True,
                 landcolor="#1c1230",oceancolor="#140d20",coastlinecolor=th["border"],coastlinewidth=0.5,
                 projection_type="natural earth",bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=0,b=0))
    if mt_ct.empty:
        return fig,[m_kri("No Data","N/A","",th["muted"])],html.Div("No data",style={"color":th["muted"],"fontSize":"12px"})
    def add_arcs(df_flow, rgb, name):
        if df_flow.empty or "partner_lat" not in df_flow.columns: return
        g = df_flow.groupby(["partnerISO","partner_lat","partner_lon"])["primaryValue"].sum().reset_index()
        if g.empty: return
        max_v = g["primaryValue"].max() or 1
        for _,r in g.iterrows():
            if r["partner_lat"]==0 and r["partner_lon"]==0: continue
            raw_w = r["primaryValue"]/max_v; w=max(1.5,min(10,raw_w*10)); op=max(0.35,min(1.0,raw_w))
            rv,gv,bv=int(rgb[0]),int(rgb[1]),int(rgb[2])
            fig.add_trace(go.Scattergeo(lat=[r["partner_lat"],IRELAND_LAT],lon=[r["partner_lon"],IRELAND_LON],
                mode="lines",showlegend=False,
                line=dict(width=w,color=f"rgba({rv},{gv},{bv},{op:.2f})"),
                hovertemplate=f"<b>{r['partnerISO']}</b><br>{name}: {fmt_val(r['primaryValue'])}<extra></extra>"))
            fig.add_trace(go.Scattergeo(lat=[r["partner_lat"]],lon=[r["partner_lon"]],mode="markers",showlegend=False,
                marker=dict(size=max(5,w*1.2),color=f"rgba({rv},{gv},{bv},0.9)"),
                text=f"{r['partnerISO']}: {fmt_val(r['primaryValue'])}",hoverinfo="text"))
    if flow in ("Both","M"): add_arcs(imp,(179,157,219),"Imports")
    if flow in ("Both","X"): add_arcs(exp,(77,208,225),"Exports")
    fig.add_trace(go.Scattergeo(lat=[IRELAND_LAT],lon=[IRELAND_LON],mode="markers+text",
        marker=dict(size=14,color=th["accent"],line=dict(color=WHITE,width=2)),
        text=["Ireland"],textposition="top center",textfont=dict(color=th["text"],size=12),showlegend=False))
    if flow in ("Both","M"): fig.add_trace(go.Scattergeo(lat=[None],lon=[None],mode="lines",name="Imports",line=dict(color="rgba(179,157,219,1)",width=3),showlegend=True))
    if flow in ("Both","X"): fig.add_trace(go.Scattergeo(lat=[None],lon=[None],mode="lines",name="Exports",line=dict(color="rgba(77,208,225,1)",width=3),showlegend=True))
    fig.update_layout(legend=dict(bgcolor="rgba(20,13,32,0.9)",bordercolor=th["border"],borderwidth=1,font=dict(color=th["text"],size=12),x=0.01,y=0.99),font=dict(color=th["text"]))
    top_exp="N/A"; top_exp_val=""
    if not exp.empty:
        ge=exp.groupby('partnerISO')['primaryValue'].sum(); top_exp=ge.idxmax(); top_exp_val=fmt_val(ge.max())
    kpi = html.Div([
        m_kri("MedTech imports", fmt_val(imp["primaryValue"].sum()) if not imp.empty else "N/A","", th["c3"], f"Year {year}"),
        m_kri("MedTech exports", fmt_val(exp["primaryValue"].sum()) if not exp.empty else "N/A","", th["accent"], f"Year {year}"),
        m_kri("Top export market", top_exp,"", th["c2"], top_exp_val),
        m_kri("Import partners", str(imp['partnerISO'].nunique()) if not imp.empty else "0","", th["accent2"], "Countries"),
    ], style={"display":"flex","gap":"12px"})
    def country_rows(df, colour, label):
        if df.empty: return []
        g = df.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False)
        rows=[html.Div([html.Span(label,style={"fontSize":"9px","color":colour,"fontWeight":"600"}),
                        html.Span(f"({fmt_val(g.sum())} total)",style={"fontSize":"9px","color":th["muted"],"marginLeft":"6px"})],
                       style={"padding":"5px 0 3px","borderBottom":f"1px solid {th['border']}","marginBottom":"3px"})]
        for partner,val in g.head(10).items():
            bar_w = min(100,val/g.iloc[0]*100) if g.iloc[0]>0 else 0
            rows.append(html.Div([
                html.Div([html.Span("o",style={"color":colour,"fontSize":"8px","marginRight":"4px","fontWeight":"bold"}),
                          html.Span(partner,style={"fontSize":"10px","color":th["text"],"flex":"1","overflow":"hidden","textOverflow":"ellipsis","whiteSpace":"nowrap"}),
                          html.Span(fmt_val(val),style={"fontSize":"10px","color":colour,"fontWeight":"600","flexShrink":"0","marginLeft":"4px"})],
                         style={"display":"flex","alignItems":"center","marginBottom":"2px"}),
                html.Div(style={"background":BG_INPUT,"borderRadius":"2px","height":"3px","overflow":"hidden","marginBottom":"5px"},
                         children=[html.Div(style={"width":f"{bar_w:.0f}%","height":"100%","background":colour,"borderRadius":"2px"})]),
            ]))
        return rows
    imp_rows = country_rows(imp,th["c3"],"IMPORTS") if flow in ("Both","M") else []
    exp_rows = country_rows(exp,th["accent"],"EXPORTS") if flow in ("Both","X") else []
    panel = html.Div([
        html.Div([html.Span(f"Year {year}",style={"fontSize":"10px","color":th["text"],"fontWeight":"600"}),
                  html.Span(" -- Trade Partners",style={"fontSize":"9px","color":th["muted"]})],
                 style={"marginBottom":"8px","paddingBottom":"6px","borderBottom":f"1px solid {th['border']}"}),
        *imp_rows,*exp_rows,
    ])
    return fig, kpi, panel

make_raw_toggle("m-d5-raw-toggle-btn","m-d5-raw-panel","m-d5-raw-content","m-d5-raw-source-tabs",[
    ("Parent Code Flows", mt_ct[mt_ct['codeType']=='parent'].head(500) if not mt_ct.empty else None,
     ["refYear","flowCode","partnerISO","cmdCode","hsLabel","primaryValue","partner_lat","partner_lon"]),
])

# ── M-D6: Stress Test ─────────────────────────────────────────────────────────
def m_d6_layout():
    th = THEMES[MSEC]
    return html.Div([
        m_header("MedTech -- Stress Test",
                 "Supplier removal impact per sector (parent codes). Alternatives shown.",
                 [yr_dropdown_generic("m-d6-year", MT_DD_YEARS, multi=True, default=2023),
                  html.Div([
                      html.Label("Remove Country", style={"fontSize":"11px","color":th["muted"],"display":"block","marginBottom":"4px"}),
                      dcc.Dropdown(id="m-d6-country",
                          options=[{"label":p,"value":p} for p in MT_PARTNERS if p not in ["World","Areas, nes"]],
                          value=MT_PARTNERS[0] if MT_PARTNERS else "United States", clearable=False,
                          style={"fontSize":"13px","width":"200px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}"}),
                  ])]),
        html.Div("Scenario assumes no short-term substitution from alternative suppliers. Real-world impact is typically lower as reshoring and rerouting mitigate some losses over 3-12 months.",
                 style={"fontSize":"11px","color":th["muted"],"marginBottom":"8px","fontStyle":"italic"}),
        html.Div(id="m-d6-kri", style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(m_card("Supply lost by sector if country removed (%)",[m_graph("m-d6-bar",280)], flex="2", info_id="Supply lost by sector if country removed (%)"),
            m_card("Top alternative suppliers",[html.Div(id="m-d6-alts")], flex="1", info_id="Top alternative suppliers")),
        raw_data_section("m-d6",[
            ("Parent Code Imports", mt_ct[(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
             ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
        ]),
    ])

@app.callback(
    Output("m-d6-kri","children"), Output("m-d6-bar","figure"), Output("m-d6-alts","children"),
    Input("url","pathname"), Input("m-d6-year","value"), Input("m-d6-country","value"))
def cb_m_d6(pathname, year_val, country):
    th = THEMES[MSEC]; year = resolve_year(year_val, MT_CT_YMAX)
    if not country: country = "United States"
    imp = mt_filter(year,'M','parent')
    if imp.empty: return [],go.Figure(),html.P("No data")
    results = []
    for code in MT_PARENT_CODES:
        pdf = imp[imp['cmdCode'].astype(str)==str(code)]; tv = pdf['primaryValue'].sum()
        if tv==0: continue
        cv = pdf[pdf['partnerISO']==country]['primaryValue'].sum(); ls = cv/tv*100
        results.append({"sector":MT_PARENT_LABELS.get(code,code),"lost_share":round(ls,1),
                        "impact":stress_label(ls),"colour":stress_colour(ls),"total":tv,"lost_val":cv})
    results = sorted(results,key=lambda x:x["lost_share"],reverse=True)
    crit=sum(1 for r in results if r["impact"]=="CRITICAL")
    sev =sum(1 for r in results if r["impact"]=="SEVERE")
    var =sum(r["lost_val"] for r in results)
    signif = sum(1 for r in results if r["impact"]=="SIGNIFICANT")
    minimal = sum(1 for r in results if r["impact"]=="MINIMAL")
    _affected = sum(1 for r in results if r["lost_share"] >= 0.1)
    cards = html.Div([
        m_kri("Sectors affected",
              str(_affected),
              "",
              th["green"] if _affected == 0 else th["accent2"],
              "Sectors with >= 0.1% supply from this country"),
        m_kri("Critical (>70%)",   str(crit),"",         th["red"],     "Sector loses >70% supply"),
        m_kri("Severe (40-70%)",   str(sev),"",           th["red"] if sev>0 else th["muted"], "Sector loses 40-70% supply"),
        m_kri("Significant (20-40%)", str(signif),"",    th["amber"] if signif>0 else th["muted"], "Sector loses 20-40% supply"),
        m_kri("Value at risk",     fmt_val(var),"",
              th["red"] if var>1e9 else th["amber"] if var>1e8 else th["green"] if var>0 else th["muted"],
              f"Annual imports from {country}" if var>0 else f"No imports from {country}"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap"})
    fig = go.Figure(go.Bar(y=[r["sector"] for r in results],x=[r["lost_share"] for r in results],
        orientation="h",marker_color=[r["colour"] for r in results],
        text=[f"< 0.1% -- negligible" if r['lost_share'] < 0.1 else f"{r['lost_share']}%  {r['impact']}" for r in results],textposition="inside",insidetextanchor="middle",
        textfont=dict(size=10,color="#ffffff")))
    fig.add_vline(x=70,line_dash="dash",line_color=th["red"],opacity=0.7,
        annotation_text="CRITICAL >70%",annotation_position="top",
        annotation_font=dict(color=th["red"],size=10))
    fig.add_vline(x=40,line_dash="dot",line_color=th["amber"],opacity=0.7,
        annotation_text="SEVERE >40%",annotation_position="top",
        annotation_font=dict(color=th["amber"],size=10))
    fig.add_vrect(x0=0,x1=20, fillcolor=th["green"],opacity=0.04,line_width=0)
    fig.add_vrect(x0=20,x1=40,fillcolor=th["amber"],opacity=0.04,line_width=0)
    fig.add_vrect(x0=40,x1=70,fillcolor=C_ORANGE,opacity=0.05,line_width=0)
    fig.add_vrect(x0=70,x1=120,fillcolor=C_RED,opacity=0.05,line_width=0)
    themed_layout(fig,MSEC,False)
    fig.update_xaxes(title=dict(text="% of sector supply that would be lost if country removed",
                     font=dict(color=th["muted"],size=11),standoff=12),range=[0,120],ticksuffix="%")
    fig.update_yaxes(title=dict(text="MedTech Sector",font=dict(color=th["muted"],size=11),standoff=12))
    # Add interpretation note as annotation at bottom
    fig.add_annotation(text="Bars show: if this country stopped all exports to Ireland tomorrow, what % of that sector's total imports would be lost?",
        xref="paper",yref="paper",x=0.5,y=-0.22,showarrow=False,
        font=dict(size=9,color=th["muted"]),xanchor="center")
    alts = imp[imp['partnerISO']!=country].groupby('partnerISO')['primaryValue'].sum(); tot_imp = imp['primaryValue'].sum() or 1
    alt_els=[html.Div([
        html.Div([html.Span(p,style={"color":th["text"],"fontSize":"11px","flex":"1"}),
                  html.Span(f"{v/tot_imp*100:.1f}%",style={"color":th["accent"],"fontSize":"11px","fontWeight":"600"})],
                 style={"display":"flex","justifyContent":"space-between","marginBottom":"3px"}),
        html.Div(style={"background":BG_INPUT,"borderRadius":"3px","height":"7px","overflow":"hidden","marginBottom":"7px"},
                 children=[html.Div(style={"width":f"{min(v/tot_imp*100/40*100,100):.0f}%","height":"100%","background":th["accent"],"borderRadius":"3px"})]),
    ]) for p,v in alts.sort_values(ascending=False).head(6).items()]

    # Build plain-English interpretation
    lines_interp = []
    for r in results:
        ls = r["lost_share"]
        sec = r["sector"]
        if ls < 0.1:
            lines_interp.append(f"{sec}: {country} supplies negligible amounts (< 0.1%) -- no meaningful impact")
        elif r["impact"] == "CRITICAL":
            lines_interp.append(f"CRITICAL -- {sec}: {country} supplies {ls:.1f}% -- losing this partner would be catastrophic")
        elif r["impact"] == "SEVERE":
            lines_interp.append(f"SEVERE -- {sec}: {country} supplies {ls:.1f}% -- losing this partner requires urgent alternative sourcing")
        elif r["impact"] == "SIGNIFICANT":
            lines_interp.append(f"SIGNIFICANT -- {sec}: {country} supplies {ls:.1f}% -- meaningful disruption, alternatives available ({', '.join(list(alts.sort_values(ascending=False).head(2).index))})")
        else:
            lines_interp.append(f"LOW RISK -- {sec}: {country} supplies only {ls:.1f}% -- easily replaceable")
    overall = "LOW RISK" if crit==0 and sev==0 and signif==0 else "CRITICAL" if crit>0 else "SEVERE" if sev>0 else "MANAGEABLE"
    interp_colour = th["red"] if crit>0 else th["amber"] if sev>0 else th["c2"] if signif>0 else th["green"]
    interp_el = html.Div([
        html.Div([
            html.Span("OVERALL ASSESSMENT",style={"fontSize":"8px","color":th["muted"],"letterSpacing":"0.7px"}),
            html.Span(f"  {overall}",style={"fontSize":"10px","color":interp_colour,"fontWeight":"700"}),
        ],style={"marginBottom":"6px","paddingBottom":"6px","borderBottom":f"1px solid {th['border']}"}),
        *[html.Div(line, style={"fontSize":"10px","color":th["muted"],"marginBottom":"4px","lineHeight":"1.4"}) for line in lines_interp],
        html.Div(style={"marginTop":"8px","paddingTop":"6px","borderTop":f"1px solid {th['border']}"},
                 children=[html.Span("What does this mean? ",style={"fontSize":"9px","color":th["text"],"fontWeight":"500"}),
                            html.Span(f"If {country} stopped all MedTech exports to Ireland tomorrow, no single sector would lose more than {max((r['lost_share'] for r in results),default=0):.0f}% of its supply. The top alternative source is {list(alts.sort_values(ascending=False).head(1).index)[0] if not alts.empty else 'N/A'}.",
                                      style={"fontSize":"9px","color":th["muted"]})]),
    ],style={"padding":"10px","background":th["card"],"border":f"1px solid {th['border']}","borderRadius":"6px","marginTop":"8px"})

    return cards, fig, html.Div([html.Div(alt_els), interp_el])

make_raw_toggle("m-d6-raw-toggle-btn","m-d6-raw-panel","m-d6-raw-content","m-d6-raw-source-tabs",[
    ("Parent Code Imports", mt_ct[(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
     ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
])


# ── M-D7: Scenario War Room ───────────────────────────────────────────────────
def m_d7_layout():
    th = THEMES[MSEC]
    def slider_row(label, sid, min_v, max_v, step, val, suffix=""):
        return html.Div([
            html.Div([html.Span(label,style={"fontSize":"12px","color":th["muted"]}),
                      html.Span(id=f"{sid}-out",style={"fontSize":"13px","color":th["accent"],"fontWeight":"600","marginLeft":"8px"})],
                     style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"}),
            dcc.Slider(id=sid,min=min_v,max=max_v,step=step,value=val,
                       marks={min_v:{"label":f"{min_v}{suffix}","style":{"color":th["muted"],"fontSize":"10px"}},
                              max_v:{"label":f"{max_v}{suffix}","style":{"color":th["muted"],"fontSize":"10px"}}},
                       tooltip={"placement":"bottom","always_visible":False}),
        ], style={"marginBottom":"16px"})
    return html.Div([
        m_header("MedTech -- Scenario War Room",
                 "Monte Carlo supply disruption simulation -- MedTech-calibrated value at risk",
                 []),
        row(
            html.Div([
                m_card("Scenario Parameters",[
                    html.Div([
                        html.Label("Disruption Type",style={"fontSize":"10px","color":th["muted"],"display":"block","marginBottom":"5px"}),
                        dcc.Dropdown(id="m-d7-distype",
                            options=[{"label":"Supply Chain Disruption","value":"supply"},
                                     {"label":"Tariff Shock","value":"tariff"},
                                     {"label":"Regulatory Block","value":"reg"},
                                     {"label":"All Combined","value":"all"}],
                            value="supply",clearable=False,
                            style={"fontSize":"12px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}","marginBottom":"10px"}),
                        html.Label("Product Scope",style={"fontSize":"10px","color":th["muted"],"display":"block","marginBottom":"5px"}),
                        dcc.Dropdown(id="m-d7-scope",
                            options=[{"label":"All MedTech Sectors","value":"all"},
                                     {"label":"Medical Instruments only","value":"9018"},
                                     {"label":"Implants and Stents only","value":"9021"},
                                     {"label":"X-Ray Apparatus only","value":"9022"}],
                            value="all",clearable=False,
                            style={"fontSize":"12px","backgroundColor":"#ffffff","color":"#111111","border":f"1px solid {th['border']}","marginBottom":"10px"}),
                    ]),
                    slider_row("Disruption Severity (%)","m-d7-sev",0,100,1,75,"%"),
                    slider_row("Demand Surge Factor","m-d7-dem",1.0,3.0,0.1,1.4,"x"),
                    slider_row("Monte Carlo Iterations","m-d7-iters",100,2000,100,1000,""),
                ], info_id="Scenario Parameters"),
                html.Div(id="m-d7-kpi",style={"display":"flex","gap":"12px","marginTop":"12px","flexWrap":"wrap"}),
            ], style={"flex":"1"}),
            m_card("Potential Loss Distribution",[m_graph("m-d7-hist",300)], flex="2", info_id="VaR Distribution"),
        ),
        row(m_card("12-Month Impact Timeline",[m_graph("m-d7-tl",220)], flex="1", info_id="12-Month Impact Timeline"),
            m_card("Impact by MedTech Sector",[m_graph("m-d7-prod",220)], flex="1", info_id="Impact by MedTech Sector")),
        m_card("Percentile Summary Table",[html.Div(id="m-d7-ptbl")], info_id="Percentile Summary Table"),
        html.Div("Scenario assumes no short-term substitution from alternative suppliers. Real-world impact is typically lower as reshoring and rerouting mitigate some losses over 3-12 months.",
                 style={"fontSize":"11px","color":th["muted"],"marginTop":"12px","fontStyle":"italic","textAlign":"center"}),
        raw_data_section("m-d7",[
            ("MedTech Imports (base for Monte Carlo)",
             mt_ct[(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
             ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
        ]),
    ])

@app.callback(Output("m-d7-sev-out","children"),Input("m-d7-sev","value"))
def upd_m_sev(v): return f"{v or 0}%"
@app.callback(Output("m-d7-dem-out","children"),Input("m-d7-dem","value"))
def upd_m_dem(v): return f"{v or 1:.1f}x"
@app.callback(Output("m-d7-iters-out","children"),Input("m-d7-iters","value"))
def upd_m_iters(v): return f"{v or 1000:,}"

@app.callback(
    Output("m-d7-kpi","children"), Output("m-d7-hist","figure"),
    Output("m-d7-tl","figure"), Output("m-d7-prod","figure"), Output("m-d7-ptbl","children"),
    Input("m-d7-sev","value"), Input("m-d7-dem","value"), Input("m-d7-iters","value"),
    Input("m-d7-distype","value"), Input("m-d7-scope","value"))
def cb_m_d7(severity, demand, iters, distype, scope):
    th = THEMES[MSEC]
    sev = (severity or 75)/100; dem = demand or 1.4; n = iters or 1000

    # Base import value — scoped to selected sector(s)
    if not mt_ct.empty:
        yr = MT_CT_YMAX
        if scope == "all":
            base = mt_ct[(mt_ct['refYear']==yr)&(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='M')]['primaryValue'].sum()
        else:
            base = mt_ct[(mt_ct['refYear']==yr)&(mt_ct['cmdCode']==scope)&(mt_ct['flowCode']=='M')]['primaryValue'].sum()
        if base == 0: base = 3.72e9
    else:
        base = 3.72e9

    # Disruption type multiplier
    type_mult = {"supply":1.0,"tariff":0.6,"reg":0.8,"all":1.3}.get(distype,1.0)

    np.random.seed(42)
    samples = [base*sev*type_mult*np.random.uniform(0.65,1.35) +
               base*(dem-1)*np.random.uniform(0.75,1.25) for _ in range(n)]
    arr = np.array(samples)/1e9
    p5,p25,p50,p75,p95 = np.percentile(arr,[5,25,50,75,95])
    imp_pct = round(p50/(base/1e9)*100,1)

    kpis = html.Div([
        m_kri("Median VaR",f"EUR {p50:.2f}","bn",th["red"] if p50>2 else th["amber"],"P50 most likely outcome"),
        m_kri("Import impact",f"{imp_pct}","%",th["red"] if imp_pct>50 else th["amber"],"% of annual imports"),
        m_kri("P95 Worst Case",f"EUR {p95:.2f}","bn",th["red"],"Only 5% worse than this"),
        m_kri("P5 Best Case",f"EUR {p5:.2f}","bn",th["green"],"Only 5% better than this"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap"})

    fig_h = go.Figure(go.Histogram(x=arr,nbinsx=40,marker_color=th["c2"],opacity=0.75,
                                    name=f"Simulated outcomes (n={n:,})"))
    for val,lbl,col in [(p5,"P5",th["green"]),(p50,"P50",th["amber"]),(p95,"P95",th["red"])]:
        fig_h.add_vline(x=val,line_dash="dash",line_color=col,
            annotation_text=f"{lbl}: EUR {val:.2f}bn",annotation_position="top",
            annotation_font=dict(color=col,size=10))
    themed_layout(fig_h,MSEC,True)
    fig_h.update_xaxes(title=dict(text=f"Potential Loss (EUR Billion) | n={n:,} iterations",font=dict(color=th["muted"],size=11),standoff=12))
    fig_h.update_yaxes(title=dict(text="Number of Simulations",font=dict(color=th["muted"],size=11),standoff=12))

    months = list(range(1,13)); np.random.seed(42)
    mi = [p50*(1-np.exp(-0.3*m))*np.random.uniform(0.9,1.1) for m in months]
    fig_tl = go.Figure(go.Scatter(x=months,y=mi,mode="lines+markers",fill="tozeroy",
        fillcolor=hex_rgba(th["red"],0.10),line=dict(color=th["red"],width=2.5),
        marker=dict(size=6,color=th["red"]),name="Cumulative impact"))
    themed_layout(fig_tl,MSEC,True)
    fig_tl.update_xaxes(title=dict(text="Month After Disruption",font=dict(color=th["muted"],size=11),standoff=12),
                         tickvals=months,ticktext=[f"M{m}" for m in months])
    fig_tl.update_yaxes(title=dict(text="Cumulative Potential Loss (EUR bn)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_tl.update_layout(legend=dict(orientation="h",y=-0.30,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=11)))

    # Impact by MedTech sector
    if not mt_ct.empty:
        sector_base = {code: mt_ct[(mt_ct['refYear']==MT_CT_YMAX)&(mt_ct['cmdCode']==code)&(mt_ct['flowCode']=='M')]['primaryValue'].sum()
                       for code in MT_PARENT_CODES}
    else:
        sector_base = {'9018':1.95e9,'9021':1.69e9,'9022':0.08e9}
    sector_labels = [MT_PARENT_LABELS[c] for c in MT_PARENT_CODES]
    sector_impacts = [sector_base.get(c,0)*sev*type_mult/1e9 for c in MT_PARENT_CODES]
    sector_impacts_sorted = sorted(zip(sector_labels,sector_impacts),key=lambda x:x[1])
    sl,sv = zip(*sector_impacts_sorted) if sector_impacts_sorted else ([],[])
    fig_p = go.Figure(go.Bar(y=list(sl),x=list(sv),orientation="h",
        marker_color=[stress_colour(v/p50*100) for v in sv],
        text=[f"EUR {v:.2f}bn" for v in sv],textposition="inside",insidetextanchor="middle",
        textfont=dict(size=10,color="#ffffff"),name="Estimated impact"))
    themed_layout(fig_p,MSEC,False)
    fig_p.update_xaxes(title=dict(text="Estimated Impact (EUR Billion)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_p.update_yaxes(title=dict(text="MedTech Sector",font=dict(color=th["muted"],size=11),standoff=12))

    ptbl = dash_table.DataTable(
        data=[{"Percentile":l,"Potential Loss":f"EUR {v:.3f}bn","Interpretation":i} for l,v,i in
              [("P5 (Best Case)",p5,"Only 5% of scenarios are lower"),
               ("P25 (Lower Quartile)",p25,"25% of scenarios are lower"),
               ("P50 (Median)",p50,"Most likely outcome -- central estimate"),
               ("P75 (Upper Quartile)",p75,"75% of scenarios are lower"),
               ("P95 (Worst Case)",p95,"Only 5% of scenarios are worse")]],
        columns=[{"name":c,"id":c} for c in ["Percentile","Potential Loss","Interpretation"]],
        style_header={"backgroundColor":"#221638","color":th["text"],"fontWeight":"600","fontSize":"12px","border":f"1px solid {th['border']}"},
        style_cell={"padding":"10px 14px","fontSize":"12px","fontFamily":"Inter,Segoe UI,Arial",
                    "backgroundColor":th["card"],"color":th["text"],"border":f"1px solid {th['border']}"},
        style_data_conditional=[
            {"if":{"row_index":4},"backgroundColor":"#2d1a2e","color":th["red"]},
            {"if":{"row_index":2},"backgroundColor":"#2a2220","color":th["amber"]},
            {"if":{"row_index":0},"backgroundColor":"#1a2d20","color":th["green"]},
        ])
    return kpis, fig_h, fig_tl, fig_p, ptbl

# ── M-D8: Product Analysis ────────────────────────────────────────────────────
def m_d8_layout():
    th = THEMES[MSEC]
    return html.Div([
        m_header("MedTech -- Product Analysis",
                 "Sub-codes ONLY: 901839 Catheters . 901890 Other Instruments . 902190 Cardiovascular Implants. Never mixed with parent codes.",
                 [yr_dropdown_generic("m-d8-year", MT_DD_YEARS, multi=True, default=2023)]),
        html.Div([
            html.Div("Sub-codes only on this dashboard -- no parent codes. HHI calculated per product separately.",
                     style={"fontSize":"10px","color":th["amber"],"background":"rgba(251,191,36,0.08)",
                            "border":f"1px solid rgba(251,191,36,0.25)","borderRadius":"6px",
                            "padding":"8px 12px","marginBottom":"12px"}),
        ]),
        html.Div(id="m-d8-kri", style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}),
        row(m_card("Export value by product (sub-codes)",[m_graph("m-d8-exp-bar",260)], flex="1", info_id="Export value by product (sub-codes)"),
            m_card("Import value by product (sub-codes)",[m_graph("m-d8-imp-bar",260)], flex="1", info_id="Import value by product (sub-codes)")),
        row(m_card("HHI per product (sub-codes, imports)",[m_graph("m-d8-hhi",240)], flex="1", info_id="HHI per product (sub-codes, imports)"),
            m_card("Top import source per product",[html.Div(id="m-d8-sources")], flex="1", info_id="Top import source per product")),
        raw_data_section("m-d8",[
            ("Sub-Code Exports", mt_ct[(mt_ct['codeType']=='sub')&(mt_ct['flowCode']=='X')].head(500) if not mt_ct.empty else None,
             ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
            ("Sub-Code Imports", mt_ct[(mt_ct['codeType']=='sub')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
             ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
        ]),
    ])

@app.callback(
    Output("m-d8-kri","children"), Output("m-d8-exp-bar","figure"),
    Output("m-d8-imp-bar","figure"), Output("m-d8-hhi","figure"), Output("m-d8-sources","children"),
    Input("url","pathname"), Input("m-d8-year","value"))
def cb_m_d8(pathname, year_val):
    th = THEMES[MSEC]; year = resolve_year(year_val, MT_CT_YMAX)
    exp_s = mt_filter(year,'X','sub'); imp_s = mt_filter(year,'M','sub')
    PAL = [th["c1"],th["c2"],th["c3"]]

    cards = html.Div([
        m_kri("Catheters exports", fmt_val(exp_s[exp_s['cmdCode']=='901839']['primaryValue'].sum()),"", th["c1"], "HS 901839"),
        m_kri("Other Instruments exp", fmt_val(exp_s[exp_s['cmdCode']=='901890']['primaryValue'].sum()),"", th["c2"], "HS 901890"),
        m_kri("Cardiovascular exp", fmt_val(exp_s[exp_s['cmdCode']=='902190']['primaryValue'].sum()),"", th["c3"], "HS 902190"),
        m_kri("Total sub-code exports", fmt_val(exp_s['primaryValue'].sum()),"", th["accent"], f"Sub-codes {year}"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap"})

    def product_bar(df, title, palette):
        fig = go.Figure()
        data = [(MT_SUB_LABELS.get(c,c), df[df['cmdCode'].astype(str)==str(c)]['primaryValue'].sum()) for c in MT_SUB_CODES]
        data.sort(key=lambda x: x[1])
        if data:
            yl, xv = zip(*data)
            fig.add_trace(go.Bar(y=list(yl),x=[v/1e9 for v in xv],orientation="h",
                marker_color=palette[:len(yl)],
                text=[f"EUR {v/1e9:.2f}bn" for v in xv],textposition="outside",
                textfont=dict(size=10,color=th["text"])))
        themed_layout(fig,MSEC,False)
        fig.update_xaxes(title=dict(text="Value (EUR Billion)",font=dict(color=th["muted"],size=11),standoff=12))
        fig.update_yaxes(title=dict(text="Product",font=dict(color=th["muted"],size=11),standoff=12))
        return fig

    fig_exp = product_bar(exp_s,"Exports",PAL)
    fig_imp = product_bar(imp_s,"Imports",[th["c3"],th["c4"],th["c2"]])

    # HHI per sub-code
    fig_hhi = go.Figure()
    hhi_data = []
    for code in MT_SUB_CODES:
        sub = imp_s[imp_s['cmdCode']==code]
        tot = sub['primaryValue'].sum() or 1
        h = round(sum((v/tot)**2 for v in sub.groupby('partnerISO')['primaryValue'].sum().values),4) if not sub.empty else 0
        hhi_data.append((MT_SUB_LABELS.get(code,code), h))
    hhi_data.sort(key=lambda x:x[1])
    if hhi_data:
        yl,xv = zip(*hhi_data)
        fig_hhi.add_trace(go.Bar(y=list(yl),x=list(xv),orientation="h",
            marker_color=[hhi_colour(h) for h in xv],
            text=[f"{h:.4f}  {hhi_label(h)}" for h in xv],textposition="outside",
            textfont=dict(size=10,color=th["text"])))
    fig_hhi.add_vline(x=0.25,line_dash="dot",line_color=th["amber"],opacity=0.7,annotation_text="High 0.25",annotation_font=dict(color=th["amber"],size=10))
    themed_layout(fig_hhi,MSEC,False)
    fig_hhi.update_xaxes(title=dict(text="HHI Score (0=diversified, 1=single source)",font=dict(color=th["muted"],size=11),standoff=12))
    fig_hhi.update_yaxes(title=dict(text="Product",font=dict(color=th["muted"],size=11),standoff=12))

    # Top source per product
    source_els = []
    for code in MT_SUB_CODES:
        sub = imp_s[imp_s['cmdCode']==code]
        label = MT_SUB_LABELS.get(code,code)
        if sub.empty:
            source_els.append(html.Div(f"{label}: No data",style={"fontSize":"10px","color":th["muted"],"marginBottom":"8px"}))
            continue
        gi = sub.groupby('partnerISO')['primaryValue'].sum().sort_values(ascending=False)
        top3 = gi.head(3)
        source_els.append(html.Div([
            html.Div(label,style={"fontSize":"10px","color":th["text"],"fontWeight":"500","marginBottom":"3px"}),
            *[html.Div([
                html.Span(p,style={"fontSize":"9px","color":th["muted"],"flex":"1"}),
                html.Span(f"{v/gi.sum()*100:.0f}%",style={"fontSize":"9px","color":th["accent"],"fontWeight":"600"}),
            ],style={"display":"flex","justifyContent":"space-between","padding":"2px 0"})
              for p,v in top3.items()],
        ],style={"marginBottom":"10px","paddingBottom":"8px","borderBottom":f"1px solid {th['border']}"}))
    return cards, fig_exp, fig_imp, fig_hhi, html.Div(source_els)

make_raw_toggle("e-d7-raw-toggle-btn","e-d7-raw-panel","e-d7-raw-content","e-d7-raw-source-tabs",[
    ("Comtrade Imports (base for Monte Carlo)",
     comtrade[(comtrade["refyear"]==CT_YEAR_MAX)&(comtrade["flowcode"]=="M")].head(500) if not comtrade.empty else None,
     ["refyear","partnerdesc","cmddesc","primaryvalue"]),
    ("SEAI Imports",
     seai[seai["flow"]=="Imports"].head(200) if not seai.empty else None,
     ["year","flow","oil","natural_gas","electricity","total"]),
])
make_raw_toggle("m-d7-raw-toggle-btn","m-d7-raw-panel","m-d7-raw-content","m-d7-raw-source-tabs",[
    ("MedTech Imports (base for Monte Carlo)",
     mt_ct[(mt_ct['codeType']=='parent')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
     ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
])
make_raw_toggle("m-d8-raw-toggle-btn","m-d8-raw-panel","m-d8-raw-content","m-d8-raw-source-tabs",[
    ("Sub-Code Exports", mt_ct[(mt_ct['codeType']=='sub')&(mt_ct['flowCode']=='X')].head(500) if not mt_ct.empty else None,
     ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
    ("Sub-Code Imports", mt_ct[(mt_ct['codeType']=='sub')&(mt_ct['flowCode']=='M')].head(500) if not mt_ct.empty else None,
     ["refYear","partnerISO","cmdCode","hsLabel","primaryValue"]),
])


# ???????????????????????????????????????????????????????
# RUN
# ???????????????????????????????????????????????????????
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
