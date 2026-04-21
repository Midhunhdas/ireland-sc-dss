"""
chart_tooltips.py
─────────────────
Hover-tooltip definitions for every chart/card in the Ireland SC-DSS dashboard.

Each entry is keyed by the card title used in app.py and contains:
    title   : heading shown inside the tooltip (usually same as card title)
    body    : plain-language explanation of what the chart shows
    legend  : optional list of (threshold_label, colour, meaning) tuples
              colours: "green" | "amber" | "orange" | "red" | "blue" | "grey"
    source  : optional data-source footer line

If a card's title is not in CHART_TOOLTIPS, no tooltip icon is rendered.
Cards with info_id=None (or omitted) behave exactly as before.
"""

CHART_TOOLTIPS = {

    # ═══════════════════════════════════════════════════════════════════
    # ENERGY SECTOR
    # ═══════════════════════════════════════════════════════════════════

    # ── E-D1: Strategic Overview ────────────────────────────────────────
    "HHI by fuel type": {
        "title": "HHI by Fuel Type",
        "body": ("The Herfindahl-Hirschman Index (HHI) measures how concentrated "
                 "Ireland's imports are for each fuel. A score near 0 means imports "
                 "come from many countries (safe). A score near 1 means almost all "
                 "imports come from one country (risky)."),
        "legend": [
            ("HHI = 0.00–0.15", "green",  "Competitive"),
            ("HHI = 0.15–0.25", "amber",  "Moderate"),
            ("HHI = 0.25–0.40", "orange", "High"),
            ("HHI > 0.40",      "red",    "Critical"),
        ],
        "source": "UN Comtrade HS 27xx · Calculated as sum of (country share)²",
    },
    "Risk signal feed": {
        "title": "Risk Signal Feed",
        "body": ("Automated alerts generated from current indicators. Each signal "
                 "flags a specific risk condition: high supplier concentration, "
                 "falling self-sufficiency, dominant single-source imports, or "
                 "critical fuel dependencies."),
        "legend": [
            ("CRITICAL", "red",    "Immediate action required"),
            ("HIGH",     "orange", "Close monitoring advised"),
            ("WATCH",    "amber",  "Elevated but stable"),
            ("OK",       "green",  "Within safe range"),
        ],
        "source": "Derived from SEAI NEB and UN Comtrade",
    },
    "Oil import dependency trend (%)": {
        "title": "Oil Import Dependency Trend",
        "body": ("Percentage of Ireland's total energy imports made up of oil, "
                 "tracked year by year. Oil has historically been Ireland's largest "
                 "import. A declining trend indicates progress toward diversification."),
        "legend": [
            ("< 40%",   "green",  "Well diversified"),
            ("40–55%",  "amber",  "Moderate reliance"),
            ("55–65%",  "orange", "High dependency"),
            ("> 65%",   "red",    "Critical dependency"),
        ],
        "source": "SEAI National Energy Balance (1990–2024)",
    },
    "Self-sufficiency trend (%)": {
        "title": "Self-Sufficiency Trend",
        "body": ("Share of Ireland's energy demand met by domestic production "
                 "(peat, renewables, wind, biomass). Higher percentages reduce "
                 "exposure to global price shocks and supply disruptions."),
        "legend": [
            ("> 30%",   "green",  "Resilient"),
            ("20–30%",  "amber",  "Moderate"),
            ("10–20%",  "orange", "Low"),
            ("< 10%",   "red",    "Critical"),
        ],
        "source": "SEAI National Energy Balance",
    },
    "Critical fuel summary": {
        "title": "Critical Fuel Summary",
        "body": ("Per-fuel snapshot combining import volume, share of total imports, "
                 "HHI score, and risk classification. Use this table to identify "
                 "which specific fuels carry the highest concentration risk."),
        "legend": [
            ("Competitive", "green",  "HHI < 0.15"),
            ("Moderate",    "amber",  "0.15–0.25"),
            ("High",        "orange", "0.25–0.40"),
            ("Critical",    "red",    "HHI > 0.40"),
        ],
        "source": "UN Comtrade HS 27xx",
    },

    # ── E-D2: Import Dependency ─────────────────────────────────────────
    "HHI by Fuel Type": {
        "title": "HHI by Fuel Type",
        "body": ("Same as strategic overview but shown for the selected year(s). "
                 "The Herfindahl-Hirschman Index quantifies supplier concentration "
                 "for each fuel — higher bars mean Ireland depends on fewer "
                 "countries for that fuel."),
        "legend": [
            ("HHI = 0.00–0.15", "green",  "Competitive"),
            ("HHI = 0.15–0.25", "amber",  "Moderate"),
            ("HHI = 0.25–0.40", "orange", "High"),
            ("HHI > 0.40",      "red",    "Critical"),
        ],
        "source": "UN Comtrade HS 27xx",
    },
    "Country Dependency Score": {
        "title": "Country Dependency Score",
        "body": ("Composite risk score for each supplying country, blending three factors: "
                 "(1) share of total imports, (2) single-fuel dominance, "
                 "(3) governance risk from World Bank Worldwide Governance Indicators "
                 "(Political Stability & Absence of Violence percentile). "
                 "Higher scores mean Ireland is more exposed to disruption from that country."),
        "legend": [
            ("Weighting", "grey", "0.4 × share + 0.3 × dominance + 0.3 × WGI gov. risk"),
        ],
        "source": "UN Comtrade HS 27xx · World Bank WGI 2024 (data year 2023)",
    },
    "Continental Breakdown -- Import vs Export": {
        "title": "Continental Breakdown",
        "body": ("Distribution of Ireland's energy trade across continents, split by "
                 "import and export flows. Helps identify regional clusters and "
                 "geographic concentration beyond the country level."),
        "source": "UN Comtrade HS 27xx · Values in EUR",
    },
    "HHI Trend Over Time": {
        "title": "HHI Trend Over Time",
        "body": ("Historical evolution of the aggregate HHI across all imported "
                 "fuels. A rising line means Ireland's import sources are becoming "
                 "more concentrated; a falling line means diversification."),
        "source": "UN Comtrade HS 27xx · Year range configurable above",
    },

    # ── E-D3: Supply Flow ──────────────────────────────────────────────
    "Supply Flow Sankey Diagram": {
        "title": "Supply Flow Sankey",
        "body": ("Flow diagram tracing energy imports from source countries to "
                 "Ireland, and exports from Ireland to destination countries. "
                 "Ribbon thickness is proportional to trade value in EUR."),
        "source": "UN Comtrade HS 27xx · Filtered by selected year(s)",
    },
    "Top Import Sources": {
        "title": "Top Import Sources",
        "body": ("Ranking of the largest countries supplying Ireland's imports "
                 "by total value. A single dominant country indicates concentration "
                 "risk; broader diversification reduces vulnerability."),
        "source": "UN Comtrade HS 27xx",
    },
    "Top Export Markets": {
        "title": "Top Export Markets",
        "body": ("Ranking of the largest destinations for Ireland's exports by "
                 "total value. Concentration in one market increases exposure to "
                 "that market's tariffs, demand shocks, or regulatory changes."),
        "source": "UN Comtrade HS 27xx",
    },
    "Top Export Destinations": {
        "title": "Top Export Destinations",
        "body": ("Top partner countries receiving Ireland's exports, ranked by "
                 "trade value. Use alongside Flow Summary to assess export-side "
                 "concentration risk."),
        "source": "UN Comtrade HS 27xx",
    },
    "Flow Summary": {
        "title": "Flow Summary",
        "body": ("High-level aggregate of the year's trade flows: total imports, "
                 "total exports, trade balance, and number of unique trading "
                 "partners. Use as a quick sanity check alongside the Sankey."),
        "source": "UN Comtrade HS 27xx",
    },

    # ── E-D4: Trade Map ────────────────────────────────────────────────
    "Bilateral Energy Trade Map": {
        "title": "Bilateral Energy Trade Map",
        "body": ("Geographic visualisation of Ireland's energy trade. Arcs connect "
                 "Ireland to each trading partner; arc thickness indicates trade "
                 "volume. Red = imports into Ireland, green = exports out."),
        "legend": [
            ("Red arcs",   "red",   "Imports into Ireland"),
            ("Green arcs", "green", "Exports from Ireland"),
        ],
        "source": "UN Comtrade HS 27xx · Great-circle projection",
    },

    # ── E-D6: Stress Testing ───────────────────────────────────────────
    "Supply Lost by Product (%)": {
        "title": "Supply Lost by Product",
        "body": ("If imports from the selected country were cut off, this shows "
                 "the percentage of each fuel's supply that would be lost. "
                 "Higher bars indicate greater vulnerability to that supplier."),
        "legend": [
            ("< 20%",  "green",  "Manageable"),
            ("20–40%", "amber",  "Significant"),
            ("40–70%", "orange", "Severe"),
            ("> 70%",  "red",    "Critical"),
        ],
        "source": "UN Comtrade HS 27xx · Counterfactual removal simulation",
    },
    "Top Alternative Suppliers": {
        "title": "Top Alternative Suppliers",
        "body": ("Countries that could potentially replace the disrupted supplier, "
                 "ranked by their current export capacity in the same fuels. "
                 "A short list means limited substitution options."),
        "source": "UN Comtrade HS 27xx · Global export data",
    },
    "Detailed Impact Table": {
        "title": "Detailed Impact Table",
        "body": ("Row-level breakdown of the stress-test scenario: each fuel's "
                 "pre-shock share, post-shock share, volume lost, and classified "
                 "severity. Gives the numeric backing to the summary bar chart."),
        "source": "UN Comtrade HS 27xx",
    },

    # ── E-D7: Scenario War Room ────────────────────────────────────────
    "Scenario Parameters": {
        "title": "Scenario Parameters",
        "body": ("Configure the Monte-Carlo simulation: shock type, severity "
                 "distribution, affected countries, and number of simulated runs. "
                 "Changes here drive all other charts in this tab."),
        "source": "User-defined inputs",
    },
    "VaR Distribution": {
        "title": "VaR Distribution",
        "body": ("Distribution of simulated losses across all Monte-Carlo runs. "
                 "The 95th percentile (red line) is the Value-at-Risk — there is "
                 "a 5% probability the loss exceeds this amount."),
        "legend": [
            ("< €100M",   "green",  "Low VaR"),
            ("€100M–1B",  "amber",  "Moderate"),
            ("> €1B",     "red",    "High VaR"),
        ],
        "source": "Monte-Carlo simulation · 1,000+ runs",
    },
    "12-Month Impact Timeline": {
        "title": "12-Month Impact Timeline",
        "body": ("Projected month-by-month losses after shock onset, assuming "
                 "gradual supply-chain recovery. Peak impact typically lands 3–6 "
                 "months in, tapering as alternatives come online."),
        "source": "Expert-judgement recovery curves · Not peer-reviewed",
    },
    "Impact by Product": {
        "title": "Impact by Product",
        "body": ("Which fuels bear the brunt of the shock. Products with high "
                 "pre-shock import concentration from the affected countries take "
                 "the largest hits."),
        "source": "UN Comtrade HS 27xx",
    },
    "Percentile Summary Table": {
        "title": "Percentile Summary Table",
        "body": ("Key percentiles of the simulated loss distribution (p50, p75, "
                 "p90, p95, p99). Use p95 as the standard VaR figure and p99 as "
                 "the tail-risk worst-case estimate."),
        "source": "Monte-Carlo simulation output",
    },

    # ── E-D10: ML Analysis ─────────────────────────────────────────────
    "Anomaly Detection -- Trade Flow Outliers": {
        "title": "Anomaly Detection",
        "body": ("Isolation Forest algorithm flags unusual trade flows — volume "
                 "spikes, unexpected partners, or sharp share changes. Red points "
                 "are potential outliers worth investigating."),
        "source": "sklearn IsolationForest · contamination=0.05",
    },
    "K-Means Clustering -- Fuel Risk Groups": {
        "title": "K-Means Fuel Clustering",
        "body": ("Unsupervised grouping of fuels based on HHI, top-share, and "
                 "volatility. Clusters with similar risk profiles are shown in the "
                 "same colour — helps spot fuels that behave alike."),
        "source": "sklearn KMeans · k=3",
    },
    "HHI Trend & Forecast to 2030": {
        "title": "HHI Forecast to 2030",
        "body": ("Historical HHI trend plus a linear-regression projection to 2030. "
                 "Shaded band shows the forecast confidence interval. Treat as a "
                 "directional indicator, not a precise prediction."),
        "source": "Linear regression · 95% CI band",
    },
    "Cascade Failure -- Supplier Removal Simulation": {
        "title": "Cascade Failure Simulation",
        "body": ("Sequentially removes the top N suppliers and shows cumulative "
                 "supply loss. Steep curves indicate fragile supply chains where "
                 "losing just a few partners causes major disruption."),
        "source": "UN Comtrade HS 27xx · Sequential removal model",
    },

    # ── E-D14: Export Dependency ───────────────────────────────────────
    "Export Market Concentration": {
        "title": "Export Market Concentration",
        "body": ("Share of Ireland's total energy exports going to each destination "
                 "country. High concentration in one market (especially UK) raises "
                 "exposure to tariffs or trade-policy shifts."),
        "legend": [
            ("Top 1 < 25%",     "green",  "Diversified"),
            ("Top 1 = 25–40%",  "amber",  "Moderate"),
            ("Top 1 = 40–60%",  "orange", "Concentrated"),
            ("Top 1 > 60%",     "red",    "Highly concentrated"),
        ],
        "source": "UN Comtrade HS 27xx",
    },
    "Tariff Scenario Impact": {
        "title": "Tariff Scenario Impact",
        "body": ("Estimated revenue at risk under hypothetical tariff scenarios "
                 "(10%, 25%, 50% rates) applied by major trading partners. "
                 "Pass-through multiplier applied per expert judgement."),
        "legend": [
            ("< €10M",    "green",  "Minor"),
            ("€10–100M",  "amber",  "Moderate"),
            ("> €100M",   "red",    "Major"),
        ],
        "source": "Expert-judgement multipliers · Not peer-reviewed",
    },
    "Export Vulnerability Matrix": {
        "title": "Export Vulnerability Matrix",
        "body": ("Cross-tabulates export products against destination markets to "
                 "reveal concentrated cells — products heavily reliant on a single "
                 "country. Darker cells indicate greater vulnerability."),
        "source": "UN Comtrade HS 27xx",
    },

    # ═══════════════════════════════════════════════════════════════════
    # AGRICULTURE SECTOR
    # ═══════════════════════════════════════════════════════════════════

    # ── A-D1: Strategic Overview ───────────────────────────────────────
    "Export value by commodity": {
        "title": "Export Value by Commodity",
        "body": ("Total export value in euros, grouped by HS chapter (broad "
                 "commodity category). Dairy and meat typically dominate Irish "
                 "agricultural exports, reflecting Ireland's grass-based farming."),
        "source": "UN Comtrade HS 01–22",
    },
    "Key risk signals": {
        "title": "Key Risk Signals",
        "body": ("Automated alerts for agricultural supply chain risks: import "
                 "dependency in staple foods, export concentration in key markets, "
                 "and self-sufficiency gaps."),
        "legend": [
            ("RISK",  "red",    "Action required"),
            ("HIGH",  "orange", "Close monitoring"),
            ("WATCH", "amber",  "Elevated"),
            ("OK",    "green",  "Safe"),
        ],
        "source": "CSO · UN Comtrade · FAOSTAT",
    },
    "Export market share": {
        "title": "Export Market Share",
        "body": ("Share of Irish agri-food exports going to each destination "
                 "country. UK remains dominant despite Brexit; EU markets "
                 "(Netherlands, Germany, France) form the next tier."),
        "source": "UN Comtrade HS 01–22",
    },
    "Food self-sufficiency trend": {
        "title": "Food Self-Sufficiency Trend",
        "body": ("Ratio of domestic food production to total food consumption. "
                 "Values above 100% mean Ireland is a net exporter for that "
                 "commodity; below 100% means net importer."),
        "legend": [
            ("> 100%",  "green",  "Net exporter"),
            ("80–100%", "amber",  "Near self-sufficient"),
            ("< 80%",   "red",    "Import dependent"),
        ],
        "source": "FAOSTAT Food Balance Sheets",
    },

    # ── A-D2: Import Dependency ────────────────────────────────────────
    "Import HHI by commodity group": {
        "title": "Import HHI by Commodity Group",
        "body": ("Herfindahl-Hirschman Index for each HS chapter on the import "
                 "side. Shows which commodity categories Ireland sources from a "
                 "narrow set of countries versus a broad base."),
        "legend": [
            ("HHI = 0.00–0.15", "green",  "Competitive"),
            ("HHI = 0.15–0.25", "amber",  "Moderate"),
            ("HHI = 0.25–0.40", "orange", "High"),
            ("HHI > 0.40",      "red",    "Critical"),
        ],
        "source": "UN Comtrade HS 01–22",
    },
    "Top import sources": {
        "title": "Top Import Sources",
        "body": ("Largest supplying countries for Irish agri-food imports, ranked "
                 "by total value. UK and EU countries dominate; changes over time "
                 "reflect Brexit, supply chain shifts, and new trade agreements."),
        "source": "UN Comtrade HS 01–22",
    },
    "Continental import breakdown": {
        "title": "Continental Import Breakdown",
        "body": ("Agricultural imports grouped by continent of origin. Helps "
                 "identify regional dependencies and spot diversification "
                 "opportunities beyond Europe."),
        "source": "UN Comtrade HS 01–22",
    },
    "Import trend 2010-2024": {
        "title": "Import Trend 2010–2024",
        "body": ("Year-by-year total agri-food import value. Upward trends may "
                 "indicate growing domestic demand, declining self-sufficiency, or "
                 "currency effects inflating euro values."),
        "source": "UN Comtrade HS 01–22",
    },
    "Import by commodity × source matrix (top 10 products)": {
        "title": "Import Product × Source Matrix",
        "body": ("Heatmap of the top 10 imported products against top 6 source "
                 "countries. Darker cells = higher import value. Reveals which "
                 "specific product-country pairs carry the most risk."),
        "source": "UN Comtrade HS 01–22",
    },

    # ── A-D3: Export Dependency ────────────────────────────────────────
    "Top export markets": {
        "title": "Top Export Markets",
        "body": ("Largest destination countries for Irish agri-food exports. "
                 "High UK share reflects geographic proximity and historical "
                 "trading relationships; increasing EU-27 share is a post-Brexit "
                 "trend."),
        "source": "UN Comtrade HS 01–22",
    },
    "Tariff impact by market": {
        "title": "Tariff Impact by Market",
        "body": ("Estimated revenue impact if major markets imposed tariffs. "
                 "Dairy and beef are most exposed given their export volumes. "
                 "Multipliers applied per expert judgement."),
        "legend": [
            ("< €50M",     "green",  "Minor"),
            ("€50–500M",   "amber",  "Moderate"),
            ("> €500M",    "red",    "Major"),
        ],
        "source": "Expert-judgement multipliers · Not peer-reviewed",
    },
    "Export by commodity ? market matrix": {
        "title": "Export Commodity × Market Matrix",
        "body": ("Cross-tabulation of top 8 export commodities against top "
                 "destination markets. Concentrated cells reveal product-market "
                 "pairs most vulnerable to specific market disruptions."),
        "source": "UN Comtrade HS 01–22",
    },

    # ── A-D4: Supply Flow ──────────────────────────────────────────────
    "Agriculture Supply Flow Sankey": {
        "title": "Agricultural Supply Flow",
        "body": ("Sankey diagram tracing agri-food flows: import origins → "
                 "Ireland → export destinations. Shows how Ireland acts as both "
                 "importer of inputs and exporter of processed products."),
        "source": "UN Comtrade HS 01–22",
    },

    # ── A-D5: Trade Map ────────────────────────────────────────────────
    "Agri Trade Map": {
        "title": "Agricultural Trade Map",
        "body": ("Geographic visualisation of Ireland's agri-food trade flows. "
                 "Orange arcs = imports into Ireland, green arcs = exports. "
                 "Arc thickness shows trade value."),
        "legend": [
            ("Orange arcs", "orange", "Imports into Ireland"),
            ("Green arcs",  "green",  "Exports from Ireland"),
        ],
        "source": "UN Comtrade HS 01–22",
    },

    # ── A-D6: Stress Test ──────────────────────────────────────────────
    "Supply lost by commodity (%)": {
        "title": "Supply Lost by Commodity",
        "body": ("Percentage of each commodity's supply that would be lost if "
                 "imports from the selected country were cut off. Commodities "
                 "with few alternative suppliers are most exposed."),
        "legend": [
            ("< 20%",  "green",  "Manageable"),
            ("20–40%", "amber",  "Significant"),
            ("40–70%", "orange", "Severe"),
            ("> 70%",  "red",    "Critical"),
        ],
        "source": "UN Comtrade HS 01–22",
    },

    # ── A-D7: Seasonal Risk ────────────────────────────────────────────
    "Monthly trade pattern (CSO)": {
        "title": "Monthly Trade Pattern",
        "body": ("Average monthly trade flows showing seasonal patterns. Peaks "
                 "and troughs indicate harvest cycles, stockpiling behaviour, "
                 "and periods of elevated supply-chain stress."),
        "source": "CSO monthly trade statistics",
    },
    "Seasonal risk signal": {
        "title": "Seasonal Risk Signal",
        "body": ("Highlights months where trade volumes deviate significantly "
                 "from the annual average — these are the periods when supply "
                 "disruptions have the largest relative impact."),
        "legend": [
            ("Within 1σ",  "green",  "Normal"),
            ("1–2σ",       "amber",  "Elevated"),
            ("> 2σ",       "red",    "Anomalous"),
        ],
        "source": "CSO monthly data · Z-score vs 12-month mean",
    },

    # ── A-D8: Food Security ────────────────────────────────────────────
    "CSO agricultural output trend": {
        "title": "CSO Agricultural Output Trend",
        "body": ("Value of domestic agricultural output over time, from CSO's "
                 "Agricultural Output statistics. Measures primary production "
                 "before processing, trade, or stock changes."),
        "source": "CSO Agricultural Output and Input datasets",
    },
    "Production vs domestic supply": {
        "title": "Production vs Domestic Supply",
        "body": ("Compares what Ireland produces domestically against total "
                 "domestic food supply (production + imports − exports). The gap "
                 "is filled by net imports."),
        "source": "FAOSTAT Food Balance Sheets",
    },
    "Food categories breakdown": {
        "title": "Food Categories Breakdown",
        "body": ("Share of each food category (cereals, meat, dairy, fish, fruit "
                 "& vegetables, etc.) in total domestic food supply. Reveals "
                 "which categories dominate the Irish diet."),
        "source": "FAOSTAT Food Balance Sheets",
    },
    "Per capita food supply trend (kg/capita/yr)": {
        "title": "Per Capita Food Supply",
        "body": ("Annual food supply per person in kilograms. Trends over time "
                 "reflect dietary shifts, population growth, and changes in "
                 "consumption patterns."),
        "source": "FAOSTAT · kg per capita per year",
    },

    # ═══════════════════════════════════════════════════════════════════
    # MEDTECH SECTOR
    # ═══════════════════════════════════════════════════════════════════

    # ── M-D1: Strategic Overview ───────────────────────────────────────
    "Export value by sector": {
        "title": "Export Value by Sector",
        "body": ("Irish MedTech exports split by HS code: Medical Instruments "
                 "(9018), Implants & Stents (9021), and X-Ray Apparatus (9022). "
                 "Instruments and Implants dominate; Ireland hosts many US MedTech "
                 "manufacturers."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "Import value by sector": {
        "title": "Import Value by Sector",
        "body": ("Irish MedTech imports by parent HS code. Imports are largely "
                 "components and sub-assemblies that are processed and exported "
                 "as finished devices."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "Import vs export balance": {
        "title": "Import vs Export Balance",
        "body": ("Trade surplus or deficit per MedTech sub-sector. Positive bars "
                 "= net exporter. Ireland runs a substantial MedTech surplus, "
                 "reflecting its role as a manufacturing export hub."),
        "legend": [
            ("Positive", "green", "Net exporter"),
            ("Negative", "red",   "Net importer"),
        ],
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },

    # ── M-D2: Import Dependency ────────────────────────────────────────
    "HHI by sector": {
        "title": "HHI by MedTech Sector",
        "body": ("Supplier concentration index for each MedTech sub-sector's "
                 "imports. High HHI in Implants or Instruments would flag a "
                 "strategic dependency on few component suppliers."),
        "legend": [
            ("HHI = 0.00–0.15", "green",  "Competitive"),
            ("HHI = 0.15–0.25", "amber",  "Moderate"),
            ("HHI = 0.25–0.40", "orange", "High"),
            ("HHI > 0.40",      "red",    "Critical"),
        ],
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "Continental breakdown": {
        "title": "Continental Breakdown",
        "body": ("MedTech trade flows grouped by continent. USA and Europe "
                 "dominate both directions, reflecting the integration of Irish "
                 "MedTech with transatlantic supply chains."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "Import trend 2015-2024": {
        "title": "Import Trend 2015–2024",
        "body": ("Annual MedTech import value. Growth reflects expanding "
                 "manufacturing capacity; dips may correspond to COVID-era "
                 "disruptions or inventory cycles."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },

    # ── M-D3: Export Dependency ────────────────────────────────────────
    "Top export markets (parent codes)": {
        "title": "Top Export Markets",
        "body": ("Destination countries for Irish MedTech exports across all "
                 "three parent HS codes. USA is the dominant market (~45%), "
                 "reflecting Ireland's role as manufacturing hub for US firms."),
        "legend": [
            ("USA < 30%",  "green",  "Diversified"),
            ("USA 30–45%", "amber",  "Moderate concentration"),
            ("USA > 45%",  "red",    "High concentration"),
        ],
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "Tariff scenario -- revenue at risk (EUR M)": {
        "title": "Tariff Scenario — Revenue at Risk",
        "body": ("Estimated revenue impact under hypothetical tariff scenarios "
                 "by major markets (particularly US tariffs). Given ~45% US "
                 "share, even modest tariffs translate to material revenue risk."),
        "legend": [
            ("< €100M",   "green",  "Minor"),
            ("€100–500M", "amber",  "Moderate"),
            ("> €500M",   "red",    "Major"),
        ],
        "source": "Expert-judgement multipliers · Not peer-reviewed",
    },

    # ── M-D4: Supply Flow ──────────────────────────────────────────────
    "Supply Flow Sankey": {
        "title": "MedTech Supply Flow",
        "body": ("Sankey diagram tracing MedTech flows from supply origins "
                 "through Ireland to export destinations. Ireland functions as "
                 "a transformation hub — receiving components and shipping "
                 "finished devices."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "Import / export trend (EUR bn)": {
        "title": "Import / Export Trend",
        "body": ("Annual import and export totals in EUR billions. The widening "
                 "gap between the lines represents Ireland's growing MedTech "
                 "trade surplus."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "CAGR summary": {
        "title": "CAGR Summary",
        "body": ("Compound Annual Growth Rate for key MedTech metrics over the "
                 "available time window. Higher CAGRs indicate fast-expanding "
                 "product categories; negative values flag shrinking segments."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022 · 2015–2024",
    },

    # ── M-D5: Trade Map ────────────────────────────────────────────────
    "MedTech Trade Map": {
        "title": "MedTech Trade Map",
        "body": ("Geographic view of MedTech trade flows. Lavender arcs = "
                 "imports into Ireland, teal arcs = exports. Arc thickness "
                 "proportional to trade value."),
        "legend": [
            ("Lavender arcs", "grey",  "Imports into Ireland"),
            ("Teal arcs",     "blue",  "Exports from Ireland"),
        ],
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },

    # ── M-D6: Stress Test ──────────────────────────────────────────────
    "Supply lost by sector if country removed (%)": {
        "title": "Supply Lost by Sector",
        "body": ("If imports from the selected country were cut off, this shows "
                 "the percentage of each MedTech sub-sector's supply that would "
                 "be lost. Reveals which sectors are most exposed to that country."),
        "legend": [
            ("< 20%",  "green",  "Manageable"),
            ("20–40%", "amber",  "Significant"),
            ("40–70%", "orange", "Severe"),
            ("> 70%",  "red",    "Critical"),
        ],
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "Top alternative suppliers": {
        "title": "Top Alternative Suppliers",
        "body": ("Countries that could potentially replace the disrupted "
                 "supplier, based on their current export volumes in the same "
                 "MedTech categories."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022 · Global export data",
    },
    "Impact by MedTech Sector": {
        "title": "Impact by MedTech Sector",
        "body": ("How the stress scenario's losses distribute across Medical "
                 "Instruments, Implants, and X-Ray Apparatus. Sectors with "
                 "higher country exposure show larger impacts."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },

    # ── M-D7: Scenario War Room ────────────────────────────────────────
    # (Scenario Parameters, VaR Distribution, 12-Month Impact Timeline,
    #  Percentile Summary Table already covered above under Energy —
    #  same titles, tooltips apply identically across sectors)

    # ── M-D8: Product Analysis ─────────────────────────────────────────
    "Export value by product (sub-codes)": {
        "title": "Export Value by Product",
        "body": ("Exports broken out at the HS 6-digit level: Catheters (901839), "
                 "Other Instruments (901890), Cardiovascular Implants (902190). "
                 "Reveals which specific product lines drive the surplus."),
        "source": "UN Comtrade HS 901839 / 901890 / 902190",
    },
    "Import value by product (sub-codes)": {
        "title": "Import Value by Product",
        "body": ("Imports at the 6-digit sub-code level. Compare to export "
                 "values for the same codes to identify net-exporter versus "
                 "net-importer product lines."),
        "source": "UN Comtrade HS 901839 / 901890 / 902190",
    },
    "HHI per product (sub-codes, imports)": {
        "title": "HHI per Product",
        "body": ("Import-side concentration index for each 6-digit product. "
                 "High HHI at this granular level may indicate reliance on "
                 "specialised suppliers for specific device components."),
        "legend": [
            ("HHI = 0.00–0.15", "green",  "Competitive"),
            ("HHI = 0.15–0.25", "amber",  "Moderate"),
            ("HHI = 0.25–0.40", "orange", "High"),
            ("HHI > 0.40",      "red",    "Critical"),
        ],
        "source": "UN Comtrade HS 901839 / 901890 / 902190",
    },
    "Top import source per product": {
        "title": "Top Import Source per Product",
        "body": ("Single largest supplier country for each 6-digit product. "
                 "Widely concentrated top sources signal product-specific "
                 "dependency risks."),
        "source": "UN Comtrade HS 901839 / 901890 / 902190",
    },
}


def get_tooltip(info_id):
    """Return the tooltip dict for the given chart title, or None if absent."""
    if not info_id:
        return None
    return CHART_TOOLTIPS.get(info_id)
