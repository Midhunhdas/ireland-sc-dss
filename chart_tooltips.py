"""
chart_tooltips.py  —  Ireland SC-DSS Dashboard
────────────────────────────────────────────────
Hover-tooltip definitions for every chart in the dashboard.

Keys are the exact strings app.py's chart_info_icon() looks up:
    "sector:chart-id"

where sector ∈ {'energy', 'agri', 'medtech'} and chart-id is the
dcc.Graph ID (e.g. 'e-d1-hhi-bar').

Each entry contains:
    title   — tooltip heading shown in bold orange
    desc    — plain-language description of what the chart shows
    source  — footer line identifying the data source

app.py's chart_info_icon() function reads:
    tt.get("title", "")        → .tt-head
    tt.get("desc", "")         → .tt-desc
    tt.get("source", "")       → .tt-src
    tt.get("measures", [])     → (optional metric rows)
    tt.get("formula", "")      → (optional formula row)

Missing keys cause the "?" icon to render as an empty span (silently absent).
"""

TOOLTIPS = {

    # ═══════════════════════════════════════════════════════════════════
    # ENERGY SECTOR
    # ═══════════════════════════════════════════════════════════════════

    # ── E-D1 Strategic Overview ─────────────────────────────────────────
    "energy:e-d1-hhi-bar": {
        "title": "HHI by Fuel Type",
        "desc":  ("Herfindahl-Hirschman Index for each major fuel — how "
                  "concentrated Ireland's imports are across supplier countries. "
                  "Near 0 = many suppliers (safe). Near 1 = one dominant supplier "
                  "(risky). Bands: <0.15 competitive, 0.15-0.25 moderate, "
                  "0.25-0.40 high, >0.40 critical."),
        "source": "UN Comtrade HS 27xx · Σ(country_share)² per fuel",
    },
    "energy:e-d1-oil-trend": {
        "title": "Oil Import Dependency Trend",
        "desc":  ("Share of Ireland's oil requirement met by imports over time. "
                  "Ireland has no domestic crude production, so this tracks close "
                  "to 100% and reflects demand shifts rather than supply changes."),
        "source": "SEAI National Energy Balance 1990-2024",
    },
    "energy:e-d1-ss-trend": {
        "title": "Self-Sufficiency Trend",
        "desc":  ("Indigenous production as a share of Primary Energy Requirement. "
                  "2024 value: 20.3%. Declining trend driven by Corrib gas field "
                  "running down. 2020 spike (26.7%) was COVID demand collapse, "
                  "not a structural improvement."),
        "source": "SEAI NEB · Indigenous Production ÷ PER",
    },

    # ── E-D2 Import Dependency ──────────────────────────────────────────
    "energy:e-d2-hhi": {
        "title": "HHI by Fuel Type",
        "desc":  ("Same as the strategic overview but shown for the selected "
                  "year(s). Lets you compare supplier concentration across fuels "
                  "for a specific time window. Red bars flag the most concentrated "
                  "fuel categories."),
        "source": "UN Comtrade HS 27xx",
    },
    "energy:e-d2-dep": {
        "title": "Country Dependency Score",
        "desc":  ("Composite dependency score per supplier country, combining "
                  "import share (40%), product dominance (30%) and governance "
                  "risk (30%). Higher = more exposed. UK at ~0.72 reflects its "
                  "dominant share plus high product dominance."),
        "source": "UN Comtrade + World Bank WGI · 0.4·share + 0.3·dominance + 0.3·gov_risk",
    },
    "energy:e-d2-cont": {
        "title": "Continental Breakdown",
        "desc":  ("Imports and exports aggregated by region. Europe dominates "
                  "imports (~70%) driven by UK gas and Netherlands/Belgium refined "
                  "petroleum. Americas share is mostly US crude oil."),
        "source": "UN Comtrade · classified via CONTINENT_MAP",
    },
    "energy:e-d2-ht": {
        "title": "HHI Trend Over Time",
        "desc":  ("Weighted supplier-concentration HHI across all fuels, year by "
                  "year. Rising line = concentration increasing (more risk). "
                  "Critical threshold at 0.40 marked in red."),
        "source": "UN Comtrade · weighted country-HHI aggregate",
    },

    # ── E-D3 Supply Flow ────────────────────────────────────────────────
    "energy:e-d3-sankey": {
        "title": "Supply Flow Sankey",
        "desc":  ("Visualises energy flows from supplier countries through fuel "
                  "categories to Ireland. Thicker bands = larger trade value. "
                  "Reveals both supplier concentration and product-mix dependency "
                  "in a single view."),
        "source": "UN Comtrade · latest year imports by country × product",
    },

    # ── E-D4 Trade Map ──────────────────────────────────────────────────
    "energy:e-d4-map": {
        "title": "Bilateral Energy Trade Map",
        "desc":  ("Geographic view of Ireland's energy trade. Circle size reflects "
                  "trade value; colour distinguishes imports (red) from exports "
                  "(green). Hover each circle for country-level detail."),
        "source": "UN Comtrade · aggregated by partner country",
    },

    # ── E-D6 Stress Testing ─────────────────────────────────────────────
    "energy:e-d6-bar": {
        "title": "Supply Lost by Product",
        "desc":  ("What Ireland loses if the selected country disappears as a "
                  "supplier. Percentages are per-fuel shares of that supplier. "
                  "A UK removal shows ~97% gas loss — the Moffat Interconnector "
                  "single-point-of-failure."),
        "source": "UN Comtrade · per-product share of selected supplier",
    },

    # ── E-D7 Scenario War Room ──────────────────────────────────────────
    "energy:e-d7-hist": {
        "title": "VaR Distribution",
        "desc":  ("Monte Carlo simulation of potential losses across 1,000 "
                  "randomised scenarios. The histogram shows outcome frequency. "
                  "Red line = P95 Value-at-Risk (5% chance of loss exceeding it). "
                  "Green line = P5 best case."),
        "source": "Monte Carlo · loss = base·sev·U(0.7,1.3) + base·(dem-1)·U(0.8,1.2)",
    },
    "energy:e-d7-tl": {
        "title": "12-Month Impact Timeline",
        "desc":  ("Projected cumulative loss month-by-month after disruption "
                  "onset. Shape reflects gradual supply-chain recovery — peak "
                  "impact typically lands months 3-6, then tapers as alternative "
                  "suppliers and stockpiles come online."),
        "source": "Model projection · exponential-decay recovery curve",
    },
    "energy:e-d7-prod": {
        "title": "Impact by Product",
        "desc":  ("Median loss apportioned to each fuel category based on its "
                  "share of total imports. Shows which fuels drive the scenario "
                  "outcome — usually crude petroleum and natural gas by a wide "
                  "margin."),
        "source": "UN Comtrade · P50 loss × per-fuel import share",
    },

    # ── E-D10 ML Analysis ───────────────────────────────────────────────
    "energy:e-d10-anomaly": {
        "title": "Anomaly Detection",
        "desc":  ("IsolationForest flags unusual trade flows. X-axis = trade "
                  "value; Y-axis = anomaly score (lower = more anomalous). Red X "
                  "markers are the outliers: flows that deviate sharply from "
                  "historical pattern in value, year-on-year change, or share."),
        "source": "scikit-learn IsolationForest on (log value, YoY change, partner share)",
    },
    "energy:e-d10-cluster": {
        "title": "K-Means Clustering",
        "desc":  ("Groups fuels by risk profile using (HHI, supplier count, "
                  "volatility). Red cluster = high-risk fuels with concentrated "
                  "supply and volatile pricing. Green = low-risk, diversified. "
                  "Amber = middle ground."),
        "source": "scikit-learn KMeans · 3 clusters · random_state=42",
    },
    "energy:e-d10-forecast": {
        "title": "HHI Forecast to 2030",
        "desc":  ("Historical weighted-HHI series with a linear projection to "
                  "2030. Rising projection (red dashed) = concentration trend "
                  "worsens without intervention. Critical threshold at 0.40 "
                  "marked for reference."),
        "source": "NumPy polyfit · linear regression on 2010-2024 HHI series",
    },
    "energy:e-d10-cascade": {
        "title": "Cascade Failure Simulation",
        "desc":  ("Removes suppliers one at a time in order of largest share. "
                  "Y-axis = cumulative supply lost. Critical threshold (70%) is "
                  "crossed after just 3 removals: UK → Netherlands → USA. A "
                  "resilient supply chain would need 15+ removals."),
        "source": "UN Comtrade · cumulative share of ranked suppliers",
    },

    # ── E-D14 Export Dependency ─────────────────────────────────────────
    "energy:e-d14-pie": {
        "title": "Export Market Concentration",
        "desc":  ("Share of Ireland's energy exports by destination country. "
                  "Ireland's energy exports are much smaller than imports "
                  "(refined products and re-exports mainly), but concentration "
                  "here affects trade balance."),
        "source": "UN Comtrade HS 27xx exports",
    },
    "energy:e-d14-tar": {
        "title": "Tariff Scenario Impact",
        "desc":  ("Revenue-at-risk if the selected tariff rate is applied to "
                  "exports to the chosen market. Pass-through slider controls "
                  "how much of the tariff exporters absorb (margin compression) "
                  "vs. pass to the buyer."),
        "source": "Comtrade × tariff rate × pass-through coefficient",
    },
    "energy:e-d14-vm": {
        "title": "Export Vulnerability Matrix",
        "desc":  ("Heatmap crossing destination markets with product categories. "
                  "Hot cells (red) = high trade value and thus high exposure to a "
                  "market- or product-specific shock. Cool cells = minimal "
                  "revenue at stake."),
        "source": "UN Comtrade · market × product matrix",
    },

    # ═══════════════════════════════════════════════════════════════════
    # AGRICULTURE SECTOR
    # ═══════════════════════════════════════════════════════════════════

    # ── A-D1 Strategic Overview ─────────────────────────────────────────
    "agri:a-d1-commodity": {
        "title": "Export Value by Commodity",
        "desc":  ("Irish agri-food exports broken out by HS chapter. Beef and "
                  "dairy dominate — together they account for roughly half of "
                  "total agri exports. The chart reveals product-level "
                  "concentration risk."),
        "source": "UN Comtrade HS 01-22 + CSO trade statistics",
    },
    "agri:a-d1-market": {
        "title": "Export Market Share",
        "desc":  ("Share of Irish agri exports going to each destination market. "
                  "UK dominates at ~37% — the highest bilateral dependency of any "
                  "Irish export sector. No single alternative market can absorb "
                  "more than 10%."),
        "source": "UN Comtrade · by partner country",
    },
    "agri:a-d1-trend": {
        "title": "Food Self-Sufficiency Trend",
        "desc":  ("Share of consumed food Ireland produces itself. Formula: "
                  "Production / (Production + Imports - Exports). 2022 value: "
                  "74.4% (item-matched). Below 100% = net importer. Below the "
                  "long-run EU average."),
        "source": "FAOSTAT Food Balance Sheets · item-matched inner join",
    },

    # ── A-D2 Import Dependency ──────────────────────────────────────────
    "agri:a-d2-hhi": {
        "title": "Import HHI by Commodity",
        "desc":  ("Supplier concentration for each agri commodity group. Cereals "
                  "and fruit show high HHI because a few partners dominate; beef "
                  "and dairy imports are tiny and fragmented. High bars = "
                  "concentration risk for that category."),
        "source": "UN Comtrade HS 01-22 · Σ(country_share)² per commodity",
    },
    "agri:a-d2-sources": {
        "title": "Top Import Sources",
        "desc":  ("Largest supplier countries for Irish agri imports, ranked by "
                  "value. UK ~39% — same country as our #1 export market, "
                  "creating the double-sided Brexit exposure. Netherlands, "
                  "Germany and France follow as mid-tier suppliers."),
        "source": "UN Comtrade · by partner country",
    },
    "agri:a-d2-cont": {
        "title": "Continental Import Breakdown",
        "desc":  ("Agri imports grouped by region. Europe dominates (UK + EU), "
                  "with meaningful contributions from South America (beef, "
                  "soybean) and Asia (rice, tropical products). Geographic "
                  "diversification is limited."),
        "source": "UN Comtrade · classified by CONTINENT_MAP",
    },
    "agri:a-d2-trend": {
        "title": "Import Trend 2010-2024",
        "desc":  ("Total agri imports over the 15-year dataset window. Upward "
                  "trend reflects population growth and shifting food "
                  "preferences. Brexit caused a visible dip in 2019-2021 that "
                  "has since reversed."),
        "source": "UN Comtrade · annual totals",
    },
    "agri:a-d2-matrix": {
        "title": "Commodity × Source Matrix",
        "desc":  ("Heatmap of the top 10 imported commodities against the top 6 "
                  "source countries. Hot cells identify concentration "
                  "choke-points — specific product-country pairs where a shock "
                  "would hit hardest."),
        "source": "UN Comtrade · product × partner cross-tabulation",
    },

    # ── A-D3 Export Dependency ──────────────────────────────────────────
    "agri:a-d3-markets": {
        "title": "Top Export Markets",
        "desc":  ("Ranked destination markets for Irish agri exports. UK's "
                  "dominance at 37% is a structural feature — geographic "
                  "proximity plus Brexit-era integration. Second-tier markets "
                  "(USA, EU27) matter for diversification."),
        "source": "UN Comtrade · by partner country",
    },
    "agri:a-d3-tariff-fig": {
        "title": "Tariff Impact by Market",
        "desc":  ("Revenue at risk if the chosen tariff rate is imposed on "
                  "exports to the selected market. Pass-through controls "
                  "exporter absorption: 50% = half absorbed as margin "
                  "compression, 100% = full pass to buyer."),
        "source": "Comtrade × tariff × pass-through · WTO-calibrated",
    },
    "agri:a-d3-matrix": {
        "title": "Commodity × Market Matrix",
        "desc":  ("Cross-tabulation of agri commodities and destination markets. "
                  "Red cells = concentrated exposure (high value + single "
                  "market). Reveals which product-market combinations are most "
                  "vulnerable to a targeted shock."),
        "source": "UN Comtrade · product × partner exports matrix",
    },

    # ── A-D4 Supply Flow ────────────────────────────────────────────────
    "agri:a-d4-sankey": {
        "title": "Agriculture Supply Flow",
        "desc":  ("Sankey visualisation of agri trade flows: supplier countries → "
                  "commodity categories → destination markets. Reveals circular "
                  "trade patterns (e.g. UK both as supplier AND destination for "
                  "different product categories)."),
        "source": "UN Comtrade · multi-stage flow aggregation",
    },

    # ── A-D5 Trade Map ──────────────────────────────────────────────────
    "agri:a-d5-map": {
        "title": "Agri Trade Map",
        "desc":  ("Geographic view of Ireland's agri trade. Circle size = trade "
                  "value. Colour distinguishes imports from exports. Hover any "
                  "circle for the bilateral breakdown with that country."),
        "source": "UN Comtrade · by partner country",
    },

    # ── A-D6 Stress Test ────────────────────────────────────────────────
    "agri:a-d6-bar": {
        "title": "Supply Lost by Commodity",
        "desc":  ("What Ireland loses per commodity if the selected country "
                  "disappears as an agri supplier. UK removal shows widespread "
                  "high losses because UK is the single largest source across "
                  "most categories."),
        "source": "UN Comtrade · per-commodity share of selected supplier",
    },

    # ── A-D7 Seasonal Risk ──────────────────────────────────────────────
    "agri:a-d7-monthly": {
        "title": "Monthly Trade Pattern",
        "desc":  ("Month-by-month agri trade from CSO data, sorted "
                  "chronologically. Reveals the seasonality annual data hides — "
                  "domestic production peaks in spring/summer, imports spike in "
                  "winter when local supply is at its minimum."),
        "source": "CSO Ireland · monthly trade by commodity group",
    },
    "agri:a-d7-risk": {
        "title": "Seasonal Risk Signal",
        "desc":  ("Bi-monthly z-score measuring how unusual each period's trade "
                  "volume is vs. the annual average. High bars = extreme "
                  "deviation (either surge or trough). Jan-Feb typically scores "
                  "highest — the compound-risk window."),
        "source": "CSO monthly data · z = |bi_sum - 2μ| / (2σ), rescaled 0-100",
    },
    "agri:a-d7-output": {
        "title": "Agricultural Output Trend",
        "desc":  ("CSO's index of agricultural output over time — production "
                  "volume and price effects combined. Provides a "
                  "production-side counterbalance to the trade-based supply "
                  "risk indicators on this page."),
        "source": "CSO Ireland · Agricultural Output Index",
    },

    # ── A-D8 Food Security ──────────────────────────────────────────────
    "agri:a-d8-balance": {
        "title": "Production vs Domestic Supply",
        "desc":  ("Per-item comparison: how much Ireland produces versus how "
                  "much it consumes. Categories far above the diagonal = big net "
                  "exporters (beef, dairy). Far below = heavy net importers "
                  "(cereals, fruit, vegetables)."),
        "source": "FAOSTAT Food Balance Sheets · item-level production vs domestic supply",
    },
    "agri:a-d8-items": {
        "title": "Food Categories Breakdown",
        "desc":  ("Item-level self-sufficiency ratios. Beef ~400%+, dairy ~500% "
                  "(massive net exporter). Cereals ~60%, fruit/veg ~25% (heavy "
                  "net importer). Aggregates to 74.4% overall SSR but with huge "
                  "variation by category."),
        "source": "FAOSTAT · item-level SSR computations",
    },
    "agri:a-d8-percapita": {
        "title": "Per Capita Food Supply Trend",
        "desc":  ("Kilograms of food supply per capita per year, tracked over "
                  "time. Reveals long-term dietary shifts — meat consumption "
                  "plateau, rising vegetable intake, processed food growth. "
                  "Supplements the aggregate SSR with a demand-side view."),
        "source": "FAOSTAT Food Balance Sheets · per-capita kg/year",
    },

    # ═══════════════════════════════════════════════════════════════════
    # MEDICAL TECHNOLOGY SECTOR
    # ═══════════════════════════════════════════════════════════════════

    # ── M-D1 Strategic Overview ─────────────────────────────────────────
    "medtech:m-d1-exp-bar": {
        "title": "Export Value by Sector",
        "desc":  ("Irish MedTech exports split by HS code: Medical Instruments "
                  "(9018), Implants & Stents (9021), and X-Ray Apparatus (9022). "
                  "Instruments and Implants dominate. Ireland ranks 2nd in EU "
                  "MedTech exports."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "medtech:m-d1-imp-bar": {
        "title": "Import Value by Sector",
        "desc":  ("MedTech imports by product category. Much smaller than "
                  "exports — Ireland runs a 4.6× trade surplus in MedTech. "
                  "Import mix shows which categories Ireland depends on foreign "
                  "suppliers for inputs or finished goods."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "medtech:m-d1-balance": {
        "title": "Import vs Export Balance",
        "desc":  ("Net trade balance per MedTech category. All three parent "
                  "codes show substantial export surplus. Absolute sizes signal "
                  "which product line is most strategically important to Irish "
                  "MedTech exports."),
        "source": "UN Comtrade · export minus import per HS code",
    },

    # ── M-D2 Import Dependency ──────────────────────────────────────────
    "medtech:m-d2-hhi": {
        "title": "HHI by Sector",
        "desc":  ("Import supplier concentration for each MedTech category. "
                  "HS 9021 sits at 0.209 — moderately concentrated but improved "
                  "from 0.319 in 2015 (a decade of visible diversification). "
                  "HS 9018 at 0.168 is the most diversified."),
        "source": "UN Comtrade HS 9018/9021/9022 · Σ(country_share)² per category",
    },
    "medtech:m-d2-sources": {
        "title": "Top Import Sources",
        "desc":  ("Supplier countries ranked by MedTech import value. USA "
                  "dominates at ~34%. Germany, UK, Belgium follow. The "
                  "Asia-Pacific share (China, Japan, Singapore) has grown "
                  "steadily — meaningful for supply diversification."),
        "source": "UN Comtrade · by partner country for MedTech HS codes",
    },
    "medtech:m-d2-cont": {
        "title": "Continental Breakdown",
        "desc":  ("MedTech imports by region. Americas (USA-dominated) and "
                  "Europe together account for most imports. Asia's share has "
                  "grown notably post-COVID as firms diversified away from "
                  "single-continent dependencies."),
        "source": "UN Comtrade · classified via CONTINENT_MAP",
    },
    "medtech:m-d2-trend": {
        "title": "Import Trend 2015-2024",
        "desc":  ("Total MedTech imports over the 10-year window. Steady growth "
                  "reflects expanding domestic MedTech production needing inputs, "
                  "plus growing healthcare spending. Dip in 2020 was COVID "
                  "supply disruption, quickly recovered."),
        "source": "UN Comtrade · annual totals HS 9018/9021/9022",
    },
    "medtech:m-d2-conc": {
        "title": "Import Source Concentration (Top 5)",
        "desc":  ("Combined share of the top 5 supplier countries for MedTech "
                  "imports. Typically around 80%, showing that while dozens of "
                  "countries supply MedTech, the bulk comes from a small group "
                  "of major trading partners."),
        "source": "UN Comtrade · cumulative share of top-N suppliers",
    },

    # ── M-D3 Export Dependency ──────────────────────────────────────────
    "medtech:m-d3-markets": {
        "title": "Top Export Markets",
        "desc":  ("Destination markets for Irish MedTech exports. USA is #1 at "
                  "~35% — the single largest exposure in any Irish sector. "
                  "Germany, UK, Netherlands, Belgium follow. Full list is much "
                  "more diversified than imports."),
        "source": "UN Comtrade · exports HS 9018/9021/9022 by partner",
    },
    "medtech:m-d3-sectors": {
        "title": "Export Value by Sector",
        "desc":  ("MedTech exports broken out by HS code for the selected year "
                  "and filters. Reveals which product line is most exposed to a "
                  "market-specific shock like tariff changes on instruments "
                  "vs. implants."),
        "source": "UN Comtrade HS 9018 / 9021 / 9022",
    },
    "medtech:m-d3-conc": {
        "title": "Export Market Concentration (Top 5)",
        "desc":  ("Combined share of the top 5 destination markets. Usually "
                  "70-80%, highlighting that Irish MedTech is heavily dependent "
                  "on a handful of rich-country markets rather than broadly "
                  "diversified worldwide."),
        "source": "UN Comtrade · cumulative share of top-N markets",
    },
    "medtech:m-d3-tariff-fig": {
        "title": "Tariff Scenario Impact",
        "desc":  ("Revenue at risk if the selected tariff is imposed on "
                  "MedTech exports to the chosen market. Pass-through slider "
                  "controls exporter absorption: 0.5 = exporters absorb half, "
                  "1.0 = full tariff passed to the buyer."),
        "source": "Comtrade × tariff × pass-through · WTO-calibrated",
    },

    # ── M-D4 Supply Flow ────────────────────────────────────────────────
    "medtech:m-d4-sankey": {
        "title": "Supply Flow Sankey",
        "desc":  ("MedTech trade flows visualised: suppliers → product "
                  "categories → markets. Reveals cross-border intra-firm "
                  "patterns (many MedTech multinationals have Ireland plants "
                  "that import inputs and re-export finished goods)."),
        "source": "UN Comtrade · multi-stage flow aggregation",
    },
    "medtech:m-d4-trend": {
        "title": "Import / Export Trend",
        "desc":  ("Annual import and export totals side-by-side over time. "
                  "The widening gap illustrates Ireland's growing MedTech "
                  "trade surplus — exports have grown faster than imports "
                  "through the decade."),
        "source": "UN Comtrade · annual totals",
    },

    # ── M-D5 Trade Map ──────────────────────────────────────────────────
    "medtech:m-d5-map": {
        "title": "MedTech Trade Map",
        "desc":  ("Geographic view of Ireland's MedTech trade. Circle size "
                  "reflects trade value; colour distinguishes imports from "
                  "exports. Hover any country for the bilateral trade detail "
                  "across the three HS codes."),
        "source": "UN Comtrade · aggregated by partner country",
    },

    # ── M-D6 Stress Test ────────────────────────────────────────────────
    "medtech:m-d6-bar": {
        "title": "Supply Lost by Sector",
        "desc":  ("What Ireland loses per MedTech category if the selected "
                  "supplier is removed. USA removal shows high losses "
                  "concentrated on HS 9022 (X-Ray apparatus) — that's the most "
                  "US-dependent MedTech subcategory."),
        "source": "UN Comtrade · per-category share of selected supplier",
    },

    # ── M-D7 Scenario War Room ──────────────────────────────────────────
    "medtech:m-d7-hist": {
        "title": "VaR Distribution",
        "desc":  ("1,000-run Monte Carlo simulation of MedTech losses under the "
                  "configured scenario. Red line = P95 Value-at-Risk. The "
                  "distribution shape indicates how much tail risk exists — "
                  "right-skewed = asymmetric downside."),
        "source": "Monte Carlo · sklearn-seeded, random_state=42",
    },
    "medtech:m-d7-tl": {
        "title": "12-Month Impact Timeline",
        "desc":  ("Cumulative projected loss month-by-month after disruption. "
                  "MedTech recovery tends to be slower than energy because "
                  "alternative suppliers take time to scale, requalify products, "
                  "and clear regulatory hurdles."),
        "source": "Model projection · exponential-decay recovery curve",
    },
    "medtech:m-d7-prod": {
        "title": "Impact by MedTech Sector",
        "desc":  ("Median loss apportioned across the three MedTech categories. "
                  "Shows which product line absorbs the largest share of the "
                  "scenario hit — typically HS 9018 (instruments) by share, "
                  "HS 9021 by absolute trade value."),
        "source": "UN Comtrade × Monte Carlo · P50 × per-category share",
    },

    # ── M-D8 Product Analysis ───────────────────────────────────────────
    "medtech:m-d8-exp-bar": {
        "title": "Export Value by Product (Sub-Codes)",
        "desc":  ("Deeper product-level view using 6-digit sub-codes: Catheters "
                  "(901839), Other Instruments (901890), Cardiovascular Implants "
                  "(902190). Reveals product-specific trade patterns hidden "
                  "inside parent-code aggregates."),
        "source": "UN Comtrade HS 901839 / 901890 / 902190 exports",
    },
    "medtech:m-d8-imp-bar": {
        "title": "Import Value by Product (Sub-Codes)",
        "desc":  ("Sub-code level imports. Which specific MedTech products "
                  "Ireland imports most — typically specialised instruments and "
                  "high-end diagnostic components. Smaller but important for "
                  "production continuity."),
        "source": "UN Comtrade HS 901839 / 901890 / 902190 imports",
    },
    "medtech:m-d8-hhi": {
        "title": "HHI per Product (Sub-Codes)",
        "desc":  ("Supplier concentration HHI for each 6-digit MedTech product. "
                  "Reveals granular concentration risk: some sub-codes have "
                  "highly concentrated supply (specialised equipment from 1-2 "
                  "countries), others are well-diversified."),
        "source": "UN Comtrade · Σ(country_share)² per 6-digit HS code",
    },


    # Aliases for MedTech cards that don't pass explicit info_id
    # (these cards lookup by title so we need title-keyed entries too):
    "medtech:Import source concentration -- top 5 countries": {
        "title": "Import Source Concentration (Top 5)",
        "desc":  ("Combined share of the top 5 supplier countries for MedTech "
                  "imports. Typically around 80%, showing that while dozens of "
                  "countries supply MedTech, the bulk comes from a small group "
                  "of major trading partners."),
        "source": "UN Comtrade · cumulative share of top-N suppliers",
    },
    "medtech:Export market concentration -- top 5": {
        "title": "Export Market Concentration (Top 5)",
        "desc":  ("Combined share of the top 5 destination markets for Irish "
                  "MedTech exports. Usually 70-80%, highlighting that Irish "
                  "MedTech is heavily dependent on a handful of rich-country "
                  "markets rather than broadly diversified worldwide."),
        "source": "UN Comtrade · cumulative share of top-N markets",
    },

}


def get_tooltip(key):
    """Retrieve a tooltip entry by key, returning None if absent."""
    return TOOLTIPS.get(key)


# Backwards-compatibility aliases.
# Some versions of app.py import CHART_TOOLTIPS instead of TOOLTIPS.
CHART_TOOLTIPS = TOOLTIPS
