"""
fx_rates.py
───────────
Annual average EUR/USD exchange rates for converting UN Comtrade USD values to EUR.

Source: European Central Bank reference rates (annual averages), cross-checked
against exchange-rates.org and poundsterlinglive.com. Rates express USD -> EUR:
multiplying a USD value by the rate for that year gives the equivalent EUR value.

Used by consolidate_energy.py, consolidate_agri.py, consolidate_medtech_new.py.
"""

# USD -> EUR annual average conversion rates
# (i.e., 1 USD = X EUR in that year, on an annual average basis)
USD_TO_EUR = {
    2010: 0.7549,
    2011: 0.7195,
    2012: 0.7782,
    2013: 0.7532,
    2014: 0.7537,
    2015: 0.9015,
    2016: 0.9038,
    2017: 0.8870,
    2018: 0.8476,
    2019: 0.8934,
    2020: 0.8771,
    2021: 0.8464,
    2022: 0.9510,
    2023: 0.9243,
    2024: 0.9233,
    2025: 0.9233,   # placeholder — equal to 2024; will be updated when ECB publishes 2025 avg
    2026: 0.9233,   # placeholder
}

# Fallback rate for any year not in the table
FALLBACK_RATE = 0.90


def usd_to_eur(value_usd, year):
    """
    Convert a USD value to EUR using the annual average rate for that year.
    Falls back to FALLBACK_RATE for unknown years.
    """
    try:
        year = int(year)
    except (ValueError, TypeError):
        return value_usd * FALLBACK_RATE
    rate = USD_TO_EUR.get(year, FALLBACK_RATE)
    return value_usd * rate


def get_rate(year):
    """Return the annual EUR/USD rate for a given year, or fallback."""
    try:
        return USD_TO_EUR.get(int(year), FALLBACK_RATE)
    except (ValueError, TypeError):
        return FALLBACK_RATE
