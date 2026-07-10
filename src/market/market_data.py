"""Market data by continent (synthetic but realistic).

For each continent: price level vs. global base price, number of active
sellers, and a demand index. Used to compute average market price,
saturation and competitiveness of the user's price.
"""

# ISO-3 codes used to draw each continent on the map
CONTINENTS = {
    "North America": {
        "price_mult": 1.10, "sellers": 850, "demand_index": 90,
        "countries": ["USA", "CAN", "MEX", "GTM", "CUB", "DOM", "HND", "PAN", "CRI"],
    },
    "South America": {
        "price_mult": 0.85, "sellers": 320, "demand_index": 55,
        "countries": ["BRA", "ARG", "CHL", "COL", "PER", "VEN", "ECU", "BOL", "PRY", "URY"],
    },
    "Europe": {
        "price_mult": 1.15, "sellers": 990, "demand_index": 85,
        "countries": ["ESP", "FRA", "DEU", "ITA", "GBR", "PRT", "NLD", "BEL", "POL",
                      "SWE", "NOR", "FIN", "AUT", "CHE", "GRC", "IRL", "DNK", "CZE",
                      "ROU", "HUN", "UKR"],
    },
    "Asia": {
        "price_mult": 0.80, "sellers": 1400, "demand_index": 95,
        "countries": ["CHN", "JPN", "IND", "KOR", "IDN", "THA", "VNM", "PHL", "MYS",
                      "SGP", "PAK", "BGD", "SAU", "ARE", "TUR", "ISR", "KAZ"],
    },
    "Africa": {
        "price_mult": 0.70, "sellers": 180, "demand_index": 40,
        "countries": ["ZAF", "NGA", "EGY", "KEN", "MAR", "GHA", "ETH", "TZA", "DZA",
                      "TUN", "SEN", "CIV", "CMR", "AGO", "MOZ"],
    },
    "Oceania": {
        "price_mult": 1.05, "sellers": 120, "demand_index": 60,
        "countries": ["AUS", "NZL", "PNG", "FJI"],
    },
}

# per-product saturation adjustment: how crowded each category is globally (0-1)
PRODUCT_CROWDING = {
    "Wireless Earbuds": 0.90,
    "Smart Watch": 0.80,
    "Coffee Maker": 0.55,
    "Yoga Mat": 0.65,
    "Gaming Mouse": 0.75,
    "Protein Powder": 0.70,
}


def market_summary(product: str, base_price: float, continent: str) -> dict:
    """Average market price, saturation level and seller count for a product+continent."""
    c = CONTINENTS[continent]
    avg_price = round(base_price * c["price_mult"], 2)

    # saturation: sellers relative to demand, scaled by product crowding
    ratio = (c["sellers"] / 10) / max(c["demand_index"], 1) * PRODUCT_CROWDING[product]
    score = min(ratio, 1.5) / 1.5  # 0-1
    if score > 0.66:
        level, emoji = "Saturado", "🔴"
    elif score > 0.40:
        level, emoji = "Competencia moderada", "🟡"
    else:
        level, emoji = "Oportunidad", "🟢"

    return {
        "avg_market_price": avg_price,
        "sellers": c["sellers"],
        "demand_index": c["demand_index"],
        "saturation_score": round(score, 2),
        "saturation_level": level,
        "saturation_emoji": emoji,
    }


def competitiveness(your_price: float, avg_market_price: float, saturation_score: float) -> dict:
    """Verdict: is your price competitive in this market?"""
    diff_pct = (your_price - avg_market_price) / avg_market_price * 100
    # in saturated markets you need to undercut; in open markets there is margin room
    threshold = 5 if saturation_score > 0.66 else 12 if saturation_score > 0.40 else 20

    if diff_pct <= 0:
        verdict, emoji = "Muy competitivo", "✅"
        note = "Estás por debajo del precio medio del mercado."
    elif diff_pct <= threshold:
        verdict, emoji = "Competitivo", "🟢"
        note = "Estás por encima de la media, pero dentro del rango aceptable para este nivel de competencia."
    elif diff_pct <= threshold * 2:
        verdict, emoji = "Poco competitivo", "🟠"
        note = "Precio notablemente superior a la media: necesitarás diferenciación (marca, calidad, envío)."
    else:
        verdict, emoji = "No competitivo", "🔴"
        note = "Precio muy por encima del mercado para el nivel de saturación actual."

    return {"diff_pct": round(diff_pct, 1), "verdict": verdict, "emoji": emoji, "note": note}
