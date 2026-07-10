"""AI business-plan advisor.

Sends the user's business plan plus current market context (trends,
saturation, pricing) to Claude and returns structured pros/cons.
"""

import anthropic

SYSTEM = (
    "Eres un asesor de negocios experto en e-commerce y pricing. "
    "Analiza el plan de negocio del usuario usando el contexto de mercado que se te da. "
    "Responde en español, en markdown, con exactamente estas secciones: "
    "**✅ Pros** (3-5 puntos), **⚠️ Contras y riesgos** (3-5 puntos), "
    "**💡 Recomendación** (2-3 frases, incluyendo si el precio y el continente elegidos tienen sentido). "
    "Sé concreto y honesto, no complaciente."
)


def analyze_plan(api_key: str, plan: str, market_context: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"CONTEXTO DE MERCADO ACTUAL:\n{market_context}\n\nPLAN DE NEGOCIO DEL USUARIO:\n{plan}",
        }],
    )
    return msg.content[0].text
