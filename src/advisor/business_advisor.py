"""AI business-plan advisor.

Sends the user's business plan plus current market context (trends,
saturation, pricing) to Claude and returns structured pros/cons.
"""

import anthropic

SYSTEM = (
    "You are a business advisor specialized in e-commerce and pricing. "
    "Analyze the user's business plan using the market context provided. "
    "Reply in the same language the user wrote their plan in, in markdown, "
    "with exactly these sections: **Pros** (3-5 points), **Cons and risks** (3-5 points), "
    "**Recommendation** (2-3 sentences, including whether the chosen price and market make sense). "
    "Be concrete and honest, not flattering. Do not use emojis."
)


def analyze_plan(api_key: str, plan: str, market_context: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"CURRENT MARKET CONTEXT:\n{market_context}\n\nUSER'S BUSINESS PLAN:\n{plan}",
        }],
    )
    return msg.content[0].text
