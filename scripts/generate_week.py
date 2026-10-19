
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-flash-latest"

RASHIS = [
    "Mesha", "Vrishabha", "Mithuna", "Karka",
    "Simha", "Kanya", "Tula", "Vrischika",
    "Dhanu", "Makara", "Kumbha", "Meena"
]

today = date.today()
# Use the Indian date, not the runner's UTC date.
from datetime import datetime
today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

result = {
    "generated_at": today.isoformat(),
    "timezone": "Asia/Kolkata",
    "rashifal": {}
}

for offset in range(7):
    day = today + timedelta(days=offset)
    readings = {}

    prompt = f"""
Create original daily Vedic-style Rashi Bhavishya
for {day.isoformat()} for all 12 rashis:
{", ".join(RASHIS)}.

Return ONLY valid JSON: an object whose keys are
the 12 exact rashi names above. Each value must
contain these string fields:
general, career, finance, relationships, wellbeing.

Write concise, distinct, non-alarming readings.
Do not claim certainty about future events.
Do not invent Panchanga or planetary positions.
Treat this as spiritual/entertainment content.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    text = response.text or ""
    # Remove a Markdown fence if the model adds one.
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip()
    )
    readings = json.loads(text)

    if set(readings.keys()) != set(RASHIS):
        raise ValueError(f"Missing or extra rashis for {day}")

    required = {
        "general", "career", "finance",
        "relationships", "wellbeing"
    }
    for rashi, reading in readings.items():
        if not isinstance(reading, dict):
            raise ValueError(f"Invalid reading: {rashi}")
        if not required.issubset(reading.keys()):
            raise ValueError(f"Missing fields: {rashi}")

    result["rashifal"][day.isoformat()] = readings

output = Path("data/rashifal-week.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"Saved weekly Rashifal to {output}")
