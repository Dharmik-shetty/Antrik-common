"""
generate_week.py
Generates 7 days of authentic Indian Panchanga and Vedic Rashifal
using Gemini 2.5 Flash and saves to data/panchang-rashifal-week.json.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from google import genai
from google.genai import types

# 1. API Client Setup
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable is missing.", file=sys.stderr)
    sys.exit(1)

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

RASHIS = [
    "Mesha", "Vrishabha", "Mithuna", "Karka",
    "Simha", "Kanya", "Tula", "Vrischika",
    "Dhanu", "Makara", "Kumbha", "Meena"
]

# Strict JSON Schema for Gemini Structured Outputs
DAY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "date": {"type": "STRING"},
        "day_of_week": {"type": "STRING"},
        "panchang": {
            "type": "OBJECT",
            "properties": {
                "tithi": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "paksha": {"type": "STRING", "enum": ["Shukla Paksha", "Krishna Paksha"]},
                        "description": {"type": "STRING"}
                    },
                    "required": ["name", "paksha"]
                },
                "nakshatra": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "ruler": {"type": "STRING"}
                    },
                    "required": ["name", "ruler"]
                },
                "yoga": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "nature": {"type": "STRING"}
                    },
                    "required": ["name", "nature"]
                },
                "karana": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"}
                    },
                    "required": ["name"]
                },
                "calendar": {
                    "type": "OBJECT",
                    "properties": {
                        "vikram_samvat": {"type": "INTEGER"},
                        "shaka_samvat": {"type": "INTEGER"},
                        "amanta_month": {"type": "STRING"},
                        "purnimanta_month": {"type": "STRING"},
                        "ayana": {"type": "STRING"},
                        "ritu": {"type": "STRING"}
                    },
                    "required": ["vikram_samvat", "amanta_month", "purnimanta_month", "ayana"]
                },
                "timings_ist": {
                    "type": "OBJECT",
                    "properties": {
                        "sunrise": {"type": "STRING"},
                        "sunset": {"type": "STRING"},
                        "rahu_kaal": {"type": "STRING"},
                        "abhijit_muhurat": {"type": "STRING"}
                    },
                    "required": ["sunrise", "sunset", "rahu_kaal", "abhijit_muhurat"]
                },
                "festival_or_vrat": {"type": "STRING"}
            },
            "required": ["tithi", "nakshatra", "yoga", "karana", "calendar", "timings_ist"]
        },
        "rashifal": {
            "type": "OBJECT",
            "properties": {
                rashi: {
                    "type": "OBJECT",
                    "properties": {
                        "summary": {"type": "STRING"},
                        "general": {"type": "STRING"},
                        "career": {"type": "STRING"},
                        "finance": {"type": "STRING"},
                        "relationships": {"type": "STRING"},
                        "health": {"type": "STRING"},
                        "lucky_number": {"type": "INTEGER"},
                        "lucky_color": {"type": "STRING"},
                        "auspicious_time": {"type": "STRING"},
                        "remedy": {"type": "STRING"},
                        "ratings": {
                            "type": "OBJECT",
                            "properties": {
                                "career": {"type": "INTEGER"},
                                "finance": {"type": "INTEGER"},
                                "love": {"type": "INTEGER"},
                                "health": {"type": "INTEGER"}
                            },
                            "required": ["career", "finance", "love", "health"]
                        }
                    },
                    "required": [
                        "summary", "general", "career", "finance",
                        "relationships", "health", "lucky_number",
                        "lucky_color", "remedy", "ratings"
                    ]
                }
                for rashi in RASHIS
            },
            "required": RASHIS
        }
    },
    "required": ["date", "day_of_week", "panchang", "rashifal"]
}

def generate_day_reading(target_date):
    date_str = target_date.strftime("%Y-%m-%d")
    day_name = target_date.strftime("%A")

    prompt = f"""You are a master Vedic Astrologer (Jyotish Acharya) using Nirayana Chitrapaksha (Lahiri) Ayanamsha.

Generate the authentic Drik Panchanga and 12-Rashi Rashifal for:
Date: {date_str} ({day_name})
Standard Reference: Indian Standard Time (IST, UTC+05:30), New Delhi / Central India anchor.

Astrological Requirements:
1. Panchanga:
   - Calculate exact Tithi, Nakshatra, Yoga, and Karana in IST.
   - Include both Amanta (South/West India) and Purnimanta (North India) lunar months.
   - Calculate standard IST Rahu Kaal, Abhijit Muhurat, Sunrise, and Sunset.
   - Note any Vrat or Hindu festival (or "Regular day" if none).
2. Rashifal (All 12 Rashis):
   - Determine predictions based on the Moon's transit (Chandra Gochar) on this date.
   - Write insightful, non-fatalistic, inspiring predictions.
   - Provide lucky number (1-9), lucky color, best time window, ratings (1-5), and a simple Vedic remedy (Upay).

Ensure all 12 Rashis ({', '.join(RASHIS)}) are populated with high quality.
"""

    # Retry up to 3 times in case of rate limits or transient errors
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DAY_SCHEMA,
                    temperature=0.2  # Low temperature for astronomical & factual consistency
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[{date_str}] Attempt {attempt + 1} failed: {e}")
            time.sleep(3 * (attempt + 1))

    raise RuntimeError(f"Failed to generate data for {date_str} after 3 attempts.")

def main():
    # Indian Standard Time anchor
    ist = ZoneInfo("Asia/Kolkata")
    today = datetime.now(ist).date()

    print(f"Generating 7-day Panchang & Rashifal starting from {today.isoformat()} (IST)...")

    weekly_data = {
        "metadata": {
            "generated_at": datetime.now(ist).isoformat(),
            "timezone": "Asia/Kolkata",
            "reference_standard": "New Delhi / Central India (IST)",
            "ayanamsha": "Lahiri / Chitrapaksha",
            "coverage": {
                "start_date": today.isoformat(),
                "end_date": (today + timedelta(days=6)).isoformat()
            }
        },
        "days": {}
    }

    for offset in range(7):
        target_date = today + timedelta(days=offset)
        date_str = target_date.strftime("%Y-%m-%d")
        print(f" -> Fetching {date_str} ({target_date.strftime('%A')})...")

        day_payload = generate_day_reading(target_date)
        weekly_data["days"][date_str] = day_payload
        time.sleep(1)  # Gentle spacing between requests

    output_path = Path("data/panchang-rashifal-week.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(weekly_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nSuccessfully saved 7-day forecast to {output_path}")

if __name__ == "__main__":
    main()
