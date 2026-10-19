"""
generate_week_multilingual.py
Generates 7 days of authentic Indian Panchanga and Vedic Rashifal in English,
then localizes into 7 Indic languages (8 languages total) using Gemini Flash.

Designed for Gemini Free Tier (15 RPM safety, token-efficient, automatic retry).
Saves individual language JSON files to data/panchang-rashifal-week-<lang>.json.
"""

import copy
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. Configuration & Client Setup
# ---------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable is missing.", file=sys.stderr)
    sys.exit(1)

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.8-flash"

# Target Languages: Base English + 7 Indic languages
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada"
}

RASHIS = [
    "Mesha", "Vrishabha", "Mithuna", "Karka",
    "Simha", "Kanya", "Tula", "Vrischika",
    "Dhanu", "Makara", "Kumbha", "Meena"
]

# Free-tier rate limit spacer: 15 RPM max -> 4.5s safe interval (~13 req/min)
REQUEST_SPACING_SECONDS = 4.5

# ---------------------------------------------------------
# 2. Schemas
# ---------------------------------------------------------

# Schema 1: Base Generation (English)
BASE_DAY_SCHEMA = {
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
                    "required": ["name", "paksha", "description"]
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

# Schema 2: Translation (Only text fields to stay strictly below output token limits)
TRANSLATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "panchang": {
            "type": "OBJECT",
            "properties": {
                "tithi_description": {"type": "STRING"},
                "festival_or_vrat": {"type": "STRING"}
            },
            "required": ["tithi_description", "festival_or_vrat"]
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
                        "lucky_color": {"type": "STRING"},
                        "auspicious_time": {"type": "STRING"},
                        "remedy": {"type": "STRING"}
                    },
                    "required": [
                        "summary", "general", "career", "finance",
                        "relationships", "health", "lucky_color",
                        "auspicious_time", "remedy"
                    ]
                }
                for rashi in RASHIS
            },
            "required": RASHIS
        }
    },
    "required": ["panchang", "rashifal"]
}

# ---------------------------------------------------------
# 3. API Execution with Retry & Backoff
# ---------------------------------------------------------
def call_gemini_with_retry(prompt: str, schema: dict, task_name: str, max_retries: int = 4) -> dict:
    """Invokes Gemini API with exponential backoff for Free Tier rate limits."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2
                )
            )
            return json.loads(response.text)
        except Exception as e:
            wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s, 40s
            print(f"   [!] {task_name} attempt {attempt + 1} encountered an error: {e}")
            if attempt < max_retries - 1:
                print(f"       Backing off for {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed {task_name} after {max_retries} attempts.")

# ---------------------------------------------------------
# 4. Stage 1: Base Generation (English)
# ---------------------------------------------------------
def generate_base_day(target_date: datetime.date) -> dict:
    date_str = target_date.strftime("%Y-%m-%d")
    day_name = target_date.strftime("%A")

    prompt = f"""You are a master Vedic Astrologer (Jyotish Acharya) using Nirayana Chitrapaksha (Lahiri) Ayanamsha.

Generate authentic Drik Panchanga and 12-Rashi Rashifal for:
Date: {date_str} ({day_name})
Reference: Indian Standard Time (IST, UTC+05:30), New Delhi anchor.

Astrological Requirements:
1. Panchanga:
   - Calculate exact Tithi, Nakshatra, Yoga, and Karana in IST.
   - Include both Amanta and Purnimanta lunar months.
   - Provide standard IST Rahu Kaal, Abhijit Muhurat, Sunrise, and Sunset.
   - Note any Vrat or Hindu festival (or 'Regular day' if none).
2. Rashifal (All 12 Rashis):
   - Determine predictions based on the Moon's transit (Chandra Gochar).
   - Write insightful, non-fatalistic, empowering predictions.
   - Provide lucky number (1-9), lucky color, best time window, ratings (1-5), and a simple Vedic remedy (Upay).

Ensure all 12 Rashis ({', '.join(RASHIS)}) are populated with high quality.
"""
    return call_gemini_with_retry(prompt, BASE_DAY_SCHEMA, task_name=f"Base Gen {date_str}")

# ---------------------------------------------------------
# 5. Stage 2: Localization (Translation & Adaptation)
# ---------------------------------------------------------
def translate_day(day_data: dict, lang_code: str, lang_name: str) -> dict:
    """Translates only text fields to maintain 100% numerical consistency."""
    date_str = day_data["date"]

    # Extract translatable fields only
    translatable_payload = {
        "panchang": {
            "tithi_description": day_data["panchang"]["tithi"].get("description", ""),
            "festival_or_vrat": day_data["panchang"].get("festival_or_vrat", "")
        },
        "rashifal": {
            rashi: {
                k: day_data["rashifal"][rashi][k]
                for k in [
                    "summary", "general", "career", "finance",
                    "relationships", "health", "lucky_color",
                    "auspicious_time", "remedy"
                ]
            }
            for rashi in RASHIS
        }
    }

    prompt = f"""You are an expert Vedic Astrologer and linguist specializing in {lang_name} ({lang_code}).
Translate and culturally adapt the following daily Panchanga and Rashifal text from English into authentic {lang_name}.

Rules:
1. Write natively in {lang_name} script with authentic Vedic astrological terminology.
2. Tone must be uplifting, respectful, and astrologically precise.
3. Keep the translation strictly faithful to the original predictions and remedies.

Source JSON:
{json.dumps(translatable_payload, ensure_ascii=False)}
"""

    translated_fields = call_gemini_with_retry(
        prompt, 
        TRANSLATION_SCHEMA, 
        task_name=f"Translate {date_str} to {lang_name}"
    )

    # Deep-merge back into the original day structure
    localized_day = copy.deepcopy(day_data)
    localized_day["panchang"]["tithi"]["description"] = translated_fields["panchang"]["tithi_description"]
    localized_day["panchang"]["festival_or_vrat"] = translated_fields["panchang"]["festival_or_vrat"]

    for rashi in RASHIS:
        for key in ["summary", "general", "career", "finance", "relationships", "health", "lucky_color", "auspicious_time", "remedy"]:
            localized_day["rashifal"][rashi][key] = translated_fields["rashifal"][rashi][key]

    return localized_day

# ---------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------
def main():
    ist = ZoneInfo("Asia/Kolkata")
    today = datetime.now(ist).date()
    start_date_str = today.isoformat()
    end_date_str = (today + timedelta(days=6)).isoformat()

    print(f"============================================================")
    print(f" Starting 7-Day Multilingual Panchang & Rashifal Generation ")
    print(f" Coverage: {start_date_str} to {end_date_str} (IST)")
    print(f" Target Languages: {', '.join(LANGUAGES.values())} (8 total)")
    print(f" Rate Limiting: {REQUEST_SPACING_SECONDS}s between calls (Safe for Free Tier)")
    print(f"============================================================\n")

    # Storage for all 8 language datasets
    datasets = {
        code: {
            "metadata": {
                "generated_at": datetime.now(ist).isoformat(),
                "language_code": code,
                "language_name": name,
                "timezone": "Asia/Kolkata",
                "reference_standard": "New Delhi / Central India (IST)",
                "ayanamsha": "Lahiri / Chitrapaksha",
                "coverage": {
                    "start_date": start_date_str,
                    "end_date": end_date_str
                }
            },
            "days": {}
        }
        for code, name in LANGUAGES.items()
    }

    # --- STEP 1: Generate Base English Week (7 requests) ---
    print(">>> STAGE 1: Generating Base English Forecast (7 Days)...")
    for offset in range(7):
        target_date = today + timedelta(days=offset)
        date_str = target_date.strftime("%Y-%m-%d")
        print(f" [+] Generating English: {date_str} ({target_date.strftime('%A')})...")

        day_payload = generate_base_day(target_date)
        datasets["en"]["days"][date_str] = day_payload
        
        # Pacing between calls
        time.sleep(REQUEST_SPACING_SECONDS)

    print("\n>>> English Generation Complete. Base data is secured.\n")

    # --- STEP 2: Translate across 7 Indic Languages (49 requests) ---
    print(">>> STAGE 2: Translating across 7 Indic Languages...")
    indic_langs = {c: n for c, n in LANGUAGES.items() if c != "en"}

    for date_str, english_day in datasets["en"]["days"].items():
        print(f"\n--- Localizing Date: {date_str} ---")
        for code, name in indic_langs.items():
            print(f" -> Localizing to {name} ({code})...")
            localized_day = translate_day(english_day, lang_code=code, lang_name=name)
            datasets[code]["days"][date_str] = localized_day

            # Pacing between calls
            time.sleep(REQUEST_SPACING_SECONDS)

    # --- STEP 3: Save JSON Outputs ---
    print("\n>>> STAGE 3: Saving all language files...")
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    for code, dataset in datasets.items():
        output_file = output_dir / f"panchang-rashifal-week-{code}.json"
        output_file.write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f" [✓] Saved {LANGUAGES[code]} -> {output_file}")

    print("\n============================================================")
    print(" All 8 languages successfully generated and saved without errors!")
    print("============================================================")

if __name__ == "__main__":
    main()
