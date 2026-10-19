"""
generate_week.py
Maintains a rolling forecast of authentic Indian Panchanga and Vedic Rashifal
in 8 languages using Gemini 3.8 Flash.

Quota Safety & Rolling Window:
- Cold start / First run: Generates Today & Tomorrow (16 requests <= 20 RPD free tier limit).
- Daily runs: Generates Tomorrow's advance forecast (8 requests <= 20 RPD free tier limit).
- All astrological facts, lucky numbers, ratings, and timings are 100% synchronized.
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

# Spacing between Gemini calls to stay comfortably under 15 RPM
REQUEST_SPACING_SECONDS = 5.0

# ---------------------------------------------------------
# 2. Schemas
# ---------------------------------------------------------
DAY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "date": {"type": "STRING"},
        "day_of_week": {"type": "STRING"},
        "day_of_week_sanskrit": {"type": "STRING"},
        "panchang": {
            "type": "OBJECT",
            "properties": {
                "tithi": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "paksha": {"type": "STRING"},
                        "description": {"type": "STRING"}
                    },
                    "required": ["name", "paksha", "description"]
                },
                "nakshatra": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "ruler": {"type": "STRING"},
                        "pada": {"type": "INTEGER"},
                        "deity": {"type": "STRING"}
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
                        "abhijit_muhurat": {"type": "STRING"},
                        "brahma_muhurta": {"type": "STRING"}
                    },
                    "required": ["sunrise", "sunset", "rahu_kaal", "abhijit_muhurat"]
                },
                "moon_sign_today": {"type": "STRING"},
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
                        "chandra_house": {"type": "INTEGER"},
                        "transit_effect": {"type": "STRING"},
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
# 3. API Invocation with Exponential Backoff
# ---------------------------------------------------------
def call_gemini_with_retry(prompt: str, schema: dict, task_name: str, max_retries: int = 4) -> dict:
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
   - Provide standard IST Rahu Kaal, Abhijit Muhurat, Brahma Muhurta, Sunrise, and Sunset.
   - Note any Vrat or Hindu festival (or 'Regular day' if none).
2. Rashifal (All 12 Rashis):
   - Determine predictions based on the Moon's transit (Chandra Gochar).
   - Write insightful, non-fatalistic, inspiring predictions.
   - Provide lucky number (1-9), lucky color, best time window, ratings (1-5), and a simple Vedic remedy (Upay).

Ensure all 12 Rashis ({', '.join(RASHIS)}) are populated with high quality.
"""
    return call_gemini_with_retry(prompt, DAY_SCHEMA, task_name=f"Base Gen {date_str}")

def translate_day(day_data: dict, lang_code: str, lang_name: str) -> dict:
    date_str = day_data["date"]
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

    localized_day = copy.deepcopy(day_data)
    localized_day["panchang"]["tithi"]["description"] = translated_fields["panchang"]["tithi_description"]
    localized_day["panchang"]["festival_or_vrat"] = translated_fields["panchang"]["festival_or_vrat"]

    for rashi in RASHIS:
        for key in ["summary", "general", "career", "finance", "relationships", "health", "lucky_color", "auspicious_time", "remedy"]:
            localized_day["rashifal"][rashi][key] = translated_fields["rashifal"][rashi][key]

    return localized_day

# ---------------------------------------------------------
# 4. Storage & Pruning
# ---------------------------------------------------------
def load_existing_datasets(output_dir: Path, today: datetime.date) -> dict:
    datasets = {}
    cutoff_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    for code, name in LANGUAGES.items():
        file_path = output_dir / f"panchang-rashifal-week-{code}.json"
        if code == "en" and not file_path.exists():
            alt_path = output_dir / "panchang-rashifal-week.json"
            if alt_path.exists():
                file_path = alt_path

        if file_path.exists():
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                valid_days = {
                    d: v for d, v in data.get("days", {}).items()
                    if d >= cutoff_date
                }
                data["days"] = valid_days
                datasets[code] = data
                continue
            except Exception as e:
                print(f"Warning: Failed to parse {file_path}: {e}")

        datasets[code] = {
            "metadata": {
                "generated_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                "language_code": code,
                "language_name": name,
                "timezone": "Asia/Kolkata",
                "reference_standard": "New Delhi / Central India (IST)",
                "ayanamsha": "Lahiri / Chitrapaksha"
            },
            "days": {}
        }

    return datasets

# ---------------------------------------------------------
# 5. Main Pipeline
# ---------------------------------------------------------
def main():
    ist = ZoneInfo("Asia/Kolkata")
    today = datetime.now(ist).date()
    tomorrow = today + timedelta(days=1)

    print(f"============================================================")
    print(f" Rolling Daily Panchang & Rashifal Generator (Gemini Flash) ")
    print(f" Current Date (IST): {today.isoformat()} | Advance Target: {tomorrow.isoformat()}")
    print(f" Target Languages: {', '.join(LANGUAGES.values())} (8 total)")
    print(f"============================================================\n")

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_existing_datasets(output_dir, today)
    en_days = datasets["en"]["days"]

    target_dates = []
    if today.strftime("%Y-%m-%d") not in en_days:
        target_dates.append(today)
    if tomorrow.strftime("%Y-%m-%d") not in en_days:
        target_dates.append(tomorrow)

    if not target_dates:
        print(f" [✓] Today ({today}) and Tomorrow ({tomorrow}) already exist.")
        day_plus_2 = today + timedelta(days=2)
        if day_plus_2.strftime("%Y-%m-%d") not in en_days:
            target_dates.append(day_plus_2)

    target_dates = target_dates[:2]

    print(f"Dates to generate this run ({len(target_dates)} day(s)): {[d.isoformat() for d in target_dates]}")
    indic_langs = {c: n for c, n in LANGUAGES.items() if c != "en"}

    for target_date in target_dates:
        date_str = target_date.strftime("%Y-%m-%d")
        print(f"\n>>> Processing Date: {date_str} ({target_date.strftime('%A')})")

        print(f" [+] Generating Base English for {date_str}...")
        english_day = generate_base_day(target_date)
        datasets["en"]["days"][date_str] = english_day
        time.sleep(REQUEST_SPACING_SECONDS)

        for code, name in indic_langs.items():
            print(f" -> Localizing {date_str} to {name} ({code})...")
            localized_day = translate_day(english_day, lang_code=code, lang_name=name)
            datasets[code]["days"][date_str] = localized_day
            time.sleep(REQUEST_SPACING_SECONDS)

    print("\n>>> Saving updated datasets...")
    for code, dataset in datasets.items():
        dataset["metadata"]["generated_at"] = datetime.now(ist).isoformat()
        dataset["metadata"]["coverage"] = {
            "start_date": min(dataset["days"].keys()) if dataset["days"] else today.isoformat(),
            "end_date": max(dataset["days"].keys()) if dataset["days"] else today.isoformat()
        }

        lang_file = output_dir / f"panchang-rashifal-week-{code}.json"
        lang_file.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f" [✓] Saved {LANGUAGES[code]} ({len(dataset['days'])} days) -> {lang_file}")

    master_file = output_dir / "panchang-rashifal-week.json"
    master_file.write_text(json.dumps(datasets["en"], ensure_ascii=False, indent=2), encoding="utf-8")

    asset_file = Path("app/src/main/assets/panchang-rashifal-week.json")
    if asset_file.parent.exists():
        asset_file.write_text(json.dumps(datasets["en"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f" [✓] Synchronized Android asset: {asset_file}")

    print("\n============================================================")
    print(" Successfully updated rolling forecast for all 8 languages!")
    print("============================================================")

if __name__ == "__main__":
    main()
