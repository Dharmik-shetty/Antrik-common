"""
generate_today.py
One-time initialization script to generate authentic Indian Panchanga and
Vedic Rashifal for TODAY across all 8 languages using Gemini 3.8 Flash.

Features:
- Targets strictly TODAY (8 API calls total: 1 English base + 7 regional translations).
- Safe within the 20 Requests-Per-Day (RPD) free-tier limit.
- 100% Native Script Localization: Translates all Panchang technical terms (Tithi name,
  Paksha, Nakshatra, Ruler, Yoga, Nature, Karana, Sanskrit Day, Moon Sign, Month)
  directly into the regional Indic scripts (Kannada, Hindi, Bengali, Tamil, Telugu, etc.).
- True State Persistence: Saves each language to disk immediately upon completion.
- Exact Resumption: Skips any language already present on disk (0 calls wasted).
"""

import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from google import genai
from google.genai import types

# Ensure UTF-8 output across all operating systems
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------
# 1. Configuration & Client Setup
# ---------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable is missing.", file=sys.stderr)
    print("Please set it via: export GEMINI_API_KEY='your-key' or $env:GEMINI_API_KEY='your-key'", file=sys.stderr)
    sys.exit(1)

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

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
                    "required": ["vikram_samvat", "shaka_samvat", "amanta_month", "purnimanta_month", "ayana", "ritu"]
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
                    "required": ["sunrise", "sunset", "rahu_kaal", "abhijit_muhurat", "brahma_muhurta"]
                },
                "moon_sign_today": {"type": "STRING"},
                "festival_or_vrat": {"type": "STRING"}
            },
            "required": ["tithi", "nakshatra", "yoga", "karana", "calendar", "timings_ist", "moon_sign_today", "festival_or_vrat"]
        },
        "rashifal": {
            "type": "OBJECT",
            "properties": {
                rashi: {
                    "type": "OBJECT",
                    "properties": {
                        "summary": {"type": "STRING"},
                        "moon_sign_position": {"type": "STRING"},
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
        "day_of_week_sanskrit": {"type": "STRING"},
        "moon_sign_today": {"type": "STRING"},
        "panchang": {
            "type": "OBJECT",
            "properties": {
                "tithi_name": {"type": "STRING"},
                "tithi_paksha": {"type": "STRING"},
                "tithi_description": {"type": "STRING"},
                "nakshatra_name": {"type": "STRING"},
                "nakshatra_ruler": {"type": "STRING"},
                "yoga_name": {"type": "STRING"},
                "yoga_nature": {"type": "STRING"},
                "karana_name": {"type": "STRING"},
                "month_name": {"type": "STRING"},
                "festival_or_vrat": {"type": "STRING"}
            },
            "required": [
                "tithi_name", "tithi_paksha", "tithi_description",
                "nakshatra_name", "nakshatra_ruler", "yoga_name",
                "yoga_nature", "karana_name", "month_name", "festival_or_vrat"
            ]
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
    "required": ["day_of_week_sanskrit", "moon_sign_today", "panchang", "rashifal"]
}

# ---------------------------------------------------------
# 3. API Invocation with Patient Backoff
# ---------------------------------------------------------
BACKOFF_DELAYS = [15, 30, 45]

def call_gemini_with_retry(prompt: str, schema: dict, task_name: str, max_retries: int = 3) -> dict:
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
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                print(f"\n[!] Daily Gemini Free-Tier Quota Limit Reached (429 RESOURCE_EXHAUSTED).")
                print(f"    Google will reset the quota at 00:00 UTC (05:30 AM IST).")
                print(f"    All languages generated so far are safely saved to disk.")
                raise SystemExit(0)

            delay_idx = min(attempt, len(BACKOFF_DELAYS) - 1)
            wait_time = BACKOFF_DELAYS[delay_idx]
            print(f"   [!] {task_name} attempt {attempt + 1}/{max_retries} encountered: {e}")
            if attempt < max_retries - 1:
                print(f"       Waiting {wait_time}s for Google server capacity spike to clear...")
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
    p = day_data["panchang"]
    
    translatable_payload = {
        "day_of_week_sanskrit": day_data.get("day_of_week_sanskrit", ""),
        "moon_sign_today": p.get("moon_sign_today", ""),
        "panchang": {
            "tithi_name": p["tithi"].get("name", ""),
            "tithi_paksha": p["tithi"].get("paksha", ""),
            "tithi_description": p["tithi"].get("description", ""),
            "nakshatra_name": p["nakshatra"].get("name", ""),
            "nakshatra_ruler": p["nakshatra"].get("ruler", ""),
            "yoga_name": p["yoga"].get("name", ""),
            "yoga_nature": p["yoga"].get("nature", ""),
            "karana_name": p["karana"].get("name", ""),
            "month_name": p["calendar"].get("amanta_month", ""),
            "festival_or_vrat": p.get("festival_or_vrat", "")
        },
        "rashifal": {
            rashi: {
                "summary": day_data["rashifal"][rashi].get("summary", ""),
                "general": day_data["rashifal"][rashi].get("general", ""),
                "career": day_data["rashifal"][rashi].get("career", ""),
                "finance": day_data["rashifal"][rashi].get("finance", ""),
                "relationships": day_data["rashifal"][rashi].get("relationships", ""),
                "health": day_data["rashifal"][rashi].get("health", ""),
                "lucky_color": day_data["rashifal"][rashi].get("lucky_color", ""),
                "auspicious_time": day_data["rashifal"][rashi].get("auspicious_time", ""),
                "remedy": day_data["rashifal"][rashi].get("remedy", "")
            }
            for rashi in RASHIS
        }
    }

    prompt = f"""You are an expert Vedic astrologer and classical linguist fluent in {lang_name} ({lang_code}).
Translate and localize the astrological text below into authentic, elegant, natural {lang_name}.

CRITICAL RULES:
1. Translate EVERYTHING entirely into the native {lang_name} script. Do NOT leave English words or English alphabets.
2. Translate all technical Vedic terms directly into {lang_name} script:
   - tithi_name (e.g., 'ದ್ವಾದಶಿ' in Kannada, 'দ্বাদশী' in Bengali, 'द्वादशी' in Hindi)
   - tithi_paksha (e.g., 'ಶುಕ್ಲ ಪಕ್ಷ' in Kannada, 'শুক্ল পক্ষ' in Bengali, 'शुक्ल पक्ष' in Hindi)
   - nakshatra_name (e.g., 'ಪೂರ್ವಭಾದ್ರಪದ' in Kannada, 'পূর্বভাদ্রপদ' in Bengali, 'पूर्वाभाद्रपद' in Hindi)
   - nakshatra_ruler (e.g., 'ಗುರು' in Kannada, 'বৃহস্পতি' in Bengali, 'बृहस्पति' in Hindi)
   - yoga_name & nature (e.g., 'ಧ್ರುವ' / 'ಶುಭ' in Kannada, 'ধ್ರುವ' / 'শুভ' in Bengali)
   - karana_name (e.g., 'ಬಾಲವ' in Kannada, 'বালব' in Bengali)
   - month_name (e.g., 'ಆಶ್ವಯುಜ' in Kannada, 'আশ্বিন' in Bengali)
   - day_of_week_sanskrit (e.g., 'ಭೌಮವಾರ' in Kannada, 'ভৌমবার' in Bengali, 'भौमवार' in Hindi)
   - moon_sign_today (e.g., 'ಮಕರ' in Kannada, 'মকর' in Bengali, 'मकर' in Hindi)
3. Tone must be uplifting, respectful, and astrologically precise.
4. Keep the translation strictly faithful to the original predictions and remedies.

Source JSON:
{json.dumps(translatable_payload, ensure_ascii=False)}
"""

    translated_fields = call_gemini_with_retry(
        prompt,
        TRANSLATION_SCHEMA,
        task_name=f"Translate {date_str} to {lang_name}"
    )

    localized_day = copy.deepcopy(day_data)
    
    # 1. Update Day and Moon Sign in localized script
    localized_day["day_of_week_sanskrit"] = translated_fields["day_of_week_sanskrit"]
    localized_day["panchang"]["moon_sign_today"] = translated_fields["moon_sign_today"]

    # 2. Update Panchang elements in localized script
    trans_p = translated_fields["panchang"]
    localized_day["panchang"]["tithi"]["name"] = trans_p["tithi_name"]
    localized_day["panchang"]["tithi"]["paksha"] = trans_p["tithi_paksha"]
    localized_day["panchang"]["tithi"]["description"] = trans_p["tithi_description"]
    localized_day["panchang"]["nakshatra"]["name"] = trans_p["nakshatra_name"]
    localized_day["panchang"]["nakshatra"]["ruler"] = trans_p["nakshatra_ruler"]
    localized_day["panchang"]["yoga"]["name"] = trans_p["yoga_name"]
    localized_day["panchang"]["yoga"]["nature"] = trans_p["yoga_nature"]
    localized_day["panchang"]["karana"]["name"] = trans_p["karana_name"]
    localized_day["panchang"]["calendar"]["amanta_month"] = trans_p["month_name"]
    localized_day["panchang"]["calendar"]["purnimanta_month"] = trans_p["month_name"]
    localized_day["panchang"]["festival_or_vrat"] = trans_p["festival_or_vrat"]

    # 3. Update Rashifal text fields
    for rashi in RASHIS:
        for key in ["summary", "general", "career", "finance", "relationships", "health", "lucky_color", "auspicious_time", "remedy"]:
            localized_day["rashifal"][rashi][key] = translated_fields["rashifal"][rashi][key]

    return localized_day

# ---------------------------------------------------------
# 4. Storage & Incremental Saving
# ---------------------------------------------------------
def save_dataset_file(output_dir: Path, code: str, dataset: dict):
    """Saves each language immediately to disk so progress is never lost."""
    ist = ZoneInfo("Asia/Kolkata")
    dataset["metadata"]["generated_at"] = datetime.now(ist).isoformat()
    if dataset["days"]:
        dataset["metadata"]["coverage"] = {
            "start_date": min(dataset["days"].keys()),
            "end_date": max(dataset["days"].keys())
        }

    lang_file = output_dir / f"panchang-rashifal-week-{code}.json"
    lang_file.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    if code == "en":
        master_file = output_dir / "panchang-rashifal-week.json"
        master_file.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

        # Always guarantee parent directory exists before writing Android asset
        asset_file = Path("app/src/main/assets/panchang-rashifal-week.json")
        try:
            asset_file.parent.mkdir(parents=True, exist_ok=True)
            asset_file.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"     [OK] Synchronized Android asset: {asset_file}")
        except Exception as e:
            print(f"     [!] Notice: Could not sync Android asset ({e})")

def load_existing_datasets(output_dir: Path) -> dict:
    datasets = {}

    for code, name in LANGUAGES.items():
        file_path = output_dir / f"panchang-rashifal-week-{code}.json"
        if code == "en" and not file_path.exists():
            alt_path = output_dir / "panchang-rashifal-week.json"
            if alt_path.exists():
                file_path = alt_path

        if file_path.exists():
            try:
                raw_text = file_path.read_text(encoding="utf-8").strip()
                if raw_text:
                    data = json.loads(raw_text)
                    datasets[code] = data
                    continue
            except Exception as e:
                print(f"Notice: Initializing fresh state for {file_path.name} ({e})")

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
# 5. Main Pipeline for TODAY ONLY
# ---------------------------------------------------------
def main():
    ist = ZoneInfo("Asia/Kolkata")
    today = datetime.now(ist).date()
    date_str = today.strftime("%Y-%m-%d")

    print(f"============================================================")
    print(f" One-Time Panchang & Rashifal Generator (TODAY ONLY)")
    print(f" Generating Date (IST): {date_str} ({today.strftime('%A')})")
    print(f" Target Languages: {', '.join(LANGUAGES.values())} (8 total)")
    print(f" Free-Tier Safe: Exactly 8 calls (Leaves 12-call buffer under 20 RPD cap)")
    print(f"============================================================\n")

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_existing_datasets(output_dir)

    # Check if TODAY is already 100% complete across all 8 languages
    is_fully_covered = all(date_str in datasets[code]["days"] for code in LANGUAGES.keys())
    if is_fully_covered:
        print(f" [OK] Today ({date_str}) is already 100% complete across all 8 languages.")
        for code in LANGUAGES.keys():
            save_dataset_file(output_dir, code, datasets[code])
        print(" [OK] Datasets verified. 0 API calls required.")
        return

    print(f"Generating Today: {date_str} ({today.strftime('%A')})")
    indic_langs = {c: n for c, n in LANGUAGES.items() if c != "en"}

    # 1. Base English Generation (Skip if already present)
    if date_str not in datasets["en"]["days"]:
        print(f" [+] Generating Base English for today {date_str}...")
        english_day = generate_base_day(today)
        datasets["en"]["days"][date_str] = english_day
        save_dataset_file(output_dir, "en", datasets["en"])
        print(f"     [OK] English saved to disk.")
        time.sleep(REQUEST_SPACING_SECONDS)
    else:
        print(f"     [OK] English already present on disk for {date_str} (0 API calls).")
        english_day = datasets["en"]["days"][date_str]

    # 2. Translate into 7 Indic Languages (Skip any language already on disk)
    for code, name in indic_langs.items():
        if date_str in datasets[code]["days"]:
            print(f"     [OK] {name} ({code}) already present on disk (0 API calls).")
            continue

        print(f" -> Localizing {date_str} to {name} ({code})...")
        localized_day = translate_day(english_day, lang_code=code, lang_name=name)
        datasets[code]["days"][date_str] = localized_day
        
        save_dataset_file(output_dir, code, datasets[code])
        print(f"     [OK] {name} saved to disk.")
        time.sleep(REQUEST_SPACING_SECONDS)

    print("\n============================================================")
    print(f" Success! Today ({date_str}) is 100% generated across all 8 languages.")
    print("============================================================")

if __name__ == "__main__":
    main()
