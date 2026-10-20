"""
generate_today.py
One-time initialization script to generate authentic Indian Panchanga and
Vedic Rashifal for TODAY across all 8 languages using Meta Llama 3.3 70B on Groq Cloud.

Features:
- Targets strictly TODAY (8 API calls total: 1 English base + 7 regional translations).
- 100% Native Script Localization: Translates all Panchang technical terms (Tithi name,
  Paksha, Nakshatra, Ruler, Yoga, Nature, Karana, Sanskrit Day, Moon Sign, Month)
  directly into the regional Indic scripts (Kannada, Hindi, Bengali, Tamil, Telugu, etc.).
- True State Persistence: Saves each language to disk immediately upon completion.
- Exact Resumption: Skips any language already present on disk (0 calls wasted).
- Zero Demand Spikes: Powered by Groq's high-speed LPU infrastructure.
"""

import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from groq import Groq

# Ensure UTF-8 output across all operating systems
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------
# 1. Configuration & Client Setup
# ---------------------------------------------------------
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY environment variable is missing.", file=sys.stderr)
    print("Please set it via: export GROQ_API_KEY='your-key' or $env:GROQ_API_KEY='your-key'", file=sys.stderr)
    sys.exit(1)

client = Groq(api_key=api_key)
MODEL_NAME = "llama-3.3-70b-versatile"

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

REQUEST_SPACING_SECONDS = 3.0

# ---------------------------------------------------------
# 2. API Invocation with Retry
# ---------------------------------------------------------
def call_groq_with_retry(system_prompt: str, user_prompt: str, task_name: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            raw_text = response.choices[0].message.content
            return json.loads(raw_text)
        except Exception as e:
            err_msg = str(e)
            if "rate_limit_exceeded" in err_msg or "429" in err_msg:
                print(f"   [!] Rate limit reached for {task_name}. Waiting 25s...")
                time.sleep(25)
                continue

            delay = 5 * (attempt + 1)
            print(f"   [!] {task_name} attempt {attempt + 1}/{max_retries} encountered: {e}")
            if attempt < max_retries - 1:
                print(f"       Waiting {delay}s before retrying...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed {task_name} after {max_retries} attempts.")

def generate_base_day(target_date: datetime.date) -> dict:
    date_str = target_date.strftime("%Y-%m-%d")
    day_name = target_date.strftime("%A")

    system_prompt = (
        "You are an expert Vedic Astrologer (Jyotish Acharya) using Nirayana Chitrapaksha (Lahiri) Ayanamsha. "
        "You always respond with valid, strictly parseable JSON adhering to the exact required schema."
    )

    user_prompt = f"""Generate authentic Drik Panchanga and 12-Rashi Rashifal for:
Date: {date_str} ({day_name})
Reference: Indian Standard Time (IST, UTC+05:30), New Delhi anchor.

Astrological Requirements:
1. Panchanga:
   - Calculate exact Tithi, Nakshatra, Yoga, and Karana in IST.
   - Include both Amanta and Purnimanta lunar months.
   - Standard IST Rahu Kaal, Abhijit Muhurat, Brahma Muhurta, Sunrise, and Sunset.
   - Note any Vrat or Hindu festival (or 'Regular day' if none).
2. Rashifal (ALL 12 Rashis: {', '.join(RASHIS)}):
   - Provide predictions based on Moon's transit (Chandra Gochar).
   - Write insightful, non-fatalistic, inspiring predictions.
   - Provide lucky number (1-9), lucky color, best time window, ratings (1-5), and a simple Vedic remedy (Upay).

You MUST output ONLY a valid JSON object with this exact structure:
{{
  "date": "{date_str}",
  "day_of_week": "{day_name}",
  "day_of_week_sanskrit": "Sanskrit weekday name (e.g. Ravivara, Somavara, Bhaumavara, Budhavara, Guruvara, Shukravara, Shanivara)",
  "panchang": {{
    "tithi": {{
      "name": "Tithi name",
      "paksha": "Shukla or Krishna",
      "description": "Spiritual significance of this tithi"
    }},
    "nakshatra": {{
      "name": "Nakshatra name",
      "ruler": "Ruling Graha",
      "pada": 1,
      "deity": "Presiding deity"
    }},
    "yoga": {{
      "name": "Yoga name",
      "nature": "Benefic or Malefic"
    }},
    "karana": {{
      "name": "Karana name"
    }},
    "calendar": {{
      "vikram_samvat": 2083,
      "shaka_samvat": 1948,
      "amanta_month": "Month name",
      "purnimanta_month": "Month name",
      "ayana": "Uttarayana or Dakshinayana",
      "ritu": "Current Vedic Season"
    }},
    "timings_ist": {{
      "sunrise": "06:25 AM",
      "sunset": "05:46 PM",
      "rahu_kaal": "Standard IST Rahu Kaal interval",
      "abhijit_muhurat": "Standard IST Abhijit Muhurat interval",
      "brahma_muhurta": "Standard IST Brahma Muhurta interval"
    }},
    "moon_sign_today": "Current transit Moon sign",
    "festival_or_vrat": "Major Hindu festival or Vrat if any, otherwise 'Regular day'"
  }},
  "rashifal": {{
    "Mesha": {{
      "summary": "1-sentence punchy forecast",
      "moon_sign_position": "Transit position",
      "chandra_house": 1,
      "transit_effect": "Positive / Neutral / Challenging",
      "general": "2-sentence overview",
      "career": "2-sentence career prediction",
      "finance": "2-sentence finance prediction",
      "relationships": "2-sentence relationship advice",
      "health": "2-sentence health guidance",
      "lucky_number": 7,
      "lucky_color": "Lucky color",
      "auspicious_time": "Time window",
      "remedy": "Simple Vedic Upay",
      "ratings": {{"career": 4, "finance": 4, "love": 4, "health": 4}}
    }},
    ... MUST include all 12 Rashis: {', '.join(RASHIS)}
  }}
}}
"""
    return call_groq_with_retry(system_prompt, user_prompt, task_name=f"Base Gen {date_str}")

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
                "summary": day_data["rashifal"].get(rashi, {}).get("summary", ""),
                "general": day_data["rashifal"].get(rashi, {}).get("general", ""),
                "career": day_data["rashifal"].get(rashi, {}).get("career", ""),
                "finance": day_data["rashifal"].get(rashi, {}).get("finance", ""),
                "relationships": day_data["rashifal"].get(rashi, {}).get("relationships", ""),
                "health": day_data["rashifal"].get(rashi, {}).get("health", ""),
                "lucky_color": day_data["rashifal"].get(rashi, {}).get("lucky_color", ""),
                "auspicious_time": day_data["rashifal"].get(rashi, {}).get("auspicious_time", ""),
                "remedy": day_data["rashifal"].get(rashi, {}).get("remedy", "")
            }
            for rashi in RASHIS
        }
    }

    system_prompt = (
        f"You are a native linguist and classical Vedic astrologer fluent in {lang_name} ({lang_code}). "
        f"You strictly translate all astrological content into authentic, elegant {lang_name} script. "
        "You always respond with valid, parseable JSON matching the requested structure."
    )

    user_prompt = f"""Translate and localize the following astrological JSON into authentic, natural {lang_name}.

CRITICAL RULES:
1. Translate EVERYTHING entirely into the native {lang_name} script. Do NOT leave English words.
2. Translate all technical terms directly into {lang_name} script:
   - Tithi (name, paksha, description)
   - Nakshatra (name, ruler)
   - Yoga (name, nature)
   - Karana (name)
   - Month name
   - Sanskrit day name
   - Moon sign name
   - Festival or vrat
3. Keep the translation strictly faithful to original predictions, colors, and remedies.

Source JSON:
{json.dumps(translatable_payload, ensure_ascii=False)}

Output ONLY a JSON object with this exact structure:
{{
  "day_of_week_sanskrit": "Localized day name in {lang_name} script",
  "moon_sign_today": "Localized moon sign in {lang_name} script",
  "panchang": {{
    "tithi_name": "...",
    "tithi_paksha": "...",
    "tithi_description": "...",
    "nakshatra_name": "...",
    "nakshatra_ruler": "...",
    "yoga_name": "...",
    "yoga_nature": "...",
    "karana_name": "...",
    "month_name": "...",
    "festival_or_vrat": "..."
  }},
  "rashifal": {{
    "Mesha": {{"summary": "...", "general": "...", "career": "...", "finance": "...", "relationships": "...", "health": "...", "lucky_color": "...", "auspicious_time": "...", "remedy": "..."}},
    ... include all 12 rashis ({', '.join(RASHIS)})
  }}
}}
"""

    translated_fields = call_groq_with_retry(
        system_prompt,
        user_prompt,
        task_name=f"Translate {date_str} to {lang_name}"
    )

    localized_day = copy.deepcopy(day_data)
    
    # 1. Update Day and Moon Sign in localized script
    localized_day["day_of_week_sanskrit"] = translated_fields.get("day_of_week_sanskrit", localized_day.get("day_of_week_sanskrit", ""))
    localized_day["panchang"]["moon_sign_today"] = translated_fields.get("moon_sign_today", localized_day["panchang"].get("moon_sign_today", ""))

    # 2. Update Panchang elements in localized script
    trans_p = translated_fields.get("panchang", {})
    localized_day["panchang"]["tithi"]["name"] = trans_p.get("tithi_name", localized_day["panchang"]["tithi"]["name"])
    localized_day["panchang"]["tithi"]["paksha"] = trans_p.get("tithi_paksha", localized_day["panchang"]["tithi"]["paksha"])
    localized_day["panchang"]["tithi"]["description"] = trans_p.get("tithi_description", localized_day["panchang"]["tithi"]["description"])
    localized_day["panchang"]["nakshatra"]["name"] = trans_p.get("nakshatra_name", localized_day["panchang"]["nakshatra"]["name"])
    localized_day["panchang"]["nakshatra"]["ruler"] = trans_p.get("nakshatra_ruler", localized_day["panchang"]["nakshatra"]["ruler"])
    localized_day["panchang"]["yoga"]["name"] = trans_p.get("yoga_name", localized_day["panchang"]["yoga"]["name"])
    localized_day["panchang"]["yoga"]["nature"] = trans_p.get("yoga_nature", localized_day["panchang"]["yoga"]["nature"])
    localized_day["panchang"]["karana"]["name"] = trans_p.get("karana_name", localized_day["panchang"]["karana"]["name"])
    localized_day["panchang"]["calendar"]["amanta_month"] = trans_p.get("month_name", localized_day["panchang"]["calendar"]["amanta_month"])
    localized_day["panchang"]["calendar"]["purnimanta_month"] = trans_p.get("month_name", localized_day["panchang"]["calendar"]["purnimanta_month"])
    localized_day["panchang"]["festival_or_vrat"] = trans_p.get("festival_or_vrat", localized_day["panchang"]["festival_or_vrat"])

    # 3. Update Rashifal text fields safely
    trans_rashifal = translated_fields.get("rashifal", {})
    for rashi in RASHIS:
        if rashi in trans_rashifal and isinstance(trans_rashifal[rashi], dict):
            for key in ["summary", "general", "career", "finance", "relationships", "health", "lucky_color", "auspicious_time", "remedy"]:
                if key in trans_rashifal[rashi]:
                    localized_day["rashifal"][rashi][key] = trans_rashifal[rashi][key]

    return localized_day

# ---------------------------------------------------------
# 3. Storage & Incremental Saving
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
# 4. Main Pipeline for TODAY ONLY
# ---------------------------------------------------------
def main():
    ist = ZoneInfo("Asia/Kolkata")
    today = datetime.now(ist).date()
    date_str = today.strftime("%Y-%m-%d")

    print(f"============================================================")
    print(f" One-Time Panchang & Rashifal Generator (Groq Llama 3.3 70B)")
    print(f" Generating Date (IST): {date_str} ({today.strftime('%A')})")
    print(f" Target Languages: {', '.join(LANGUAGES.values())} (8 total)")
    print(f" Speed & Reliability: Groq LPUs (Zero 503 capacity outages)")
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
