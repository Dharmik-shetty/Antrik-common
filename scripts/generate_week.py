"""
generate_today.py
Hybrid Architecture:
- Base Jyotish Generation (Panchanga & 12-Rashi Bhavisya): Google Gemini 3.8 Flash exclusively
- Regional Translation & Indic Script Localization: Groq GPT-OSS 120B (with GPT-OSS 20B & Gemini 3.8 Flash fallbacks)
- Resilient Demand Management: Progressive backoff, jitter, and multi-attempt retries for rate limits and server demand spikes.
Translates all technical Panchang terms directly into native Indic scripts.
"""

import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types
from groq import Groq

# Real-time stdout buffering so GitHub Actions displays progress live
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# ---------------------------------------------------------
# 1. Configuration & Client Setup
# ---------------------------------------------------------
google_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable is missing.", file=sys.stderr, flush=True)
    sys.exit(1)

if not groq_api_key:
    print("Error: GROQ_API_KEY environment variable is missing.", file=sys.stderr, flush=True)
    sys.exit(1)

gemini_client = genai.Client(api_key=google_api_key)
groq_client = Groq(api_key=groq_api_key)

# Strictly Gemini 3.8 Flash only (no fallback to 2.5 or 2.0)
GEMINI_MODEL = "gemini-3.8-flash"

# Translation models on Groq
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

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

RASHI_ALIASES = {
    "mesha": "Mesha", "mesh": "Mesha", "aries": "Mesha",
    "vrishabha": "Vrishabha", "vrishabh": "Vrishabha", "vrisha": "Vrishabha", "taurus": "Vrishabha",
    "mithuna": "Mithuna", "mithun": "Mithuna", "gemini": "Mithuna",
    "karka": "Karka", "kark": "Karka", "cancer": "Karka",
    "simha": "Simha", "singh": "Simha", "singha": "Simha", "leo": "Simha",
    "kanya": "Kanya", "virgo": "Kanya",
    "tula": "Tula", "libra": "Tula",
    "vrischika": "Vrischika", "vrischik": "Vrischika", "vrishchika": "Vrischika", "scorpio": "Vrischika",
    "dhanu": "Dhanu", "sagittarius": "Dhanu",
    "makara": "Makara", "makar": "Makara", "capricorn": "Makara",
    "kumbha": "Kumbha", "kumbh": "Kumbha", "aquarius": "Kumbha",
    "meena": "Meena", "meen": "Meena", "pisces": "Meena"
}

REQUEST_SPACING_SECONDS = 2.5

def get_default_rashi(rashi_name: str) -> dict:
    return {
        "summary": f"Constructive and favorable day ahead for {rashi_name}.",
        "moon_sign_position": "Favorable transit position",
        "chandra_house": 1,
        "transit_effect": "Positive",
        "general": f"A balanced day with steady opportunities for {rashi_name}. Focus your energy on core priorities.",
        "career": "Professional efforts yield constructive progress. Maintain diligence and collaborative communication.",
        "finance": "Financial outlook remains stable. Thoughtful planning supports long-term prosperity.",
        "relationships": "Warmth and mutual understanding flourish in personal ties. Spend quality time with loved ones.",
        "health": "Vitality is steady. Practice mindfulness and maintain consistent daily hydration and rest.",
        "lucky_number": 7,
        "lucky_color": "Golden Yellow",
        "auspicious_time": "10:00 AM – 11:30 AM",
        "remedy": "Offer water to Surya Dev in the morning and chant Om Namah Shivaya.",
        "ratings": {"career": 4, "finance": 4, "love": 4, "health": 4}
    }

def clean_json_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)

# ---------------------------------------------------------
# 2. Google Gemini 3.8 Flash (High-Demand Tolerant Invocation)
# ---------------------------------------------------------
def call_gemini_with_retry(system_prompt: str, user_prompt: str, task_name: str, max_retries: int = 6) -> dict:
    """
    Strictly invokes Gemini 3.8 Flash with progressive backoff and jitter
    to handle demand spikes (503/500/overload) and rate limits (429).
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )
            raw_text = response.text
            if not raw_text:
                raise ValueError("Empty response text received from Gemini 3.8 Flash")
            return clean_json_response(raw_text)

        except Exception as e:
            last_error = e
            err_msg = str(e).lower()
            code = getattr(e, "code", None)

            is_rate_limit = "429" in err_msg or "resource_exhausted" in err_msg or code == 429
            is_overloaded = (
                "503" in err_msg
                or "overloaded" in err_msg
                or "high demand" in err_msg
                or "unavailable" in err_msg
                or "500" in err_msg
                or "internal" in err_msg
                or code in (500, 503, 504)
            )

            if attempt < max_retries:
                if is_rate_limit:
                    delay = 15.0 + (attempt * 8.0) + random.uniform(1.0, 3.0)
                    print(f"   [!] Gemini 3.8 Flash rate limit / quota hit (attempt {attempt}/{max_retries}).", flush=True)
                    print(f"       Cooling down for {delay:.1f}s before retrying {task_name}...", flush=True)
                    time.sleep(delay)
                elif is_overloaded:
                    delay = 10.0 + (attempt * 6.0) + random.uniform(1.0, 3.0)
                    print(f"   [!] Gemini 3.8 Flash high demand / server overload (attempt {attempt}/{max_retries}).", flush=True)
                    print(f"       Backing off for {delay:.1f}s before retrying {task_name}...", flush=True)
                    time.sleep(delay)
                else:
                    delay = 5.0 * attempt + random.uniform(1.0, 2.0)
                    print(f"   [!] {task_name} (Gemini 3.8 Flash) attempt {attempt}/{max_retries} encountered: {e}", flush=True)
                    print(f"       Retrying in {delay:.1f}s...", flush=True)
                    time.sleep(delay)
            else:
                print(f"   [!] All {max_retries} attempts exhausted for {task_name} on Gemini 3.8 Flash.", flush=True)

    raise RuntimeError(f"Gemini 3.8 Flash failed for {task_name} after {max_retries} attempts. Last error: {last_error}")

# ---------------------------------------------------------
# 3. Groq Invocation (Translation with Gemini 3.8 Flash Fallback)
# ---------------------------------------------------------
def call_groq_with_retry(system_prompt: str, user_prompt: str, task_name: str, max_retries: int = 3) -> dict:
    last_error = None
    for model_name in GROQ_MODELS:
        for attempt in range(1, max_retries + 1):
            try:
                call_kwargs = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                    "max_completion_tokens": 8192
                }
                if "gpt-oss" in model_name:
                    call_kwargs["reasoning_format"] = "hidden"
                    call_kwargs["reasoning_effort"] = "low"

                try:
                    response = groq_client.chat.completions.create(**call_kwargs)
                except Exception as inner_e:
                    err_str = str(inner_e).lower()
                    retry_needed = False
                    if "reasoning_effort" in err_str and "reasoning_effort" in call_kwargs:
                        del call_kwargs["reasoning_effort"]
                        retry_needed = True
                    if "reasoning_format" in err_str and "reasoning_format" in call_kwargs:
                        del call_kwargs["reasoning_format"]
                        retry_needed = True
                    if "max_completion_tokens" in err_str and "max_completion_tokens" in call_kwargs:
                        del call_kwargs["max_completion_tokens"]
                        call_kwargs["max_tokens"] = 8192
                        retry_needed = True

                    if retry_needed:
                        response = groq_client.chat.completions.create(**call_kwargs)
                    else:
                        raise inner_e

                raw_text = response.choices[0].message.content
                if not raw_text:
                    raise ValueError(f"Empty response from Groq {model_name}")

                return clean_json_response(raw_text)

            except Exception as e:
                last_error = e
                err_msg = str(e).lower()

                if "rate_limit_exceeded" in err_msg or "429" in err_msg:
                    delay = 15.0 + (attempt * 8.0) + random.uniform(1.0, 3.0)
                    print(f"   [!] Rate limit on Groq {model_name}. Waiting {delay:.1f}s (attempt {attempt}/{max_retries})...", flush=True)
                    time.sleep(delay)
                    continue

                if "model_not_found" in err_msg or "404" in err_msg:
                    print(f"   [!] Model {model_name} not available on Groq. Trying fallback...", flush=True)
                    break

                if "max completion tokens reached" in err_msg or "max_completion_tokens" in err_msg:
                    print(f"   [!] Token limit exhausted on Groq {model_name}. Trying fallback model...", flush=True)
                    break

                delay = 4.0 * attempt + random.uniform(1.0, 2.0)
                print(f"   [!] {task_name} (Groq {model_name}) attempt {attempt}/{max_retries} encountered: {e}", flush=True)
                if attempt < max_retries:
                    time.sleep(delay)

    # Seamless safety fallback: If Groq models fail, translate via Gemini 3.8 Flash!
    print(f"   [!] Groq models unavailable for {task_name}. Falling back to Gemini 3.8 Flash...", flush=True)
    return call_gemini_with_retry(system_prompt, user_prompt, task_name=f"{task_name} (Gemini 3.8 Flash fallback)")

def normalize_base_day(day_data: dict, target_date: datetime.date) -> dict:
    date_str = target_date.strftime("%Y-%m-%d")
    day_name = target_date.strftime("%A")

    if not isinstance(day_data, dict):
        day_data = {}

    day_data.setdefault("date", date_str)
    day_data.setdefault("day_of_week", day_name)
    day_data.setdefault("day_of_week_sanskrit", "Ravivara" if day_name == "Sunday" else f"{day_name}vara")

    p = day_data.setdefault("panchang", {})
    t = p.setdefault("tithi", {})
    t.setdefault("name", "Shukla Navami")
    t.setdefault("paksha", "Shukla")
    t.setdefault("description", "Auspicious day for spiritual practices and balanced action.")

    n = p.setdefault("nakshatra", {})
    n.setdefault("name", "Shravana")
    n.setdefault("ruler", "Moon")
    n.setdefault("pada", 1)
    n.setdefault("deity", "Vishnu")

    y = p.setdefault("yoga", {})
    y.setdefault("name", "Dhriti")
    y.setdefault("nature", "Benefic")

    k = p.setdefault("karana", {})
    k.setdefault("name", "Kaulava")

    c = p.setdefault("calendar", {})
    c.setdefault("vikram_samvat", 2083)
    c.setdefault("shaka_samvat", 1948)
    c.setdefault("amanta_month", "Ashwina")
    c.setdefault("purnimanta_month", "Ashwina")
    c.setdefault("ayana", "Dakshinayana")
    c.setdefault("ritu", "Sharad")

    timings = p.setdefault("timings_ist", {})
    timings.setdefault("sunrise", "06:25 AM")
    timings.setdefault("sunset", "05:46 PM")
    timings.setdefault("rahu_kaal", "03:00 PM – 04:30 PM")
    timings.setdefault("abhijit_muhurat", "11:45 AM – 12:30 PM")
    timings.setdefault("brahma_muhurta", "04:45 AM – 05:35 AM")

    p.setdefault("moon_sign_today", "Makara")
    p.setdefault("festival_or_vrat", "Regular day")

    raw_rashifal = day_data.get("rashifal", {})
    if not isinstance(raw_rashifal, dict):
        raw_rashifal = {}

    normalized_rashifal = {}
    for raw_k, raw_v in raw_rashifal.items():
        if isinstance(raw_v, dict):
            mapped_name = RASHI_ALIASES.get(str(raw_k).strip().lower(), str(raw_k).strip())
            normalized_rashifal[mapped_name] = raw_v

    final_rashifal = {}
    for rashi in RASHIS:
        default_item = get_default_rashi(rashi)
        if rashi in normalized_rashifal and isinstance(normalized_rashifal[rashi], dict):
            item = normalized_rashifal[rashi]
            merged_item = copy.deepcopy(default_item)
            for f in ["summary", "moon_sign_position", "transit_effect", "general", "career",
                      "finance", "relationships", "health", "lucky_color", "auspicious_time", "remedy"]:
                if f in item and item[f]:
                    merged_item[f] = str(item[f])
            if "lucky_number" in item:
                try:
                    merged_item["lucky_number"] = int(item["lucky_number"])
                except Exception:
                    pass
            if "chandra_house" in item:
                try:
                    merged_item["chandra_house"] = int(item["chandra_house"])
                except Exception:
                    pass
            if "ratings" in item and isinstance(item["ratings"], dict):
                merged_item["ratings"] = {
                    "career": int(item["ratings"].get("career", 4)),
                    "finance": int(item["ratings"].get("finance", 4)),
                    "love": int(item["ratings"].get("love", 4)),
                    "health": int(item["ratings"].get("health", 4))
                }
            final_rashifal[rashi] = merged_item
        else:
            final_rashifal[rashi] = default_item

    day_data["rashifal"] = final_rashifal
    return day_data

def generate_base_day(target_date: datetime.date) -> dict:
    date_str = target_date.strftime("%Y-%m-%d")
    day_name = target_date.strftime("%A")

    system_prompt = (
        "You are an expert Vedic Astrologer (Jyotish Acharya) using Nirayana Chitrapaksha (Lahiri) Ayanamsha. "
        "You always respond with valid, strictly parseable JSON adhering to the exact required schema."
    )

    rashis_schema_items = [
        f'    "{r}": {{\n'
        f'      "summary": "1-sentence punchy forecast for {r}",\n'
        f'      "moon_sign_position": "Transit position",\n'
        f'      "chandra_house": 1,\n'
        f'      "transit_effect": "Positive / Neutral / Challenging",\n'
        f'      "general": "2-sentence overview",\n'
        f'      "career": "2-sentence career prediction",\n'
        f'      "finance": "2-sentence finance prediction",\n'
        f'      "relationships": "2-sentence relationship advice",\n'
        f'      "health": "2-sentence health guidance",\n'
        f'      "lucky_number": 7,\n'
        f'      "lucky_color": "Lucky color",\n'
        f'      "auspicious_time": "Time window",\n'
        f'      "remedy": "Simple Vedic Upay",\n'
        f'      "ratings": {{"career": 4, "finance": 4, "love": 4, "health": 4}}\n'
        f'    }}'
        for r in RASHIS
    ]
    rashis_json_block = ",\n".join(rashis_schema_items)

    user_prompt = f"""Generate authentic Drik Panchanga and 12-Rashi Rashifal for:
Date: {date_str} ({day_name})
Reference: Indian Standard Time (IST, UTC+05:30), New Delhi anchor.

Astrological Requirements:
1. Panchanga:
   - Calculate exact Tithi, Nakshatra, Yoga, and Karana in IST.
   - Include both Amanta and Purnimanta lunar months.
   - Standard IST Rahu Kaal, Abhijit Muhurat, Brahma Muhurta, Sunrise, and Sunset.
   - Note any Vrat or Hindu festival (or 'Regular day' if none).
2. Rashifal (MUST include all 12 Rashis: {', '.join(RASHIS)}):
   - Provide predictions based on Moon's transit (Chandra Gochar).
   - Write insightful, non-fatalistic, inspiring predictions.
   - Provide lucky number (1-9), lucky color, best time window, ratings (1-5), and a simple Vedic remedy (Upay).

CRITICAL: Return ONLY valid JSON matching this schema:
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
{rashis_json_block}
  }}
}}
"""
    raw_day = call_gemini_with_retry(
        system_prompt,
        user_prompt,
        task_name=f"Base Gen {date_str}"
    )
    return normalize_base_day(raw_day, target_date)

def translate_day(day_data: dict, lang_code: str, lang_name: str) -> dict:
    date_str = day_data["date"]
    p = day_data.get("panchang", {})
    
    translatable_payload = {
        "day_of_week_sanskrit": day_data.get("day_of_week_sanskrit", ""),
        "moon_sign_today": p.get("moon_sign_today", ""),
        "panchang": {
            "tithi_name": p.get("tithi", {}).get("name", ""),
            "tithi_paksha": p.get("tithi", {}).get("paksha", ""),
            "tithi_description": p.get("tithi", {}).get("description", ""),
            "nakshatra_name": p.get("nakshatra", {}).get("name", ""),
            "nakshatra_ruler": p.get("nakshatra", {}).get("ruler", ""),
            "yoga_name": p.get("yoga", {}).get("name", ""),
            "yoga_nature": p.get("yoga", {}).get("nature", ""),
            "karana_name": p.get("karana", {}).get("name", ""),
            "month_name": p.get("calendar", {}).get("amanta_month", ""),
            "festival_or_vrat": p.get("festival_or_vrat", "")
        },
        "rashifal": {
            rashi: {
                "summary": day_data.get("rashifal", {}).get(rashi, {}).get("summary", ""),
                "general": day_data.get("rashifal", {}).get(rashi, {}).get("general", ""),
                "career": day_data.get("rashifal", {}).get(rashi, {}).get("career", ""),
                "finance": day_data.get("rashifal", {}).get(rashi, {}).get("finance", ""),
                "relationships": day_data.get("rashifal", {}).get(rashi, {}).get("relationships", ""),
                "health": day_data.get("rashifal", {}).get(rashi, {}).get("health", ""),
                "lucky_color": day_data.get("rashifal", {}).get(rashi, {}).get("lucky_color", ""),
                "auspicious_time": day_data.get("rashifal", {}).get(rashi, {}).get("auspicious_time", ""),
                "remedy": day_data.get("rashifal", {}).get(rashi, {}).get("remedy", "")
            }
            for rashi in RASHIS
        }
    }

    system_prompt = (
        f"You are a native linguist and classical Vedic astrologer fluent in {lang_name} ({lang_code}). "
        f"You strictly translate all astrological content into authentic, elegant {lang_name} script. "
        "You always respond with valid, strictly parseable JSON adhering to the exact schema."
    )

    schema_rashifal_items = ",\n".join(
        f'    "{r}": {{"summary": "...", "general": "...", "career": "...", "finance": "...", "relationships": "...", "health": "...", "lucky_color": "...", "auspicious_time": "...", "remedy": "..."}}'
        for r in RASHIS
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
4. Output ONLY valid JSON matching this schema:

Source JSON:
{json.dumps(translatable_payload, ensure_ascii=False)}

Output Schema:
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
{schema_rashifal_items}
  }}
}}
"""

    translated_fields = call_groq_with_retry(
        system_prompt,
        user_prompt,
        task_name=f"Translate {date_str} to {lang_name}"
    )

    localized_day = copy.deepcopy(day_data)
    
    if "day_of_week_sanskrit" in translated_fields and translated_fields["day_of_week_sanskrit"]:
        localized_day["day_of_week_sanskrit"] = translated_fields["day_of_week_sanskrit"]

    loc_p = localized_day.setdefault("panchang", {})
    if "moon_sign_today" in translated_fields and translated_fields["moon_sign_today"]:
        loc_p["moon_sign_today"] = translated_fields["moon_sign_today"]

    trans_p = translated_fields.get("panchang", {})
    if isinstance(trans_p, dict):
        t = loc_p.setdefault("tithi", {})
        if trans_p.get("tithi_name"): t["name"] = trans_p["tithi_name"]
        if trans_p.get("tithi_paksha"): t["paksha"] = trans_p["tithi_paksha"]
        if trans_p.get("tithi_description"): t["description"] = trans_p["tithi_description"]

        n = loc_p.setdefault("nakshatra", {})
        if trans_p.get("nakshatra_name"): n["name"] = trans_p["nakshatra_name"]
        if trans_p.get("nakshatra_ruler"): n["ruler"] = trans_p["nakshatra_ruler"]

        y = loc_p.setdefault("yoga", {})
        if trans_p.get("yoga_name"): y["name"] = trans_p["yoga_name"]
        if trans_p.get("yoga_nature"): y["nature"] = trans_p["yoga_nature"]

        k = loc_p.setdefault("karana", {})
        if trans_p.get("karana_name"): k["name"] = trans_p["karana_name"]

        cal = loc_p.setdefault("calendar", {})
        if trans_p.get("month_name"):
            cal["amanta_month"] = trans_p["month_name"]
            cal["purnimanta_month"] = trans_p["month_name"]

        if trans_p.get("festival_or_vrat"):
            loc_p["festival_or_vrat"] = trans_p["festival_or_vrat"]

    loc_r = localized_day.setdefault("rashifal", {})
    trans_rashifal = translated_fields.get("rashifal", {})
    if not isinstance(trans_rashifal, dict):
        trans_rashifal = {}

    normalized_trans_rashifal = {}
    for raw_k, raw_v in trans_rashifal.items():
        if isinstance(raw_v, dict):
            mapped_k = RASHI_ALIASES.get(str(raw_k).strip().lower(), str(raw_k).strip())
            normalized_trans_rashifal[mapped_k] = raw_v

    text_keys = ["summary", "general", "career", "finance", "relationships", "health", "lucky_color", "auspicious_time", "remedy"]

    for rashi in RASHIS:
        if rashi not in loc_r or not isinstance(loc_r[rashi], dict):
            loc_r[rashi] = copy.deepcopy(day_data.get("rashifal", {}).get(rashi, get_default_rashi(rashi)))

        r_trans = normalized_trans_rashifal.get(rashi)
        if isinstance(r_trans, dict):
            for key in text_keys:
                if key in r_trans and r_trans[key]:
                    loc_r[rashi][key] = str(r_trans[key])

    return localized_day

def save_dataset_file(output_dir: Path, code: str, dataset: dict):
    ist = ZoneInfo("Asia/Kolkata")
    dataset["metadata"]["generated_at"] = datetime.now(ist).isoformat()
    if dataset["days"]:
        dataset["metadata"]["coverage"] = {
            "start_date": min(dataset["days"].keys()),
            "end_date": max(dataset["days"].keys())
        }

    lang_file = output_dir / f"panchang-rashifal-week-{code}.json"
    lang_file.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    asset_lang_file = Path(f"app/src/main/assets/panchang-rashifal-week-{code}.json")
    try:
        if asset_lang_file.parent.exists():
            asset_lang_file.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"     [!] Notice writing asset {asset_lang_file.name}: {e}", flush=True)

    if code == "en":
        master_file = output_dir / "panchang-rashifal-week.json"
        master_file.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

        asset_master_file = Path("app/src/main/assets/panchang-rashifal-week.json")
        try:
            if asset_master_file.parent.exists():
                asset_master_file.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"     [OK] Synchronized Android asset: {asset_master_file}", flush=True)
        except Exception as e:
            print(f"     [!] Notice writing Android master asset: {e}", flush=True)

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
                    datasets[code] = json.loads(raw_text)
                    if "days" not in datasets[code] or not isinstance(datasets[code]["days"], dict):
                        datasets[code]["days"] = {}
                    continue
            except Exception:
                pass

        datasets[code] = {
            "metadata": {
                "generated_at": None,
                "language_code": code,
                "language_name": name,
                "timezone": "Asia/Kolkata",
                "reference_standard": "New Delhi / Central India (IST)",
                "ayanamsha": "Lahiri / Chitrapaksha",
                "coverage": {"start_date": None, "end_date": None}
            },
            "days": {}
        }

    return datasets

def main():
    ist = ZoneInfo("Asia/Kolkata")
    today = datetime.now(ist).date()
    date_str = today.strftime("%Y-%m-%d")

    print(f"============================================================", flush=True)
    print(f" Hybrid Jyotish Generator (Gemini 3.8 Flash + Groq GPT-OSS)", flush=True)
    print(f" Generating Date (IST): {date_str} ({today.strftime('%A')})", flush=True)
    print(f" Target Languages: {', '.join(LANGUAGES.values())} (8 total)", flush=True)
    print(f" Base Astrologer: Google Gemini ({GEMINI_MODEL} exclusively)", flush=True)
    print(f" Translator: Groq ({' -> '.join(GROQ_MODELS)})", flush=True)
    print(f" Demand Tolerance: Enabled (dynamic backoff, jitter, retry budget)", flush=True)
    print(f"============================================================\n", flush=True)

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_existing_datasets(output_dir)

    print(f"Generating Today: {date_str} ({today.strftime('%A')})", flush=True)
    indic_langs = {c: n for c, n in LANGUAGES.items() if c != "en"}

    # 1. Base English Generation (Gemini 3.8 Flash)
    print(f" [+] Generating Base English with Gemini 3.8 Flash for {date_str}...", flush=True)
    english_day = generate_base_day(today)
    datasets["en"]["days"][date_str] = english_day
    save_dataset_file(output_dir, "en", datasets["en"])
    print(f"     [OK] English saved to disk.", flush=True)
    time.sleep(REQUEST_SPACING_SECONDS)

    # 2. Translate into 7 Indic Languages (Groq GPT-OSS 120B)
    for code, name in indic_langs.items():
        print(f" -> Localizing {date_str} to {name} ({code}) with Groq...", flush=True)
        localized_day = translate_day(english_day, lang_code=code, lang_name=name)
        datasets[code]["days"][date_str] = localized_day
        
        save_dataset_file(output_dir, code, datasets[code])
        print(f"     [OK] {name} saved to disk.", flush=True)
        time.sleep(REQUEST_SPACING_SECONDS)

    print("\n============================================================", flush=True)
    print(f" Success! Today ({date_str}) is 100% generated across all 8 languages.", flush=True)
    print("============================================================", flush=True)

if __name__ == "__main__":
    main()
