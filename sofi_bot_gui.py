# sofi_bot_gui.py
# -*- coding: utf-8 -*-
import os
import time
import logging
import threading
import signal
import sys
import random
import re
import json
from io import BytesIO
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

# --- Networking / OCR / Discord libs ---
import requests
import numpy as np
from PIL import Image
import cv2
from dotenv import load_dotenv
from rapidfuzz import fuzz
import easyocr
import discum
from discum.utils.button import Buttoner

# --- UI (built-in, free) ---
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

# --------------------------
# Global OCR reader handle
# --------------------------
READER = None

# --------------------------
# Logging setup (adds GUI handler later)
# --------------------------
LOG_FILE_NAME = "Sofi_bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE_NAME, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("websocket").setLevel(logging.CRITICAL)
logging.getLogger("discum.gateway.gateway").setLevel(logging.ERROR)

# --------------------------
# Utility / Shared functions
# --------------------------
def preprocess_string(s: str) -> str:
    tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", s or "").upper().split()
    return " ".join(sorted(tokens))

def load_top_characters(json_path: str):
    cleaned = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {json_path}: {e}")
        return [], set()

    for idx, entry in enumerate(data):
        if isinstance(entry, dict):
            if "character" not in entry or "series" not in entry:
                full_name = entry.get("full_name", "").strip()
                if full_name:
                    parts = full_name.split()
                    if len(parts) >= 2:
                        entry["character"] = " ".join(parts[:-1])
                        entry["series"] = parts[-1]
                    else:
                        continue
                else:
                    continue
            if "full_name" not in entry:
                entry["full_name"] = f"{entry['character']} {entry['series']}"
            cleaned.append(entry)
        # else skip quietly
    return cleaned, {e["character"].lower() for e in cleaned if "character" in e}

def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def extract_generation_with_easyocr(image):
    """Return (generations_dict, name, series)."""
    global READER
    if READER is None:
        raise RuntimeError("OCR READER is not initialized")

    def clean_generation_text(ocr_results):
        cleaned_generations = {}
        for text in ocr_results:
            if len(cleaned_generations) >= 3:
                break
            original_text = (text or "").strip()
            if not original_text:
                continue
            text_upper = original_text.upper()

            # Handle 6Gxxxx → Gxxxx
            match_6g = re.match(r"^6G(\d{1,4})$", text_upper)
            if match_6g:
                gen_number = f"G{match_6g.group(1)}"
                cleaned_generations[gen_number] = int(match_6g.group(1))
                continue

            text_clean = re.sub(r"[^a-zA-Z0-9]", "", original_text)
            # ONLY substitutions on gen-like strings
            text_clean = (
                text_clean.replace("i", "1").replace("I", "1")
                .replace("o", "0").replace("O", "0")
                .replace("g", "9").replace("s", "5").replace("S", "5")
                .replace("B", "8").replace("l", "1")
            )
            if text_clean.startswith(("0", "6", "5", "9")) and not text_clean.upper().startswith("G"):
                text_clean = "G" + text_clean[1:]

            match = re.match(r"^G(\d{1,4})$", text_clean.upper())
            if match:
                gen_number = f"G{match.group(1)}"
                cleaned_generations[gen_number] = int(match.group(1))
        return cleaned_generations

    def extract_card_name_and_series(ocr_results):
        if not ocr_results or len(ocr_results) < 2:
            return None, None
        gen_index = -1
        for i, text in enumerate(ocr_results):
            if clean_generation_text([text]):
                gen_index = i
                break
        if gen_index == -1 or gen_index + 1 >= len(ocr_results):
            return None, None
        name = (ocr_results[gen_index + 1] or "").strip()
        series = (ocr_results[gen_index + 2] or "").strip() if gen_index + 2 < len(ocr_results) else None
        return name, series

    if isinstance(image, Image.Image):
        image = np.array(image)

    target_width = 300
    scale_ratio = target_width / image.shape[1]
    image = cv2.resize(image, (target_width, int(image.shape[0] * scale_ratio)), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    extracted_text = READER.readtext(image, detail=0, batch_size=2)
    generations = clean_generation_text(extracted_text)
    name, series = extract_card_name_and_series(extracted_text)
    return generations, name, series

# --- Elite detector: rainbow frame only ---------------------------------------
# Uses along-edge centerline sampling to verify a continuous hue sweep.
# Returns (elite_bool, metrics_dict)  — signature preserved for callers.

def _minimal_circular_range_deg(h_arr: np.ndarray) -> float:
    if h_arr.size == 0:
        return 0.0
    d = np.sort(h_arr.astype(np.float32) * 2.0)  # OpenCV hue→degrees (0..360)
    gaps = np.diff(np.r_[d, d[0] + 360.0])
    largest_gap = float(np.max(gaps)) if gaps.size else 0.0
    return 360.0 - largest_gap

def _circular_std_deg(h_arr: np.ndarray) -> float:
    if h_arr.size == 0:
        return 0.0
    ang = (h_arr.astype(np.float32) * 2.0) * (np.pi / 180.0)
    s = np.sin(ang).sum()
    c = np.cos(ang).sum()
    n = float(h_arr.size)
    R = np.sqrt(s*s + c*c) / max(n, 1.0)
    R = float(np.clip(R, 1e-6, 0.999999))
    return float(np.sqrt(-2.0 * np.log(R)) * (180.0 / np.pi))

def _unwrap_degrees(seq_deg: np.ndarray) -> np.ndarray:
    if seq_deg.size == 0:
        return seq_deg.astype(np.float32)
    s = seq_deg.astype(np.float32) * 2.0
    out = [s[0]]
    for v in s[1:]:
        base = out[-1]
        cand = np.array([v-360.0, v, v+360.0], dtype=np.float32)
        out.append(cand[np.argmin(np.abs(cand - base))])
    return np.array(out, dtype=np.float32)

def _line_hist_metrics(h_arr: np.ndarray, hue_bins: int, bin_min_frac: float):
    if h_arr.size == 0:
        return dict(nonzero_bins=0, coverage=0.0, peak_frac=1.0, sector_count=0)
    hist, _ = np.histogram(h_arr, bins=hue_bins, range=(0, 180))
    total = float(hist.sum())
    peak_frac = float(hist.max() / total) if total > 0 else 1.0
    bin_frac = hist / max(total, 1.0)
    active = (bin_frac > bin_min_frac)
    nonzero_bins = int(active.sum())
    coverage = nonzero_bins / float(hue_bins)
    extended = np.r_[active, active[0]]
    sector_count = int(np.sum(extended[1:] & ~extended[:-1]))
    return dict(nonzero_bins=nonzero_bins, coverage=coverage,
                peak_frac=peak_frac, sector_count=sector_count)

def _centerline(edge, H, W, t, m):
    if edge == 'top':
        y = int(np.clip(m + t // 2, 0, H - 1)); xs = np.arange(W); ys = np.full_like(xs, y)
    elif edge == 'bottom':
        y = int(np.clip(H - m - t // 2 - 1, 0, H - 1)); xs = np.arange(W); ys = np.full_like(xs, y)
    elif edge == 'left':
        x = int(np.clip(m + t // 2, 0, W - 1)); ys = np.arange(H); xs = np.full_like(ys, x)
    else:  # right
        x = int(np.clip(W - m - t // 2 - 1, 0, W - 1)); ys = np.arange(H); xs = np.full_like(ys, x)
    return xs, ys

def _parallel_offsets(t, num_lines):
    if num_lines <= 1:
        return [0]
    offs = np.linspace(-max(1, t // 4), max(1, t // 4), num_lines).astype(int)
    return list(np.unique(offs))

def _sample_line_hues(hsv, xs, ys, edge, off, sat_min, val_min):
    H, W = hsv.shape[:2]
    if edge in ('top', 'bottom'):
        y = np.clip(ys + off, 0, H - 1)
        h = hsv[y, xs, 0]; s = hsv[y, xs, 1]; v = hsv[y, xs, 2]
    else:
        x = np.clip(xs + off, 0, W - 1)
        h = hsv[ys, x, 0]; s = hsv[ys, x, 1]; v = hsv[ys, x, 2]
    mask = (s >= sat_min) & (v >= val_min)
    return h[mask].astype(np.float32)

def is_elite_card(
    card_img_bgr: np.ndarray,
    # geometry
    ring_ratio: float = 0.06,
    edge_margin: float = 0.03,
    # vivid gate
    sat_min: int = 60,
    val_min: int = 75,
    # hue hist
    hue_bins: int = 36,
    bin_min_frac: float = 0.02,
    # along-edge rainbow gates
    line_range_min_deg: float = 180.0,
    line_coverage_req: float = 0.10,   # tuned to catch thin rainbows
    line_sector_req: int = 2,
    line_peak_max: float = 0.55,
    line_count_min: int = 120,
    # smoothness + across-thickness consistency
    smooth_tv_over_range_max: float = 5.0,
    big_jump_deg: float = 45.0,
    big_jump_max_frac: float = 0.20,
    across_lines: int = 5,
    across_std_max_deg: float = 15.0,
    # card aggregation
    min_rainbow_edges: int = 2,
    require_horiz_vert: bool = True,
):
    """
    Return (elite_bool, metrics_dict).
    Elite == card frame shows a rainbow gradient on at least two edges
    (and at least one horizontal + one vertical if require_horiz_vert=True).
    """
    bgr = card_img_bgr
    if bgr is None or bgr.size == 0:
        return False, {"reason": "empty"}
    H, W = bgr.shape[:2]
    if H < 40 or W < 40:
        return False, {"reason": "too_small"}

    # light denoise so we don't pick artwork textures as color changes
    bgr = cv2.bilateralFilter(bgr, 9, 60, 60)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    t = max(1, int(round(min(H, W) * ring_ratio)))
    m = max(1, int(round(min(H, W) * edge_margin)))

    edges_metrics = []
    rainbow_ok = 0
    horiz_ok = False
    vert_ok = False

    for edge in ("top", "bottom", "left", "right"):
        xs, ys = _centerline(edge, H, W, t, m)
        offsets = _parallel_offsets(t, across_lines)
        lines = [_sample_line_hues(hsv, xs, ys, edge, off, sat_min, val_min) for off in offsets]

        # choose the center line for primary stats
        center_idx = offsets.index(0) if 0 in offsets else len(offsets)//2
        line_h = lines[center_idx]
        line_count = int(line_h.size)
        line_range_deg = _minimal_circular_range_deg(line_h)
        hm = _line_hist_metrics(line_h, hue_bins, bin_min_frac)

        uw = _unwrap_degrees(line_h)
        if uw.size >= 2:
            diffs = np.abs(np.diff(uw))
            tv = float(np.sum(diffs))
            big_jump_frac = float(np.mean(diffs >= big_jump_deg)) if diffs.size else 0.0
            tv_over_range = tv / max(line_range_deg, 1e-6)
        else:
            big_jump_frac = 0.0
            tv_over_range = 0.0

        # across-thickness (perpendicular) hue consistency
        across_std_list = []
        if len(lines) >= 2 and line_h.size > 0:
            n = min(200, line_h.size)
            idxs = np.linspace(0, line_h.size - 1, n).astype(int)
            stack = []
            for L in lines:
                if L.size == line_h.size:
                    stack.append(L[idxs])
                elif L.size == 0:
                    stack.append(np.full_like(idxs, np.nan, dtype=np.float32))
                else:
                    jj = (idxs.astype(np.float32) * (L.size - 1) / max(line_h.size - 1, 1)).astype(int)
                    stack.append(L[jj])
            Hstack = np.stack(stack, axis=0)
            for k in range(Hstack.shape[1]):
                col = Hstack[:, k]; col = col[~np.isnan(col)]
                if col.size >= 2:
                    across_std_list.append(_circular_std_deg(col))
        across_std_med = float(np.median(across_std_list)) if across_std_list else 999.0

        ok = (
            line_count >= line_count_min and
            line_range_deg >= line_range_min_deg and
            hm["coverage"] >= line_coverage_req and
            hm["sector_count"] >= line_sector_req and
            hm["peak_frac"] <= line_peak_max and
            tv_over_range <= smooth_tv_over_range_max and
            big_jump_frac <= big_jump_max_frac and
            across_std_med <= across_std_max_deg
        )

        edges_metrics.append({
            "edge": edge, "ok": bool(ok),
            "line_count": line_count,
            "line_hue_range_deg": float(line_range_deg),
            "line_coverage": float(hm["coverage"]),
            "line_sector_count": int(hm["sector_count"]),
            "line_peak_frac": float(hm["peak_frac"]),
            "tv_over_range": float(tv_over_range),
            "big_jump_frac": float(big_jump_frac),
            "across_std_med_deg": float(across_std_med),
        })

        if ok:
            rainbow_ok += 1
            if edge in ("top", "bottom"):
                horiz_ok = True
            else:
                vert_ok = True

    elite = (rainbow_ok >= min_rainbow_edges) and (not require_horiz_vert or (horiz_ok and vert_ok))

    metrics = {
        "edges": edges_metrics,
        "rainbow_edges": rainbow_ok,
        "passes_horiz": horiz_ok,
        "passes_vert":  vert_ok,
        "elite": elite,
        "thresholds": {
            "ring_ratio": ring_ratio, "edge_margin": edge_margin,
            "sat_min": sat_min, "val_min": val_min,
            "hue_bins": hue_bins, "bin_min_frac": bin_min_frac,
            "line_range_min_deg": line_range_min_deg,
            "line_coverage_req": line_coverage_req,
            "line_sector_req": line_sector_req,
            "line_peak_max": line_peak_max,
            "line_count_min": line_count_min,
            "smooth_tv_over_range_max": smooth_tv_over_range_max,
            "big_jump_deg": big_jump_deg,
            "big_jump_max_frac": big_jump_max_frac,
            "across_lines": across_lines,
            "across_std_max_deg": across_std_max_deg,
            "min_rainbow_edges": min_rainbow_edges,
            "require_horiz_vert": require_horiz_vert,
        }
    }
    return elite, metrics

def extract_card_for_index(index, card_crop):
    try:
        # Convert to ndarray
        if isinstance(card_crop, Image.Image):
            rgb = np.array(card_crop)
        else:
            rgb = card_crop

        if rgb.ndim == 3 and rgb.shape[2] == 3:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr = rgb

        # Elite detection
        elite, emetrics = is_elite_card(bgr)
        if elite:
            logging.info(f"🌈 Elite detected on card {index+1} → {emetrics}")

        # OCR pipeline
        processed = preprocess_image(card_crop)
        gens, name, series = extract_generation_with_easyocr(processed)

        return index, (gens, name, series, elite)

    except Exception as e:
        logging.warning(f"⚠️ Failed to process card index {index}: {e}")
        return index, ({}, "", "", False)

def extract_card_generations(image: Image.Image):
    """
    Splits the 3-card image, OCRs each card in parallel, and returns:
      { index: {
          'generations': dict,
          'name': str,
          'series': str,
          'elite': bool
        }, ... }
    """
    card_width = image.width // 3
    card_info = {}

    def _crop_card(i: int) -> Image.Image:
        left = i * card_width
        right = left + card_width if i < 2 else image.width  # last card gets any remainder
        return image.crop((left, 0, right, image.height))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_card_for_index, i, _crop_card(i)) for i in range(3)]

        for fut in futures:
            try:
                i, (gens, name, series, elite) = fut.result()

                # record if we have any useful signal at all
                if gens or name or elite:
                    card_info[i] = {
                        "generations": gens or {},
                        "name": name or "",
                        "series": series or "",
                        "elite": bool(elite),
                    }
                else:
                    # ensure the key exists with defaults (optional)
                    card_info[i] = {
                        "generations": {},
                        "name": "",
                        "series": "",
                        "elite": False,
                    }

                logging.info(
                    f"✅ Card {i}: Gen={list((gens or {}).keys()) or '∅'}, "
                    f"Name='{name or ''}', Series='{series or ''}', Elite={elite}"
                )

            except Exception as e:
                logging.warning(f"⚠️ Failed to extract data for card: {e}")
                # ensure safe default
                card_info[len(card_info)] = {
                    "generations": {},
                    "name": "",
                    "series": "",
                    "elite": False,
                }

    logging.debug(f"🚀 FINAL CARD INFO: {card_info}")
    return card_info

def find_best_character_match(name, series, TOP_CHARACTERS_LIST):
    if not name:
        return None, 0, 0, None, None
    try:
        normalized_name = preprocess_string((name or "").rstrip("-"))
        normalized_series = preprocess_string((series or "").rstrip("-"))

        best_entry = None
        best_score = 0
        best_series_score = 0
        best_index = 0

        for idx, entry in enumerate(TOP_CHARACTERS_LIST):
            character = preprocess_string((entry.get("character", "") or "").rstrip("...").rstrip("-"))
            series_name = preprocess_string((entry.get("series", "") or "").rstrip("...").rstrip("-"))

            if len(character) <= 4:
                name_score = fuzz.ratio(normalized_name, character)
            else:
                name_score = fuzz.partial_ratio(normalized_name, character)

            if name_score > 85:
                if len(series_name) <= 4:
                    series_score = fuzz.ratio(normalized_series, series_name)
                else:
                    series_score = fuzz.partial_ratio(normalized_series, series_name)

                if series_score > 85 and name_score + series_score > best_score + best_series_score:
                    best_score = name_score
                    best_series_score = series_score
                    best_entry = entry
                    best_index = idx

        if best_entry:
            full_name = best_entry.get("full_name") or f"{best_entry['character']} {best_entry['series']}"
            return best_score, best_index, best_entry.get("likes", 0), full_name, best_entry.get("series")
    except Exception as e:
        logging.warning(f"Match failed for {name} - {series}: {e}")
    return None, 0, 0, None, None

# --------------------------
# Bot Manager
# --------------------------
class SofiBotManager:
    SOFI_BOT_ID = "853629533855809596"

    def __init__(self, cfg_path=".env", leaderboard="sofi_leaderboard_extended.json"):
        self.cfg_path = cfg_path
        self.leaderboard_path = leaderboard

        # locks & state
        self.last_drop_id_lock = Lock()
        self.last_processed_time_lock = Lock()

        self.pending_claim = {"message_id": None, "timestamp": None, "user_id": None, "triggered": False}
        self.last_processed_time = 0
        self.PROCESS_COOLDOWN_SECONDS = 240
        self.last_message_received = time.time()
        self.WATCHDOG_TIMEOUT = 510

        self.stop_event = threading.Event()
        self.bot = None
        self.bot_thread = None
        self.sd_thread = None
        self.ka_thread = None
        self.watchdog_thread = None
        self.claim_timeout_thread = None

        self.bot_identity_logged = False
        self.sent_initial_sd = False  # ensure first sd fires after on_ready

        # config defaults
        load_dotenv(self.cfg_path)
        self.TOKEN = os.getenv("DISCORD_TOKEN", "")
        self.GUILD_ID = os.getenv("GUILD_ID", "")
        self.CHANNEL_ID = os.getenv("CHANNEL_ID", "")
        self.USER_ID = os.getenv("USER_ID", "")
        self.SD_INTERVAL_SEC = int(os.getenv("SD_INTERVAL_SEC", "480"))
        self.USE_GPU = (os.getenv("OCR_USE_GPU", "0") == "1")

        # preferences
        prefs_raw = os.getenv("PREFERENCES", "no_gen,low_gen,high_likes,series_match")
        self.preferences = [p.strip().lower() for p in prefs_raw.split(",") if p.strip()]
        valid_prefs = {"no_gen", "low_gen", "high_likes", "series_match"}
        self.preferences = [p for p in self.preferences if p in valid_prefs]
        if not self.preferences:
            self.preferences = ["no_gen", "low_gen", "high_likes", "series_match"]
        self.series_preference = os.getenv("SERIES_NAME", "").strip()

        # leaderboard
        self.TOP_CHARACTERS_LIST, self.TOP_CHARACTERS_SET = load_top_characters(self.leaderboard_path)

    # --------- Threads ---------
    def message_watchdog(self):
        while not self.stop_event.is_set():
            time.sleep(10)
            elapsed = time.time() - self.last_message_received
            if elapsed > self.WATCHDOG_TIMEOUT:
                logging.error(f"🛑 No message received for {elapsed:.1f}s. Restarting bot (soft)…")
                self.restart_bot()
                return  # let new watchdog start in restart_bot

    def reset_claim_if_timed_out(self):
        while not self.stop_event.is_set():
            time.sleep(1)
            if self.pending_claim["triggered"] and time.time() - (self.pending_claim["timestamp"] or 0) > 7:
                logging.warning("⚠️ No Sofi confirmation received. Resetting claim trigger.")
                self.pending_claim["triggered"] = False

    def periodic_sd_sender(self):
        """Wait until initial 'sd' is sent (in on_ready), then send every ~8 minutes."""
        while not self.stop_event.is_set() and not self.sent_initial_sd:
            time.sleep(0.2)

        while not self.stop_event.is_set():
            wait_time = float(self.SD_INTERVAL_SEC) + random.uniform(0, 10)
            logging.info(f"⏳ Waiting {wait_time:.1f}s before sending next 'sd'…")
            self.stop_event.wait(wait_time)
            if self.stop_event.is_set():
                break
            if not self.bot or not self.bot.gateway.session_id:
                logging.warning("⚠️ Gateway session missing. Skipping 'sd'.")
                continue
            try:
                self.bot.sendMessage(self.CHANNEL_ID, "sd")
                logging.info("📤 Sent 'sd' command.")
            except Exception:
                logging.exception("⚠️ Failed to send 'sd'")

    def keep_alive(self):
        previous_latency = None
        failure_count = 0
        while not self.stop_event.is_set():
            time.sleep(30)
            try:
                latency = self.bot.gateway.latency if self.bot else None
                if latency is None or latency == previous_latency:
                    failure_count += 1
                else:
                    failure_count = 0
                previous_latency = latency
                if failure_count >= 3:
                    logging.error("🛑 Gateway appears frozen. Restarting bot (soft)…")
                    self.restart_bot()
                    return
            except Exception as e:
                logging.error(f"❌ keep_alive check failed: {e}")

    # --------- Bot actions ---------
    def click_discord_button(self, custom_id, channel_id, guild_id, m):
        try:
            self.bot.click(
                applicationID=m["author"]["id"],
                channelID=channel_id,
                guildID=m.get("guild_id"),
                messageID=m["id"],
                messageFlags=m["flags"],
                data={"component_type": 2, "custom_id": custom_id},
            )
            logging.info(f"Dispatched click for button {custom_id}")
        except Exception as e:
            logging.warning(f"Exception during click: {e}")

    def click_bouquet_then_best_from_image(self, pil_image, buttons_components, image_received_time, channel_id, guild_id, m):
        card_count = 3
        button_ids = [
            btn["custom_id"]
            for component_row in buttons_components
            for btn in component_row.get("components", [])
            if btn["type"] == 2 and btn["style"] in (1, 2)
        ]
        if len(button_ids) != 3:
            logging.warning(f"Expected 3 buttons but found: {len(button_ids)}")
            return

        card_info = extract_card_generations(pil_image)

        # ----- Build per-card metadata once -----
        cards = []
        for i in range(card_count):
            if card_info.get(i, {}).get("elite"):
                self.click_discord_button(button_ids[i], channel_id, guild_id, m)
                logging.info(f"✅ Claimed card {i+1} (🌈 ELITE 🌈)")
                return
            info = card_info.get(i, {})
            gens = info.get("generations", {}) or {}
            name = info.get("name") or ""
            series = info.get("series") or ""

            match_score, match_idx, likes, matched_name, matched_series = find_best_character_match(
                name, series, self.TOP_CHARACTERS_LIST
            )

            cards.append({
                "index": i,
                "gens": gens,
                "has_gen": bool(gens),
                "min_gen": min(gens.values()) if gens else None,
                "name": name,
                "series": series,
                "match_score": match_score if match_score is not None else 0,
                "likes": likes or 0,
                "matched_series": (matched_series or "")
            })

        # 🚨 Absolute rule: if any card has generation < 10, choose it now (ignore preferences)
        low_gen_candidates = [c for c in cards if c["min_gen"] is not None and c["min_gen"] < 10]
        if low_gen_candidates:
            low_gen_candidates.sort(key=lambda c: (c["min_gen"], -c["likes"], c["index"]))
            chosen = low_gen_candidates[0]
            chosen_index = chosen["index"]

            now = time.time()
            if chosen.get("gens", {}):
                with self.last_processed_time_lock:
                    cooldown_active = self.last_processed_time and (now - self.last_processed_time < self.PROCESS_COOLDOWN_SECONDS)
                    pending_active = self.pending_claim.get("triggered", False)
                    if pending_active or cooldown_active:
                        if pending_active:
                            logging.info("Skipping claim — waiting previous claim confirmation.")
                        elif cooldown_active:
                            logging.info("Skipping claim — cooldown not expired.")
                        return

            self.click_discord_button(button_ids[chosen_index], channel_id, guild_id, m)
            self.pending_claim["timestamp"] = now
            self.pending_claim["triggered"] = True
            self.pending_claim["user_id"] = self.USER_ID

            elapsed_time = time.time() - image_received_time
            logging.info(f"✅ Claimed card {chosen_index+1} (⚡ generation < 10: G{chosen['min_gen']} ⚡) in {elapsed_time:.2f}s")
            return

        def filter_by_pref(candidates, pref):
            if pref == "no_gen":
                kept = [c for c in candidates if not c["has_gen"]]
                return kept if kept else candidates

            if pref == "low_gen":
                with_gen = [c for c in candidates if c["min_gen"] is not None]
                if not with_gen:
                    return candidates
                minval = min(c["min_gen"] for c in with_gen)
                return [c for c in with_gen if c["min_gen"] == minval]

            if pref == "high_likes":
                with_likes = [c for c in candidates if c["likes"] > 0]
                if not with_likes:
                    return candidates
                maxlikes = max(c["likes"] for c in with_likes)
                return [c for c in with_likes if c["likes"] == maxlikes]

            if pref == "series_match":
                key = (self.series_preference or "").strip().lower()
                if not key:
                    return candidates
                matched = [
                    c for c in candidates
                    if key in (c["matched_series"] or "").lower()
                    or key in (c["series"] or "").lower()
                ]
                return matched if matched else candidates

            return candidates

        candidates = cards[:]
        for pref in self.preferences:
            before = candidates
            candidates = filter_by_pref(candidates, pref)
            if not candidates:
                candidates = before

        def sort_key(c):
            gen = c["min_gen"] if c["min_gen"] is not None else 10**9
            return (gen, -c["likes"])

        candidates.sort(key=sort_key)
        chosen = candidates[0] if candidates else None
        if not chosen:
            logging.warning("No card chosen to claim.")
            return

        chosen_index = chosen["index"]

        now = time.time()
        generations = chosen.get("gens", {})

        if generations:
            with self.last_processed_time_lock:
                cooldown_active = self.last_processed_time and (now - self.last_processed_time < self.PROCESS_COOLDOWN_SECONDS)
                pending_active = self.pending_claim.get("triggered", False)
                if pending_active or cooldown_active:
                    if pending_active:
                        logging.info("Skipping claim — waiting previous claim confirmation.")
                    elif cooldown_active:
                        logging.info("Skipping claim — cooldown not expired.")
                    return

        self.click_discord_button(button_ids[chosen_index], channel_id, guild_id, m)
        self.pending_claim["timestamp"] = now
        self.pending_claim["triggered"] = True
        self.pending_claim["user_id"] = self.USER_ID

        card_claimed_time = time.time()
        elapsed_time = card_claimed_time - image_received_time

        tag = ""
        if not chosen["has_gen"]:
            tag = "no generation"
        elif chosen["min_gen"] is not None and chosen["min_gen"] < 10:
            tag = f"gen {chosen['min_gen']}"
        elif chosen["likes"] > 0:
            tag = f"likes {chosen['likes']}"
        if self.series_preference and (
            self.series_preference.lower() in (chosen["matched_series"] or "").lower()
            or self.series_preference.lower() in (chosen["series"] or "").lower()
        ):
            tag = f"{tag} / series match".strip(" /")

        logging.info(f"✅ Claimed card {chosen_index+1} ({tag or 'preference match'}) in {elapsed_time:.2f}s")

    # --------- Bot lifecycle ---------
    def _install_handlers(self):
        @self.bot.gateway.command
        def on_ready(resp):
            if not resp.event.ready:
                return
            user = resp.parsed.auto().get("user")
            if user:
                logging.info(f"Connected as {user['username']}#{user['discriminator']}")
            else:
                logging.info("Connected (user info missing).")

            # 🔥 Initial 'sd' right after gateway is ready (only once)
            if not self.sent_initial_sd:
                for attempt in range(1, 6):
                    try:
                        if self.bot and self.bot.gateway.session_id:
                            self.bot.sendMessage(self.CHANNEL_ID, "sd")
                            self.sent_initial_sd = True
                            logging.info("📤 Sent 'sd' command.")
                            break
                        else:
                            logging.debug("Gateway ready but session_id not set yet; retrying…")
                    except Exception:
                        logging.exception("⚠️ Failed to send initial 'sd'")
                    time.sleep(0.5)

        @self.bot.gateway.command
        def on_message(resp):
            if not hasattr(resp, "raw") or resp.raw.get("t") != "MESSAGE_CREATE":
                return
            data = resp.raw["d"]
            m = resp.parsed.auto()
            author_id = str(data.get("author", {}).get("id"))
            channel_id = str(data.get("channel_id"))
            content = data.get("content", "")

            # record our identity when we speak
            if not self.bot_identity_logged and author_id == self.USER_ID:
                try:
                    username = data.get("author", {}).get("username", "")
                    logging.info(f"Bot identity confirmed: {username} (ID: {author_id})")
                    self.pending_claim["user_id"] = author_id
                except Exception:
                    pass
                self.bot_identity_logged = True

            # Only listen to Sofi bot
            if author_id != SofiBotManager.SOFI_BOT_ID:
                return

            self.last_message_received = time.time()

            # confirmation of claim
            if self.pending_claim.get("triggered"):
                expected_grab = f"<@{self.pending_claim['user_id']}> **grabbed** the"
                expected_fight = f"<@{self.pending_claim['user_id']}> fought off"
                if content.startswith(expected_grab) or content.startswith(expected_fight):
                    logging.info(f"Claim confirmed: {content}")
                    with self.last_processed_time_lock:
                        self.last_processed_time = time.time()
                    self.pending_claim["triggered"] = False
                    return

            attachments = data.get("attachments", [])
            components = data.get("components", [])
            if not attachments or not components:
                return

            def process_sofi_drop(attachment_url, components, channel_id, guild_id, m):
                try:
                    response = requests.get(attachment_url, timeout=15)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    image_received_time = time.time()
                    self.click_bouquet_then_best_from_image(image, components, image_received_time, channel_id, guild_id, m)
                except Exception as e:
                    logging.warning(f"Failed to process image: {e}")

            # start a short-lived thread per attachment image
            for att in attachments:
                fn = att.get("filename", "")
                if any(fn.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
                    t = threading.Thread(
                        target=process_sofi_drop,
                        args=(att.get("url"), components, channel_id, data.get("guild_id"), m),
                        daemon=True,
                    )
                    t.start()

    def start(self):
        global READER
        if self.bot_thread and self.bot_thread.is_alive():
            logging.info("Bot already running.")
            return

        # Init OCR (download on first run). Default: CPU (packaging-friendly)
        logging.info(f"Initializing EasyOCR (gpu={self.USE_GPU}) …")
        READER = easyocr.Reader(["en"], gpu=self.USE_GPU, verbose=False)
        # warm-up
        _ = READER.readtext(np.zeros((100, 100, 3), dtype=np.uint8), detail=0)

        # Init Discum
        self.bot = discum.Client(token=self.TOKEN, log=False)
        self._install_handlers()

        self.stop_event.clear()
        self.pending_claim["user_id"] = self.USER_ID
        self.bot_identity_logged = False
        self.sent_initial_sd = False

        # helper threads
        self.sd_thread = threading.Thread(target=self.periodic_sd_sender, daemon=True)
        self.ka_thread = threading.Thread(target=self.keep_alive, daemon=True)
        self.watchdog_thread = threading.Thread(target=self.message_watchdog, daemon=True)
        self.claim_timeout_thread = threading.Thread(target=self.reset_claim_if_timed_out, daemon=True)

        self.sd_thread.start()
        self.ka_thread.start()
        self.watchdog_thread.start()
        self.claim_timeout_thread.start()

        # gateway thread
        def run_gateway():
            logging.info("Connecting to Discord gateway …")
            try:
                self.bot.gateway.run(auto_reconnect=True)
            except Exception as e:
                logging.critical(f"Failed to connect to Discord gateway: {e}")
                logging.critical("Check DISCORD_TOKEN, internet, or Discord status.")

        self.bot_thread = threading.Thread(target=run_gateway, daemon=True)
        self.bot_thread.start()

    def stop(self):
        logging.info("Stopping bot …")
        self.stop_event.set()
        try:
            if self.bot and self.bot.gateway:
                self.bot.gateway.close()
        except Exception:
            pass
        # try to join helper threads quickly
        for t in (self.sd_thread, self.ka_thread, self.watchdog_thread, self.claim_timeout_thread, self.bot_thread):
            try:
                if t and t.is_alive():
                    t.join(timeout=2.0)
            except Exception:
                pass
        logging.info("Bot stopped.")

    # ---- NEW: soft restart (no process restart) ----
    def restart_bot(self):
        logging.info("🔄 Soft-restarting bot …")
        try:
            self.stop()
        except Exception:
            pass

        # reset flags/state but keep OCR reader
        self.stop_event = threading.Event()
        self.pending_claim["triggered"] = False
        self.sent_initial_sd = False
        self.bot_identity_logged = False
        self.last_message_received = time.time()

        # recreate Discum client and handlers
        try:
            self.bot = discum.Client(token=self.TOKEN, log=False)
            self._install_handlers()
        except Exception as e:
            logging.error(f"Failed to re-create Discum client: {e}")
            return

        # restart helper threads
        self.sd_thread = threading.Thread(target=self.periodic_sd_sender, daemon=True)
        self.ka_thread = threading.Thread(target=self.keep_alive, daemon=True)
        self.watchdog_thread = threading.Thread(target=self.message_watchdog, daemon=True)
        self.claim_timeout_thread = threading.Thread(target=self.reset_claim_if_timed_out, daemon=True)

        self.sd_thread.start()
        self.ka_thread.start()
        self.watchdog_thread.start()
        self.claim_timeout_thread.start()

        # restart gateway
        def run_gateway():
            logging.info("Reconnecting to Discord gateway …")
            try:
                self.bot.gateway.run(auto_reconnect=True)
            except Exception as e:
                logging.critical(f"Failed to reconnect to Discord gateway: {e}")

        self.bot_thread = threading.Thread(target=run_gateway, daemon=True)
        self.bot_thread.start()

    # (kept for reference; not used anymore)
    def restart_process(self):
        self.stop()
        time.sleep(1)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def save_env(self):
        data = {
            "DISCORD_TOKEN": self.TOKEN,
            "GUILD_ID": self.GUILD_ID,
            "CHANNEL_ID": self.CHANNEL_ID,
            "USER_ID": self.USER_ID,
            "SD_INTERVAL_SEC": str(self.SD_INTERVAL_SEC),
            "OCR_USE_GPU": "1" if self.USE_GPU else "0",
            "PREFERENCES": ",".join(self.preferences),
            "SERIES_NAME": self.series_preference,
        }
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            for k, v in data.items():
                f.write(f"{k}={v}\n")
        logging.info(f"Saved settings to {self.cfg_path}")

# --------------------------
# Tkinter UI
# --------------------------
class TextHandler(logging.Handler):
    """Send logs to Tkinter with per-level colors using text tags."""
    def __init__(self, widget: ScrolledText):
        super().__init__()
        self.widget = widget
        self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    def emit(self, record):
        msg = self.format(record) + "\n"
        level = record.levelno
        # choose tag by level
        if level >= logging.CRITICAL:
            tag = "CRITICAL"
        elif level >= logging.ERROR:
            tag = "ERROR"
        elif level >= logging.WARNING:
            tag = "WARNING"
        elif level >= logging.INFO:
            tag = "INFO"
        else:
            tag = "DEBUG"

        # UI updates must be scheduled on the main thread
        try:
            self.widget.after(0, self._append_with_tag, msg, tag)
        except Exception:
            pass

    def _append_with_tag(self, msg, tag):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, msg, tag)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")

class SofiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sofi Farming Bot (Discum + OCR)")
        self.geometry("980x760")
        self.minsize(900, 650)

        # Bot manager
        self.manager = SofiBotManager()

        # UI
        self._build_ui()
        self._wire_logging()

        # preload fields
        self._load_fields_from_manager()

        # OS signals to close cleanly if run as script
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._start()
        self._tick_status()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", **pad)

        # Token
        ttk.Label(frm_top, text="Discord Token:").grid(row=0, column=0, sticky="w")
        self.var_token = tk.StringVar()
        self.ent_token = ttk.Entry(frm_top, textvariable=self.var_token, width=64, show="•")
        self.ent_token.grid(row=0, column=1, sticky="ew", columnspan=3)

        # Show/Hide token
        self.show_token = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_top, text="Show", variable=self.show_token, command=self._toggle_token).grid(row=0, column=4, sticky="w")

        # IDs row
        ttk.Label(frm_top, text="Guild ID:").grid(row=1, column=0, sticky="w")
        self.var_guild = tk.StringVar()
        ttk.Entry(frm_top, textvariable=self.var_guild, width=24).grid(row=1, column=1, sticky="w")

        ttk.Label(frm_top, text="Channel ID:").grid(row=1, column=2, sticky="w")
        self.var_channel = tk.StringVar()
        ttk.Entry(frm_top, textvariable=self.var_channel, width=24).grid(row=1, column=3, sticky="w")

        ttk.Label(frm_top, text="Your User ID:").grid(row=2, column=0, sticky="w")
        self.var_user = tk.StringVar()
        ttk.Entry(frm_top, textvariable=self.var_user, width=24).grid(row=2, column=1, sticky="w")

        # Interval & OCR
        ttk.Label(frm_top, text="‘sd’ Interval (sec):").grid(row=2, column=2, sticky="w")
        self.var_interval = tk.IntVar(value=480)
        ttk.Spinbox(frm_top, from_=60, to=3600, increment=10, textvariable=self.var_interval, width=10).grid(row=2, column=3, sticky="w")

        self.var_gpu = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_top, text="Use GPU for OCR (advanced)", variable=self.var_gpu).grid(row=3, column=1, sticky="w", pady=(2, 8))

        # ----- Preferences -----
        frm_prefs = ttk.LabelFrame(self, text="Claim preferences (priority order)")
        frm_prefs.pack(fill="x", padx=8, pady=6)

        choices = ["no_gen", "low_gen", "high_likes", "series_match"]

        ttk.Label(frm_prefs, text="Priority 1").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(frm_prefs, text="Priority 2").grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frm_prefs, text="Priority 3").grid(row=0, column=2, sticky="w", padx=6, pady=4)
        ttk.Label(frm_prefs, text="Priority 4").grid(row=0, column=3, sticky="w", padx=6, pady=4)

        self.var_pref1 = tk.StringVar(value="no_gen")
        self.var_pref2 = tk.StringVar(value="low_gen")
        self.var_pref3 = tk.StringVar(value="high_likes")
        self.var_pref4 = tk.StringVar(value="series_match")

        self.cmb_pref1 = ttk.Combobox(frm_prefs, values=choices, textvariable=self.var_pref1, state="readonly", width=16)
        self.cmb_pref2 = ttk.Combobox(frm_prefs, values=choices, textvariable=self.var_pref2, state="readonly", width=16)
        self.cmb_pref3 = ttk.Combobox(frm_prefs, values=choices, textvariable=self.var_pref3, state="readonly", width=16)
        self.cmb_pref4 = ttk.Combobox(frm_prefs, values=choices, textvariable=self.var_pref4, state="readonly", width=16)
        self.cmb_pref1.grid(row=1, column=0, padx=6, pady=4, sticky="w")
        self.cmb_pref2.grid(row=1, column=1, padx=6, pady=4, sticky="w")
        self.cmb_pref3.grid(row=1, column=2, padx=6, pady=4, sticky="w")
        self.cmb_pref4.grid(row=1, column=3, padx=6, pady=4, sticky="w")

        ttk.Label(frm_prefs, text="Series name (for 'series_match')").grid(row=2, column=0, sticky="w", padx=6, pady=(6, 4))
        self.var_series = tk.StringVar(value="")
        self.ent_series = ttk.Entry(frm_prefs, textvariable=self.var_series, width=40)
        self.ent_series.grid(row=2, column=1, columnspan=3, sticky="w", padx=6, pady=(6, 4))

        def _toggle_series_state(*_):
            prefs = [self.var_pref1.get(), self.var_pref2.get(), self.var_pref3.get(), self.var_pref4.get()]
            enable = "series_match" in prefs
            self.ent_series.configure(state=("normal" if enable else "disabled"))

        for cmb in (self.cmb_pref1, self.cmb_pref2, self.cmb_pref3, self.cmb_pref4):
            cmb.bind("<<ComboboxSelected>>", _toggle_series_state)
        _toggle_series_state()

        # Buttons
        frm_btns = ttk.Frame(self)
        frm_btns.pack(fill="x", **pad)

        self.btn_start = ttk.Button(frm_btns, text="Start Bot", command=self._start)
        self.btn_start.pack(side="left", padx=4)

        self.btn_stop = ttk.Button(frm_btns, text="Stop Bot", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)

        ttk.Button(frm_btns, text="Save .env", command=self._save_env).pack(side="left", padx=8)
        ttk.Button(frm_btns, text="Open Log Folder", command=self._open_log_folder).pack(side="left", padx=8)

        # Status
        frm_status = ttk.Frame(self)
        frm_status.pack(fill="x", **pad)
        self.lbl_status = ttk.Label(frm_status, text="Status: Idle", foreground="#1d4ed8")
        self.lbl_status.pack(side="left")

        # Log console
        frm_log = ttk.LabelFrame(self, text="Logs")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt_log = ScrolledText(frm_log, height=24, state="disabled")
        self.txt_log.pack(fill="both", expand=True, padx=6, pady=6)

        self.txt_log.tag_config("DEBUG",    foreground="#6b7280")  # slate-500
        self.txt_log.tag_config("INFO",     foreground="#2563eb")  # blue-600
        self.txt_log.tag_config("WARNING",  foreground="#d97706")  # amber-600
        self.txt_log.tag_config("ERROR",    foreground="#dc2626")  # red-600
        self.txt_log.tag_config("CRITICAL", foreground="#ffffff", background="#b91c1c")  # white on red
        self.txt_log.tag_config("CRITICAL", font=("TkDefaultFont", 9, "bold"))

        # Footer
        frm_footer = ttk.Frame(self)
        frm_footer.pack(fill="x", **pad)
        ttk.Label(
            frm_footer,
            text="Note: Self-bots violate Discord ToS. Use at your own risk.",
            foreground="#b91c1c",
        ).pack(side="left")

        # Grid config for top
        for c in range(5):
            frm_top.grid_columnconfigure(c, weight=1 if c in (1, 3) else 0)

    def _wire_logging(self):
        handler = TextHandler(self.txt_log)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)

    def _load_fields_from_manager(self):
        m = self.manager
        self.var_token.set(m.TOKEN)
        self.var_guild.set(m.GUILD_ID)
        self.var_channel.set(m.CHANNEL_ID)
        self.var_user.set(m.USER_ID)
        self.var_interval.set(m.SD_INTERVAL_SEC)
        self.var_gpu.set(bool(m.USE_GPU))

        # preferences
        prefs = (m.preferences + ["", "", "", ""])[:4]
        self.var_pref1.set(prefs[0] or "no_gen")
        self.var_pref2.set(prefs[1] or "low_gen")
        self.var_pref3.set(prefs[2] or "high_likes")
        self.var_pref4.set(prefs[3] or "series_match")
        self.var_series.set(m.series_preference or "")

    def _toggle_token(self):
        self.ent_token.configure(show="" if self.show_token.get() else "•")

    def _start(self):
        # push UI values into manager
        m = self.manager
        m.TOKEN = self.var_token.get().strip()
        m.GUILD_ID = self.var_guild.get().strip()
        m.CHANNEL_ID = self.var_channel.get().strip()
        m.USER_ID = self.var_user.get().strip()
        m.SD_INTERVAL_SEC = int(self.var_interval.get())
        m.USE_GPU = bool(self.var_gpu.get())

        # preferences from UI, keep order & unique
        prefs = [self.var_pref1.get(), self.var_pref2.get(), self.var_pref3.get(), self.var_pref4.get()]
        seen = set()
        m.preferences = []
        for p in prefs:
            p = p.strip().lower()
            if p and p not in seen:
                m.preferences.append(p)
                seen.add(p)
        m.series_preference = self.var_series.get().strip()

        if not m.TOKEN or not m.CHANNEL_ID or not m.USER_ID:
            messagebox.showerror("Missing", "Please fill Token, Channel ID, and User ID.")
            return

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lbl_status.configure(text="Status: Running", foreground="#16a34a")

        # start bot
        try:
            m.start()
        except Exception as e:
            logging.exception("Failed to start bot")
            self._stop()

    def _stop(self):
        try:
            self.manager.stop()
        except Exception:
            pass
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.lbl_status.configure(text="Status: Stopped", foreground="#dc2626")

    def _save_env(self):
        m = self.manager
        m.TOKEN = self.var_token.get().strip()
        m.GUILD_ID = self.var_guild.get().strip()
        m.CHANNEL_ID = self.var_channel.get().strip()
        m.USER_ID = self.var_user.get().strip()
        m.SD_INTERVAL_SEC = int(self.var_interval.get())
        m.USE_GPU = bool(self.var_gpu.get())

        # preferences from UI
        prefs = [self.var_pref1.get(), self.var_pref2.get(), self.var_pref3.get(), self.var_pref4.get()]
        seen = set()
        m.preferences = []
        for p in prefs:
            p = p.strip().lower()
            if p and p not in seen:
                m.preferences.append(p)
                seen.add(p)
        m.series_preference = self.var_series.get().strip()

        try:
            m.save_env()
            messagebox.showinfo("Saved", f"Settings saved to {m.cfg_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save .env: {e}")

    def _open_log_folder(self):
        folder = os.getcwd()
        if sys.platform.startswith("win"):
            os.startfile(folder)
        else:
            messagebox.showinfo("Folder", f"Logs are in: {folder}")

    def _signal_handler(self, *_):
        self._stop()
        self.destroy()

    def _tick_status(self):
        # live heartbeat
        if self.manager and self.btn_stop["state"] == "normal":
            since = time.time() - self.manager.last_message_received
            self.lbl_status.configure(text=f"Status: Running — Last Sofi msg {int(since)}s ago")
        self.after(1000, self._tick_status)

if __name__ == "__main__":
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="Run without Tkinter UI")
    args = p.parse_args()

    if args.headless:
        mgr = SofiBotManager()
        try:
            mgr.start()
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            mgr.stop()
    else:
        app = SofiApp()
        app.mainloop()
