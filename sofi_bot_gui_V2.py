# sofi_bot_gui.py
# -*- coding: utf-8 -*-
"""
Sofi Farming Bot (Discum + EasyOCR)

UI:
- Auto-start (no Start button).
- Change Mode: smart / normal.
- Normal priorities P1→P3: high_likes / low_gen / no_gen / series.
- Series preference (optional).
- Apply (live), Save .env, Stop, Clear Logs, Copy Logs.
- No manual reboot button (auto-reboots only when needed).

Bot:
- Auto-reboot (stop -> start) when:
  • gateway not READY at 'sd' send time
  • gateway never reaches READY after connect
  • gateway appears frozen (no events / latency stuck)
- Claim order:
  1) Claim all Acorns (buttons without like number)
  2) ELITE fast-path
  3) GEN < 10 fast-path
  4) Smart / Normal picker (ignore 4th button)
- Smart scoring: adaptive likes vs gen weighting (low gen = higher score).
- Normal priorities: P1 → P2 → P3 among (high_likes, low_gen, no_gen, series).

DISCLAIMER: Using self-bots violates Discord ToS. You assume all risk.
"""

import os
import time
import logging
import threading
import signal
import sys
import random
import re
from io import BytesIO
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Networking / OCR / Discord libs ---
import requests
import numpy as np
from PIL import Image
import cv2
from dotenv import load_dotenv
import easyocr
import discum

# --- UI (built-in) ---
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

# --------------------------
# Global OCR reader handle
# --------------------------
READER = None

# ---- Elite detector config (lean defaults) ----
ELITE_PARAMS = {
    "ring_ratio": 0.045,
    "edge_margin": 0.05,
    "sat_min": 45,
    "val_min": 60,
    "hue_bins": 36,
    "edge_nonzero_bins_min": 8,
    "edge_peak_max": 0.55,
    "edge_cstd_min": 40.0,
    "edge_count_min": 120,
}

# --------------------------
# Logging setup
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
# Utilities
# --------------------------
def _create_http_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=False,
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    sess = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({"User-Agent": "SofiBot/1.0 (+requests)"})
    return sess

def pil_from_url_or_none(session: requests.Session, url: str, timeout: float = 15.0):
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        logging.warning(f"Image fetch failed: {e}")
        return None

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
            text_upper = (original_text or "").upper()

            # Example: 6G123 -> G123
            match_6g = re.match(r"^6G(\d{1,4})$", text_upper)
            if match_6g:
                gen_number = f"G{match_6g.group(1)}"
                cleaned_generations[gen_number] = int(match_6g.group(1))
                continue

            text_clean = re.sub(r"[^a-zA-Z0-9]", "", original_text)
            # common OCR substitutions
            text_clean = (
                text_clean.replace("i", "1").replace("I", "1")
                .replace("o", "0").replace("O", "0")
                .replace("g", "9").replace("s", "5").replace("S", "5")
                .replace("B", "8").replace("l", "1")
            )
            if text_clean and text_clean[0] in ("0", "6", "5", "9") and not text_clean.upper().startswith("G"):
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

# ---------- Elite detector (lean implementation) ----------
def _segment_card_mask(col_bgr):
    h, w = col_bgr.shape[:2]
    gray = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -5)
    mix = cv2.bitwise_or(th, edges)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mix = cv2.morphologyEx(mix, cv2.MORPH_CLOSE, k, iterations=2)
    mix = cv2.morphologyEx(mix, cv2.MORPH_OPEN,  k, iterations=1)
    cnts, _ = cv2.findContours(mix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    if not cnts:
        return mask
    big = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(big)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def _make_frame_strips(raw_card_mask, bottom_exclude=0.22, inset_ratio=0.012, thick_ratio=0.032):
    h, w = raw_card_mask.shape[:2]
    ys, xs = np.where(raw_card_mask > 0)
    if xs.size == 0:
        return np.zeros_like(raw_card_mask), {}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cw, ch = (x1 - x0 + 1), (y1 - y0 + 1)
    inset = max(2, int(round(min(cw, ch) * inset_ratio)))
    thick = max(2, int(round(min(cw, ch) * thick_ratio)))
    y_cap = y0 + int(round((1.0 - bottom_exclude) * ch))

    def clamp(a, lo, hi): return max(lo, min(hi, a))
    top = (clamp(x0 + inset, 0, w - 1),
           clamp(y0 + inset, 0, h - 1),
           clamp(x1 - inset - (x0 + inset), 0, w - 1),
           clamp((y0 + inset + thick) - (y0 + inset), 0, h - 1))
    left = (clamp(x0 + inset, 0, w - 1),
            clamp(y0 + inset, 0, h - 1),
            clamp((x0 + inset + thick) - (x0 + inset), 0, w - 1),
            clamp((y_cap - inset) - (y0 + inset), 0, h - 1))
    right = (clamp(x1 - inset - thick + 1, 0, w - 1),
             clamp(y0 + inset, 0, h - 1),
             clamp(thick, 0, w - 1),
             clamp((y_cap - inset) - (y0 + inset), 0, h - 1))

    union = np.zeros_like(raw_card_mask, np.uint8)
    bands_info = {"top": top, "left": left, "right": right}
    for (x, y, ww, hh) in (top, left, right):
        if ww > 1 and hh > 1:
            union[y:y + hh, x:x + ww] = 255
    return union, bands_info

def _edge_metrics(Hdeg, S, V, h, w, side,
                  ring_ratio=ELITE_PARAMS["ring_ratio"],
                  edge_margin=ELITE_PARAMS["edge_margin"],
                  sat_min=None, val_min=None, hue_bins=ELITE_PARAMS["hue_bins"],
                  edge_nonzero_bins_min=ELITE_PARAMS["edge_nonzero_bins_min"],
                  edge_peak_max=ELITE_PARAMS["edge_peak_max"],
                  edge_cstd_min=None,
                  edge_count_min=ELITE_PARAMS["edge_count_min"]):
    if sat_min is None: sat_min = ELITE_PARAMS["sat_min"]
    if val_min is None: val_min = ELITE_PARAMS["val_min"]
    if edge_cstd_min is None: edge_cstd_min = ELITE_PARAMS["edge_cstd_min"]

    thick_x = max(2, int(w * ring_ratio))
    thick_y = max(2, int(h * ring_ratio))
    mx = max(1, int(w * edge_margin))
    my = max(1, int(h * edge_margin))
    if side == "top":
        rr, cc = slice(0, thick_y), slice(mx, w - mx)
    elif side == "bottom":
        rr, cc = slice(h - thick_y, h), slice(mx, w - mx)
    elif side == "left":
        rr, cc = slice(my, h - my), slice(0, thick_x)
    else:
        rr, cc = slice(my, h - my), slice(w - thick_x, w)

    m = (S[rr, cc] >= sat_min) & (V[rr, cc] >= val_min)
    hue = Hdeg[rr, cc][m].astype(np.float32)
    if hue.size == 0:
        return dict(ok=False, nonzero=0, peak=1.0, cstd=0.0, n=0)

    hist, _ = np.histogram(hue, bins=hue_bins, range=(0, 360))
    total = float(hist.sum())
    peak = float(hist.max() / max(1.0, total))
    nonzero = int((hist > 0).sum())

    ang = np.deg2rad(hue)
    s, c = np.sin(ang).sum(), np.cos(ang).sum()
    R = np.sqrt(s*s + c*c) / max(1.0, hue.size)
    R = float(np.clip(R, 1e-6, 0.999999))
    cstd = float(np.degrees(np.sqrt(-2.0 * np.log(R))))

    ok = (hue.size >= edge_count_min and
          nonzero >= edge_nonzero_bins_min and
          peak <= edge_peak_max and
          cstd >= float(edge_cstd_min))
    return dict(ok=ok, nonzero=nonzero, peak=peak, cstd=cstd, n=int(hue.size))

def _classify_ring(card_bgr, ring_mask, bands_info):
    if ring_mask.sum() < 200:
        return False, 0.0, {"reason": "no_ring"}
    hsv = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2HSV)
    Hdeg = hsv[..., 0].astype(np.float32) * 2.0
    S = hsv[..., 1]; V = hsv[..., 2]
    h, w = card_bgr.shape[:2]

    mt = _edge_metrics(Hdeg, S, V, h, w, "top")
    mb = _edge_metrics(Hdeg, S, V, h, w, "bottom")
    ml = _edge_metrics(Hdeg, S, V, h, w, "left")
    mr = _edge_metrics(Hdeg, S, V, h, w, "right")

    ok_top, ok_bottom, ok_left, ok_right = mt["ok"], mb["ok"], ml["ok"], mr["ok"]
    enough_edges = ((ok_top and ok_bottom) or (ok_left and ok_right) or
                    (sum([ok_top, ok_bottom, ok_left, ok_right]) >= 3))
    if not enough_edges:
        return False, 0.0, {"gate": "edge_fail"}

    score = float((mt["cstd"] + mb["cstd"] + ml["cstd"] + mr["cstd"]) / 4.0)
    elite = score >= 60.0
    return elite, score, {"score": round(score, 1)}

def is_elite_card(card_img_bgr):
    if card_img_bgr is None or card_img_bgr.size == 0:
        return False, {"reason": "empty"}
    raw_card_mask = _segment_card_mask(card_img_bgr)
    ring_mask, bands_info = _make_frame_strips(raw_card_mask, bottom_exclude=0.22)
    if ring_mask.sum() == 0:
        return False, {"reason": "no_card"}
    elite, score, dbg = _classify_ring(card_img_bgr, ring_mask, bands_info)
    dbg["score"] = round(float(score), 3)
    return elite, dbg

def extract_card_for_index(index, card_crop):
    try:
        if isinstance(card_crop, Image.Image):
            rgb = np.array(card_crop)
        else:
            rgb = card_crop
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 and rgb.shape[2] == 3 else rgb
        elite, _ = is_elite_card(bgr)
        processed = preprocess_image(card_crop)
        gens, name, series = extract_generation_with_easyocr(processed)
        return index, (gens, name, series, elite)
    except Exception as e:
        logging.warning(f"⚠️ Failed to process card index {index}: {e}")
        return index, ({}, "", "", False)

def extract_card_generations(image: Image.Image):
    card_width = image.width // 3
    card_info = {}
    def _crop_card(i: int) -> Image.Image:
        left = i * card_width
        right = left + card_width if i < 2 else image.width
        return image.crop((left, 0, right, image.height))
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_card_for_index, i, _crop_card(i)) for i in range(3)]
        for fut in futures:
            try:
                i, (gens, name, series, elite) = fut.result()
                card_info[i] = {
                    "generations": gens or {},
                    "name": name or "",
                    "series": series or "",
                    "elite": bool(elite),
                }
                logging.info(f"✅ Card {i}: Gen={list((gens or {}).keys()) or '∅'}, Name='{name or ''}', Series='{series or ''}', Elite={elite}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to extract data for a card: {e}")
                idx = len(card_info)
                card_info[idx] = {"generations": {}, "name": "", "series": "", "elite": False}
    return card_info

# --------------------------
# Bot Manager
# --------------------------
class SofiBotManager:
    SOFI_BOT_ID = "853629533855809596"

    def __init__(self, cfg_path=".env"):
        self.cfg_path = cfg_path
        self.http = _create_http_session()

        # state
        self.last_processed_time_lock = Lock()
        self.pending_claim = {"message_id": None, "timestamp": None, "user_id": None, "triggered": False}
        self.last_processed_time = 0
        self.PROCESS_COOLDOWN_SECONDS = 240

        self.stop_event = threading.Event()
        self.bot = None
        self.bot_thread = None
        self.sd_thread = None
        self.ka_thread = None
        self.watchdog_thread = None
        self.claim_timeout_thread = None

        self.bot_identity_logged = False
        self.sent_initial_sd = False
        self._run_token = 0

        # Connection health
        self.gateway_ready = False
        self.last_gateway_event = 0.0

        # Reboot guard
        self._reboot_lock = threading.Lock()
        self._last_reboot_ts = 0.0
        self.REBOOT_MIN_INTERVAL = 20.0  # avoid reboot loops

        # Load env
        load_dotenv(self.cfg_path)
        self.TOKEN = os.getenv("DISCORD_TOKEN", "")
        self.GUILD_ID = os.getenv("GUILD_ID", "")
        self.CHANNEL_ID = os.getenv("CHANNEL_ID", "")
        self.USER_ID = os.getenv("USER_ID", "")
        self.SD_INTERVAL_SEC = int(os.getenv("SD_INTERVAL_SEC", "480"))
        self.USE_GPU = (os.getenv("OCR_USE_GPU", "0") == "1")

        # Interaction timing & retries
        self.CLAIM_CONFIRM_TIMEOUT = int(os.getenv("CLAIM_CONFIRM_TIMEOUT", "15"))
        self.POST_ACORN_NORMAL_DELAY = float(os.getenv("POST_ACORN_NORMAL_DELAY", "0.8"))

        # Mode: "smart" or "normal"
        self.MODE = os.getenv("MODE", "smart").strip().lower()
        if self.MODE not in ("smart", "normal"):
            self.MODE = "smart"

        # Normal mode priorities (ordered 1..3)
        self.NORM_P1 = os.getenv("NORM_P1", "high_likes").strip().lower()
        self.NORM_P2 = os.getenv("NORM_P2", "low_gen").strip().lower()
        self.NORM_P3 = os.getenv("NORM_P3", "series").strip().lower()
        self._normalize_normal_priorities()

        # Optional series name for matching/bonus
        self.series_preference = os.getenv("SERIES_NAME", "").strip()

        # Adaptive scoring tunables (Smart mode)
        self.SCORE_BONUS_SERIES = float(os.getenv("SCORE_BONUS_SERIES", "0.12"))
        self.SCORE_BONUS_NOGEN  = float(os.getenv("SCORE_BONUS_NOGEN",  "0.03"))
        self.T_LIKE = int(os.getenv("T_LIKE", "10"))
        self.T_GEN  = int(os.getenv("T_GEN",  "40"))
        self.WL_MIN = float(os.getenv("WL_MIN", "0.15"))
        self.WL_SPAN = float(os.getenv("WL_SPAN", "0.70"))
        self.LIKES_LOG_DAMP = (os.getenv("LIKES_LOG_DAMP", "1") == "1")

        # Watchdog timeouts
        self.WATCHDOG_TIMEOUT = int(os.getenv("WATCHDOG_TIMEOUT", "600"))     # no events → reboot
        self.GATEWAY_READY_TIMEOUT = int(os.getenv("GATEWAY_READY_TIMEOUT", "120"))  # no READY after connect → reboot

        # Track connect start to enforce READY timeout
        self.gateway_start_time = 0.0

    def _normalize_normal_priorities(self):
        allowed = ["high_likes", "low_gen", "no_gen", "series"]
        picks = [p for p in [self.NORM_P1, self.NORM_P2, self.NORM_P3] if p in allowed]
        # fill with remaining unique options
        for a in allowed:
            if a not in picks:
                picks.append(a)
        self.NORM_P1, self.NORM_P2, self.NORM_P3 = picks[:3]

    # --------- Reboot (stop -> start) ---------
    def reboot_bot(self, reason: str):
        """Hard reboot: stop everything, then start with fresh gateway."""
        now = time.time()
        if (now - self._last_reboot_ts) < self.REBOOT_MIN_INTERVAL:
            logging.warning(f"⏳ Reboot suppressed (too soon). Reason: {reason}")
            return
        if not self._reboot_lock.acquire(blocking=False):
            return
        try:
            self._last_reboot_ts = now
            logging.error(f"🔄 REBOOTING BOT — Reason: {reason}")
            self.stop(internal=True)
            time.sleep(1.0)
            self.start()
        finally:
            self._reboot_lock.release()

    # --------- Threads ---------
    def message_watchdog(self):
        while not self.stop_event.is_set():
            time.sleep(5)
            if not self.bot or not getattr(self.bot, "gateway", None):
                continue
            last = self.last_gateway_event or (time.time() - 30)
            elapsed = time.time() - last
            # If no Dispatch events for a long time assume dead session and reboot
            if elapsed > self.WATCHDOG_TIMEOUT:
                self.reboot_bot(f"No gateway events for {elapsed:.1f}s")
                return

            # If we're still not READY after connecting for too long, reboot
            if (not self.gateway_ready) and self.gateway_start_time:
                since_conn = time.time() - self.gateway_start_time
                if since_conn > self.GATEWAY_READY_TIMEOUT:
                    self.reboot_bot("Gateway did not reach READY in time")
                    return

    def reset_claim_if_timed_out(self):
        while not self.stop_event.is_set():
            time.sleep(0.5)
            if self.pending_claim["triggered"]:
                elapsed = time.time() - (self.pending_claim["timestamp"] or 0)
                if elapsed > self.CLAIM_CONFIRM_TIMEOUT:
                    logging.warning("⚠️ No Sofi confirmation received. Resetting claim trigger (timeout).")
                    self.pending_claim["triggered"] = False

    def periodic_sd_sender(self):
        run_token = self._run_token
        while not self.stop_event.is_set():
            base = float(self.SD_INTERVAL_SEC)
            wait_time = max(300.0, base + random.uniform(0, 10))
            logging.info(f"⏳ Waiting {wait_time:.1f}s before sending next 'sd'…")
            if self.stop_event.wait(wait_time): break
            if run_token != self._run_token: return

            # REBOOT if gateway not READY when we want to send 'sd'
            ready = (self.gateway_ready and self.bot and getattr(self.bot.gateway, "session_id", None))
            if not ready:
                logging.warning("⚠️ Gateway not READY at 'sd' send time. Initiating REBOOT.")
                self.reboot_bot("Gateway not READY when sending sd")
                return

            try:
                self.bot.sendMessage(self.CHANNEL_ID, "sd")
                logging.info("📤 Sent 'sd' command.")
            except Exception:
                logging.exception("⚠️ Failed to send 'sd'")

    def keep_alive(self):
        same_latency_count = 0
        self._prev_latency = None
        while not self.stop_event.is_set():
            if self.stop_event.wait(30): break
            try:
                gw = getattr(self.bot, "gateway", None)
                latency = gw.latency if gw else None
                if latency is None:
                    same_latency_count += 1
                else:
                    prev = getattr(self, "_prev_latency", None)
                    same_latency_count = same_latency_count + 1 if prev == latency else 0
                    self._prev_latency = latency
                if same_latency_count >= 3:
                    self.reboot_bot("Gateway appears frozen (latency unchanged)")
                    return
            except Exception as e:
                logging.error(f"keep_alive error: {e}")

    # --------- Button click helpers ---------
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
            logging.info(f"➡️  Clicked button {custom_id}")
        except Exception as e:
            logging.warning(f"Exception during click: {e}")

    def _click_normal_with_retry(self, button_id, channel_id, guild_id, m, reason: str):
        """Click a normal-claim button, wait for confirmation up to CLAIM_CONFIRM_TIMEOUT; retry once if needed."""
        def do_click():
            try:
                self.click_discord_button(button_id, channel_id, guild_id, m)
                self.pending_claim["timestamp"] = time.time()
                self.pending_claim["triggered"] = True
                self.pending_claim["user_id"] = self.USER_ID
                logging.info(f"🔘 NORMAL click: {reason}")
            except Exception as e:
                logging.warning(f"Normal click failed: {e}")

        # 1st attempt
        do_click()
        t0 = time.time()
        while self.pending_claim["triggered"] and (time.time() - t0) < self.CLAIM_CONFIRM_TIMEOUT:
            time.sleep(0.25)

        if self.pending_claim["triggered"]:
            logging.warning("⏱️ No confirmation yet. Retrying normal click once …")
            self.pending_claim["triggered"] = False
            time.sleep(0.4)
            do_click()
            t1 = time.time()
            while self.pending_claim["triggered"] and (time.time() - t1) < self.CLAIM_CONFIRM_TIMEOUT:
                time.sleep(0.25)
            if self.pending_claim["triggered"]:
                logging.error("❌ Normal claim still not confirmed after retry.")
                self.pending_claim["triggered"] = False

    # --------- Core claim logic ---------
    def click_bouquet_then_best_from_image(self, pil_image, buttons_components, image_received_time, channel_id, guild_id, m):
        import numpy as np
        def blocked(ignore=False) -> bool:
            if ignore: return False
            now = time.time()
            with self.last_processed_time_lock:
                cooldown_active = self.last_processed_time and (now - self.last_processed_time < self.PROCESS_COOLDOWN_SECONDS)
                pending_active  = self.pending_claim.get("triggered", False)
                if pending_active:
                    logging.info("⏳ Skipping — waiting previous claim confirmation."); return True
                if cooldown_active:
                    logging.info("⏳ Skipping — cooldown not expired."); return True
            return False

        # Parse claim buttons; ignore URL buttons; ignore beyond first 3.
        pos_buttons = []
        for row in buttons_components:
            for btn in row.get("components", []):
                if btn.get("type") != 2 or not btn.get("custom_id"):
                    continue
                label = str(btn.get("label") or "")
                mlikes = re.search(r"(\d+)", label.replace(",", ""))
                likes = int(mlikes.group(1)) if mlikes else None  # None -> acorn/event button
                pos_buttons.append({"id": btn["custom_id"], "likes": likes, "label": label})
        pos_buttons = pos_buttons[:3]

        if not pos_buttons:
            logging.warning("No claimable buttons found (ignoring link buttons). Abort."); return

        acorn_positions  = [i for i, b in enumerate(pos_buttons) if b["likes"] is None]
        normal_positions = [i for i, b in enumerate(pos_buttons) if b["likes"] is not None]

        # OCR/gen extraction
        card_info = extract_card_generations(pil_image)

        # 1) Acorns first (claim all)
        acorn_clicked = False
        if acorn_positions:
            for idx, pos in enumerate(acorn_positions, start=1):
                try:
                    self.click_discord_button(pos_buttons[pos]["id"], channel_id, guild_id, m)
                    logging.info(f"🌰 Claimed acorn at position {pos+1} (#{idx}).")
                    acorn_clicked = True
                    time.sleep(0.25)
                except Exception as e:
                    logging.warning(f"Acorn click failed at pos {pos+1}: {e}")

        # If ALL are acorns, we’re done
        if not normal_positions:
            logging.info("🌰 All buttons were acorns — no normal claims to process.")
            return

        # Pause after acorns so next click is seen
        if acorn_clicked:
            time.sleep(self.POST_ACORN_NORMAL_DELAY)

        # Assemble cards
        cards = []
        for i in range(3):
            info = card_info.get(i, {}) or {}
            gens = info.get("generations", {}) or {}
            min_gen = min(gens.values()) if gens else None
            like_val = pos_buttons[i]["likes"] if i < len(pos_buttons) else 0
            cards.append({
                "pos": i,
                "likes": like_val if like_val is not None else 0,
                "gens": gens,
                "min_gen": min_gen,
                "has_gen": bool(gens),
                "name": info.get("name") or "",
                "series": info.get("series") or "",
                "elite": bool(info.get("elite", False)),
            })

        # Consider only normals after acorns
        normals = [cards[p] for p in normal_positions]

        # 2) ELITE fast path
        elite_pos = next((c["pos"] for c in normals if c["elite"]), None)
        if elite_pos is not None:
            if not blocked(ignore=acorn_clicked):
                reason = f"pos {elite_pos+1} | ELITE"
                self._click_normal_with_retry(pos_buttons[elite_pos]["id"], channel_id, guild_id, m, reason)
            return

        # 3) ABS GEN < 10 fast path
        ultra = [c for c in normals if c["min_gen"] is not None and c["min_gen"] < 10]
        if ultra:
            ultra.sort(key=lambda c: (c["min_gen"], -c["likes"], c["pos"]))
            chosen = ultra[0]
            if not blocked(ignore=acorn_clicked):
                reason = f"pos {chosen['pos']+1} | ABS gen<10 | G{chosen['min_gen']} | likes={chosen['likes']}"
                self._click_normal_with_retry(pos_buttons[chosen["pos"]]["id"], channel_id, guild_id, m, reason)
            return

        # 4) Mode-specific choice
        mode = self.MODE

        # ---- Shared helpers ----
        series_key = (self.series_preference or "").strip().lower()
        likes_values = [c["likes"] for c in normals]
        max_likes = max(likes_values) if likes_values else 0
        min_likes = min(likes_values) if likes_values else 0
        like_gap  = max_likes - min_likes

        gen_vals = [c["min_gen"] for c in normals if c["min_gen"] is not None]
        gen_gap  = (max(gen_vals) - min(gen_vals)) if gen_vals else 0
        min_seen_gen = min(gen_vals) if gen_vals else None
        max_seen_gen = max(gen_vals) if gen_vals else None

        def series_filter(pool):
            if not series_key:
                return []
            return [c for c in pool if series_key in (c["series"] or "").lower()]

        def tiebreak(pool):
            # final deterministic tie-breaker
            pool.sort(key=lambda c: (
                c["min_gen"] if c["min_gen"] is not None else 10**9,  # prefer lower gen if known
                -c["likes"],                                          # then higher likes
                c["pos"]
            ))
            return pool[0]

        # ---------- Enhanced logging: candidate table ----------
        def fmt(c):
            g = (f"G{c['min_gen']}" if c["min_gen"] is not None else "∅")
            el = "E" if c["elite"] else "-"
            sc = f"{c.get('score', 0):.3f}" if "score" in c else "---"
            return f"{c['pos']+1:^3} | {g:>4} | {c['likes']:>4} | {el:^1} | {sc:>6}"

        header = "pos | gen  | like | E | score "
        logging.info("📋 Candidates (normals):\n  " + header + "\n  " + "-"*len(header))
        for c in normals:
            logging.info("  " + fmt(c))

        if mode == "smart":
            # ===== SMART MODE: ADAPTIVE SCORING =====
            T_LIKE = self.T_LIKE
            T_GEN  = self.T_GEN
            WL_MIN = self.WL_MIN
            WL_SPAN = self.WL_SPAN
            LIKES_LOG_DAMP = self.LIKES_LOG_DAMP

            L = like_gap / (like_gap + T_LIKE) if like_gap > 0 else 0.0
            Gtight = 1.0 - (gen_gap / (gen_gap + T_GEN)) if gen_gap > 0 else 1.0

            wL = max(0.1, min(0.9, WL_MIN + WL_SPAN * (0.5 * (L + Gtight))))
            wG = 1.0 - wL
            logging.info(f"⚖️ SMART weights → likes={wL:.2f}, gen={wG:.2f}  (gap: likes={like_gap}, gen={gen_gap})")

            def score(c):
                # Likes contribution (log-damped by default)
                if max_likes <= 0:
                    like_norm = 0.0
                else:
                    like_norm = (np.log1p(c["likes"]) / max(1e-9, np.log1p(max_likes))) if LIKES_LOG_DAMP else (c["likes"] / max_likes)
                # Gen contribution: LOWER gen => HIGHER score
                if (c["min_gen"] is None or
                    min_seen_gen is None or max_seen_gen is None or max_seen_gen == min_seen_gen):
                    gen_norm = 0.0
                    nog_bonus = self.SCORE_BONUS_NOGEN if not c["has_gen"] else 0.0
                else:
                    gen_norm = (max_seen_gen - c["min_gen"]) / max(1.0, (max_seen_gen - min_seen_gen))
                    nog_bonus = 0.0
                series_bonus = self.SCORE_BONUS_SERIES if (series_key and series_key in (c["series"] or "").lower()) else 0.0
                return wL * like_norm + wG * gen_norm + series_bonus + nog_bonus

            for c in normals:
                c["score"] = score(c)

            # Re-log with scores
            logging.info("📊 SMART scores:")
            for c in normals:
                logging.info("  " + fmt(c))

            normals.sort(key=lambda c: (-c["score"],
                                        c["min_gen"] if c["min_gen"] is not None else 10**9,
                                        -c["likes"], c["pos"]))
            chosen = normals[0]
            if not blocked(ignore=acorn_clicked):
                reason = (f"pos {chosen['pos']+1} | SMART score={chosen['score']:.3f} "
                          f"| G{chosen['min_gen'] if chosen['min_gen'] is not None else '∅'} "
                          f"| likes={chosen['likes']}")
                self._click_normal_with_retry(pos_buttons[chosen["pos"]]["id"], channel_id, guild_id, m, reason)
            return

        # ---------- NORMAL MODE ----------
        def select_by_priority(pool, p1, p2, p3):
            def apply(pref, candidates):
                if not candidates:
                    return []
                if pref == "high_likes":
                    mx = max([c["likes"] for c in candidates])
                    return [c for c in candidates if c["likes"] == mx]
                elif pref == "low_gen":
                    gen_vals = [c["min_gen"] for c in candidates if c["min_gen"] is not None]
                    if not gen_vals:
                        return []
                    mn = min(gen_vals)
                    return [c for c in candidates if c["min_gen"] == mn]
                elif pref == "no_gen":
                    return [c for c in candidates if (c["min_gen"] is None or not c["has_gen"])]
                elif pref == "series":
                    ser = series_filter(candidates)
                    return ser if ser else []
                else:
                    return candidates

            logging.info(f"🎯 NORMAL priorities: {p1} → {p2} → {p3}")

            pool1 = apply(p1, pool)
            if pool1:
                logging.info(f"  • P1({p1}) matched {len(pool1)} -> {[(c['pos']+1) for c in pool1]}")
                return tiebreak(pool1)
            logging.info(f"  • P1({p1}) no match, trying P2 …")

            pool2 = apply(p2, pool)
            if pool2:
                logging.info(f"  • P2({p2}) matched {len(pool2)} -> {[(c['pos']+1) for c in pool2]}")
                return tiebreak(pool2)
            logging.info(f"  • P2({p2}) no match, trying P3 …")

            pool3 = apply(p3, pool)
            if pool3:
                logging.info(f"  • P3({p3}) matched {len(pool3)} -> {[(c['pos']+1) for c in pool3]}")
                return tiebreak(pool3)

            logging.info("  • No priority matched; applying deterministic fallback.")
            return tiebreak(pool)

        chosen = select_by_priority(normals, self.NORM_P1, self.NORM_P2, self.NORM_P3)
        if not blocked(ignore=acorn_clicked):
            reason = (f"pos {chosen['pos']+1} | NORMAL ({self.NORM_P1} > {self.NORM_P2} > {self.NORM_P3}) "
                      f"| G{chosen['min_gen'] if chosen['min_gen'] is not None else '∅'} "
                      f"| likes={chosen['likes']}")
            self._click_normal_with_retry(pos_buttons[chosen["pos"]]["id"], channel_id, guild_id, m, reason)

    # --------- Gateway handlers ---------
    def _install_handlers(self):
        @self.bot.gateway.command
        def on_any_event(resp):
            try:
                if hasattr(resp, "raw") and resp.raw.get("op") == 0:  # Dispatch
                    self.last_gateway_event = time.time()
            except Exception:
                pass

        @self.bot.gateway.command
        def on_ready(resp):
            if not resp.event.ready: return
            user = resp.parsed.auto().get("user")
            if user:
                logging.info(f"✅ Gateway READY as {user['username']}#{user['discriminator']}")
            self.gateway_ready = True
            self.last_gateway_event = time.time()
            self.gateway_start_time = 0.0
            if not self.sent_initial_sd:
                for _ in range(6):
                    try:
                        if self.bot and self.bot.gateway.session_id:
                            self.bot.sendMessage(self.CHANNEL_ID, "sd")
                            self.sent_initial_sd = True
                            logging.info("📤 Sent 'sd' command.")
                            break
                    except Exception:
                        logging.exception("⚠️ Failed to send initial 'sd'")
                    time.sleep(0.5)

        @self.bot.gateway.command
        def on_message(resp):
            if not hasattr(resp, "raw") or resp.raw.get("t") != "MESSAGE_CREATE":
                return
            self.last_gateway_event = time.time()
            data = resp.raw["d"]
            m = resp.parsed.auto()
            author_id = str(data.get("author", {}).get("id"))
            channel_id = str(data.get("channel_id"))
            guild_id = str(data.get("guild_id")) if data.get("guild_id") else None
            content = data.get("content", "")

            if author_id != SofiBotManager.SOFI_BOT_ID:
                return
            if self.GUILD_ID and guild_id != self.GUILD_ID:
                return

            # Confirmation detection (either "grabbed" or "fought off")
            if self.pending_claim.get("triggered"):
                expected_grab = f"<@{self.pending_claim['user_id']}> **grabbed** the"
                expected_fight = f"<@{self.pending_claim['user_id']}> fought off"
                if content.startswith(expected_grab) or content.startswith(expected_fight):
                    logging.info(f"✅ Claim confirmed: {content[:120]}…")
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
                    pil_image = pil_from_url_or_none(self.http, attachment_url)
                    if pil_image is None:  # fetch failed
                        return
                    image_received_time = time.time()
                    self.click_bouquet_then_best_from_image(pil_image, components, image_received_time, channel_id, guild_id, m)
                except Exception as e:
                    logging.warning(f"Failed to process image: {e}")
                    if "NameResolutionError" in str(e) or "getaddrinfo failed" in str(e).lower():
                        logging.warning("Hint: DNS failed. Try ipconfig /flushdns, netsh winsock reset, or change DNS.")

            for att in attachments:
                fn = att.get("filename", "")
                if any(fn.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
                    t = threading.Thread(target=process_sofi_drop, args=(att.get("url"), components, channel_id, data.get("guild_id"), m), daemon=True)
                    t.start()

    # --------- Lifecycle ---------
    def start(self):
        global READER
        if self.bot_thread and self.bot_thread.is_alive():
            logging.info("ℹ️ Bot already running."); return

        # OCR boot
        logging.info(f"🔎 Initializing EasyOCR (gpu={self.USE_GPU}) …")
        if READER is None:
            READER = easyocr.Reader(["en"], gpu=self.USE_GPU, verbose=False)
            _ = READER.readtext(np.zeros((100, 100, 3), dtype=np.uint8), detail=0)

        # Client
        self.bot = discum.Client(token=self.TOKEN, log=False)
        self._install_handlers()
        self.stop_event.clear()
        self.pending_claim["user_id"] = self.USER_ID
        self.bot_identity_logged = False
        self.sent_initial_sd = False
        self.gateway_ready = False
        self.last_gateway_event = 0.0
        self.gateway_start_time = time.time()
        self._run_token += 1

        # Threads
        if not (self.sd_thread and self.sd_thread.is_alive()):
            self.sd_thread = threading.Thread(target=self.periodic_sd_sender, daemon=True); self.sd_thread.start()
        if not (self.ka_thread and self.ka_thread.is_alive()):
            self.ka_thread = threading.Thread(target=self.keep_alive, daemon=True); self.ka_thread.start()
        if not (self.watchdog_thread and self.watchdog_thread.is_alive()):
            self.watchdog_thread = threading.Thread(target=self.message_watchdog, daemon=True); self.watchdog_thread.start()
        if not (self.claim_timeout_thread and self.claim_timeout_thread.is_alive()):
            self.claim_timeout_thread = threading.Thread(target=self.reset_claim_if_timed_out, daemon=True); self.claim_timeout_thread.start()

        def run_gateway():
            logging.info("🔌 Connecting to Discord gateway …")
            try:
                self.bot.gateway.run(auto_reconnect=True)
            except Exception as e:
                logging.critical(f"Failed to connect to Discord gateway: {e}")
                logging.critical("Check DISCORD_TOKEN, internet, or Discord status.")
        self.bot_thread = threading.Thread(target=run_gateway, daemon=True); self.bot_thread.start()

    def stop(self, internal=False):
        logging.info("🛑 Stopping bot …")
        self.stop_event.set()
        try:
            if self.bot and self.bot.gateway:
                self.bot.gateway.close()
        except Exception:
            pass
        for tname in ("sd_thread", "ka_thread", "watchdog_thread", "claim_timeout_thread"):
            t = getattr(self, tname, None)
            if t and t.is_alive():
                try: t.join(timeout=2.0)
                except Exception: pass
                setattr(self, tname, None)
        self.sd_thread = self.ka_thread = self.watchdog_thread = self.claim_timeout_thread = None
        self.gateway_ready = False
        self.gateway_start_time = 0.0
        # Join gateway thread
        if self.bot_thread and self.bot_thread.is_alive():
            try: self.bot_thread.join(timeout=3.0)
            except Exception: pass
        self.bot_thread = None
        logging.info("✅ Bot stopped.")
        if not internal:
            logging.info("You can Apply settings and the bot will auto-restart if needed.")

    def save_env(self):
        data = {
            "DISCORD_TOKEN": self.TOKEN,
            "GUILD_ID": self.GUILD_ID,
            "CHANNEL_ID": self.CHANNEL_ID,
            "USER_ID": self.USER_ID,
            "SD_INTERVAL_SEC": str(self.SD_INTERVAL_SEC),
            "OCR_USE_GPU": "1" if self.USE_GPU else "0",
            "CLAIM_CONFIRM_TIMEOUT": str(self.CLAIM_CONFIRM_TIMEOUT),
            "POST_ACORN_NORMAL_DELAY": str(self.POST_ACORN_NORMAL_DELAY),
            "MODE": self.MODE,  # smart | normal
            "NORM_P1": self.NORM_P1,
            "NORM_P2": self.NORM_P2,
            "NORM_P3": self.NORM_P3,
            "SERIES_NAME": self.series_preference,
            "SCORE_BONUS_SERIES": str(self.SCORE_BONUS_SERIES),
            "SCORE_BONUS_NOGEN": str(self.SCORE_BONUS_NOGEN),
            "T_LIKE": str(self.T_LIKE),
            "T_GEN": str(self.T_GEN),
            "WL_MIN": str(self.WL_MIN),
            "WL_SPAN": str(self.WL_SPAN),
            "LIKES_LOG_DAMP": "1" if self.LIKES_LOG_DAMP else "0",
            "WATCHDOG_TIMEOUT": str(self.WATCHDOG_TIMEOUT),
            "GATEWAY_READY_TIMEOUT": str(self.GATEWAY_READY_TIMEOUT),
        }
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            for k, v in data.items():
                f.write(f"{k}={v}\n")
        logging.info(f"💾 Saved settings to {self.cfg_path}")

# --------------------------
# Tkinter UI (Auto-start; with Mode & Preferences)
# --------------------------
class TextHandler(logging.Handler):
    def __init__(self, widget: ScrolledText):
        super().__init__()
        self.widget = widget
        self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    def emit(self, record):
        msg = self.format(record) + "\n"
        level = record.levelno
        if level >= logging.CRITICAL: tag = "CRITICAL"
        elif level >= logging.ERROR: tag = "ERROR"
        elif level >= logging.WARNING: tag = "WARNING"
        elif level >= logging.INFO: tag = "INFO"
        else: tag = "DEBUG"
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
        self.title("Sofi Farming Bot — Auto Start (Mode & Preferences)")
        self.geometry("1000x780")
        self.minsize(920, 660)
        self._apply_in_progress = False
        self.manager = SofiBotManager()
        self._build_ui()
        self._wire_logging()
        self._load_from_manager()
        # Auto-start immediately
        self.after(150, self._auto_start)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ---------- UI ----------
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("aqua")
        except Exception:
            pass

        pad = {"padx": 8, "pady": 6}

        # Connection
        frm_conn = ttk.LabelFrame(self, text="Connection & Runtime (.env)")
        frm_conn.pack(fill="x", **pad)

        ttk.Label(frm_conn, text="Discord Token").grid(row=0, column=0, sticky="w")
        self.var_token = tk.StringVar(value="")
        self.ent_token = ttk.Entry(frm_conn, textvariable=self.var_token, width=64, show="•")
        self.ent_token.grid(row=0, column=1, sticky="ew", columnspan=3)
        self.show_token = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_conn, text="Show", variable=self.show_token, command=self._toggle_token).grid(row=0, column=4, sticky="w")

        ttk.Label(frm_conn, text="Guild ID").grid(row=1, column=0, sticky="w")
        self.var_guild = tk.StringVar(value="")
        ttk.Entry(frm_conn, textvariable=self.var_guild, width=26).grid(row=1, column=1, sticky="w")

        ttk.Label(frm_conn, text="Channel ID").grid(row=1, column=2, sticky="w")
        self.var_channel = tk.StringVar(value="")
        ttk.Entry(frm_conn, textvariable=self.var_channel, width=26).grid(row=1, column=3, sticky="w")

        ttk.Label(frm_conn, text="Your User ID").grid(row=2, column=0, sticky="w")
        self.var_user = tk.StringVar(value="")
        ttk.Entry(frm_conn, textvariable=self.var_user, width=26).grid(row=2, column=1, sticky="w")

        ttk.Label(frm_conn, text="'sd' Interval (sec)").grid(row=2, column=2, sticky="w")
        self.var_interval = tk.IntVar(value=480)
        ttk.Spinbox(frm_conn, from_=60, to=3600, increment=10, textvariable=self.var_interval, width=10).grid(row=2, column=3, sticky="w")

        self.var_gpu = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_conn, text="Use GPU for OCR (advanced)", variable=self.var_gpu).grid(row=3, column=1, sticky="w", pady=(2, 8))

        for c in (1, 3):
            frm_conn.grid_columnconfigure(c, weight=1)

        # Mode & Preferences
        frm_prefs = ttk.LabelFrame(self, text="Mode & Preferences")
        frm_prefs.pack(fill="x", **pad)

        ttk.Label(frm_prefs, text="Mode").grid(row=0, column=0, sticky="w")
        self.var_mode = tk.StringVar(value="smart")
        self.cmb_mode = ttk.Combobox(frm_prefs, textvariable=self.var_mode, values=("smart", "normal"), state="readonly", width=12)
        self.cmb_mode.grid(row=0, column=1, sticky="w", padx=4)
        self.cmb_mode.bind("<<ComboboxSelected>>", lambda e: self._toggle_normal_controls())

        ttk.Label(frm_prefs, text="Series (optional)").grid(row=0, column=2, sticky="w")
        self.var_series = tk.StringVar(value="")
        ttk.Entry(frm_prefs, textvariable=self.var_series, width=28).grid(row=0, column=3, columnspan=2, sticky="w", padx=4)

        opts = ("high_likes", "low_gen", "no_gen", "series")
        ttk.Label(frm_prefs, text="Normal priority 1").grid(row=1, column=0, sticky="w")
        self.var_p1 = tk.StringVar(value="high_likes")
        self.cmb_p1 = ttk.Combobox(frm_prefs, textvariable=self.var_p1, values=opts, state="readonly", width=12)
        self.cmb_p1.grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(frm_prefs, text="Normal priority 2").grid(row=1, column=2, sticky="w")
        self.var_p2 = tk.StringVar(value="low_gen")
        self.cmb_p2 = ttk.Combobox(frm_prefs, textvariable=self.var_p2, values=opts, state="readonly", width=12)
        self.cmb_p2.grid(row=1, column=3, sticky="w", padx=4)

        ttk.Label(frm_prefs, text="Normal priority 3").grid(row=1, column=4, sticky="w")
        self.var_p3 = tk.StringVar(value="series")
        self.cmb_p3 = ttk.Combobox(frm_prefs, textvariable=self.var_p3, values=opts, state="readonly", width=12)
        self.cmb_p3.grid(row=1, column=5, sticky="w")

        # Buttons row
        frm_btns = ttk.Frame(self); frm_btns.pack(fill="x", **pad)
        self.btn_apply = ttk.Button(frm_btns, text="Reboot", command=self._apply_live)
        self.btn_apply.pack(side="left", padx=4)
        ttk.Button(frm_btns, text="Save .env", command=self._save_env).pack(side="left", padx=8)
        self.btn_stop  = ttk.Button(frm_btns, text="Stop Bot", command=self._stop)
        self.btn_stop.pack(side="left", padx=8)
        ttk.Button(frm_btns, text="Clear Logs", command=self._clear_logs).pack(side="left", padx=8)
        ttk.Button(frm_btns, text="Copy Logs", command=self._copy_logs).pack(side="left", padx=8)
        ttk.Button(frm_btns, text="Open Log Folder", command=self._open_log_folder).pack(side="left", padx=8)

        # Status + Logs
        frm_status = ttk.Frame(self); frm_status.pack(fill="x", **pad)
        self.lbl_status = ttk.Label(frm_status, text="Status: Booting …", foreground="#1d4ed8"); self.lbl_status.pack(side="left")

        frm_log = ttk.LabelFrame(self, text="Logs"); frm_log.pack(fill="both", expand=True, **pad)
        self.txt_log = ScrolledText(frm_log, height=24, state="disabled", wrap="none"); self.txt_log.pack(fill="both", expand=True, padx=6, pady=6)
        # colors
        for name, color in [("DEBUG","#6b7280"),("INFO","#2563eb"),("WARNING","#d97706"),("ERROR","#dc2626"),("CRITICAL","#ffffff")]:
            if name == "CRITICAL":
                self.txt_log.tag_config(name, foreground="#ffffff", background="#b91c1c")
            else:
                self.txt_log.tag_config(name, foreground=color)

        self._toggle_normal_controls()

    def _toggle_token(self):
        self.ent_token.configure(show="" if self.show_token.get() else "•")

    def _toggle_normal_controls(self):
        normal = self.var_mode.get() == "normal"
        state = "readonly" if normal else "disabled"
        for cmb in (self.cmb_p1, self.cmb_p2, self.cmb_p3):
            cmb.configure(state=state)

    def _wire_logging(self):
        handler = TextHandler(self.txt_log)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)

    def _load_from_manager(self):
        m = self.manager
        # connection
        self.var_token.set(m.TOKEN)
        self.var_guild.set(m.GUILD_ID)
        self.var_channel.set(m.CHANNEL_ID)
        self.var_user.set(m.USER_ID)
        self.var_interval.set(m.SD_INTERVAL_SEC)
        self.var_gpu.set(bool(m.USE_GPU))
        # prefs
        self.var_mode.set(m.MODE)
        self.var_series.set(m.series_preference or "")
        self.var_p1.set(m.NORM_P1); self.var_p2.set(m.NORM_P2); self.var_p3.set(m.NORM_P3)
        self._toggle_normal_controls()

    def _validate_priority_order(self):
        # Ensure unique p1/p2/p3 by auto-fixing duplicates
        vals = [self.var_p1.get(), self.var_p2.get(), self.var_p3.get()]
        allowed = ["high_likes", "low_gen", "no_gen", "series"]
        seen = []
        for i, v in enumerate(vals):
            if v not in allowed or v in seen:
                for a in allowed:
                    if a not in seen:
                        vals[i] = a; break
            seen.append(vals[i])
        self.var_p1.set(vals[0]); self.var_p2.set(vals[1]); self.var_p3.set(vals[2])

    def _apply_to_manager(self):
        self._validate_priority_order()
        m = self.manager

        # Detect changes that require reboot
        needs_reboot = False
        fields = {}

        new_TOKEN = self.var_token.get().strip()
        new_GUILD = self.var_guild.get().strip()
        new_CHAN  = self.var_channel.get().strip()
        new_USER  = self.var_user.get().strip()
        new_GPU   = bool(self.var_gpu.get())
        new_INT   = int(self.var_interval.get())

        if new_TOKEN != m.TOKEN: fields["TOKEN"] = (m.TOKEN, new_TOKEN); needs_reboot = True
        if new_GUILD != m.GUILD_ID: fields["GUILD_ID"] = (m.GUILD_ID, new_GUILD); needs_reboot = True
        if new_CHAN  != m.CHANNEL_ID: fields["CHANNEL_ID"] = (m.CHANNEL_ID, new_CHAN); needs_reboot = True
        if new_USER  != m.USER_ID: fields["USER_ID"] = (m.USER_ID, new_USER); needs_reboot = True
        if new_GPU   != m.USE_GPU: fields["USE_GPU"] = (m.USE_GPU, new_GPU); needs_reboot = True

        # Apply
        m.TOKEN = new_TOKEN
        m.GUILD_ID = new_GUILD
        m.CHANNEL_ID = new_CHAN
        m.USER_ID = new_USER
        m.USE_GPU = new_GPU
        m.SD_INTERVAL_SEC = new_INT

        # Preferences (live-applied, no reboot needed)
        old_mode = m.MODE
        m.MODE = (self.var_mode.get() or "smart").strip().lower()
        m.series_preference = self.var_series.get().strip()
        m.NORM_P1 = (self.var_p1.get() or "high_likes").strip().lower()
        m.NORM_P2 = (self.var_p2.get() or "low_gen").strip().lower()
        m.NORM_P3 = (self.var_p3.get() or "series").strip().lower()
        m._normalize_normal_priorities()

        if old_mode != m.MODE:
            logging.info(f"🔁 Switched mode: {old_mode} → {m.MODE} (live)")

        return needs_reboot, fields

    def _auto_start(self):
        try:
            self.manager.start()
            self.lbl_status.configure(text="Status: Running", foreground="#16a34a")
        except Exception:
            logging.exception("Failed to auto-start bot")
            self.lbl_status.configure(text="Status: Failed to start", foreground="#dc2626")

    def _stop(self):
        try:
            self.manager.stop()
            self.lbl_status.configure(text="Status: Stopped", foreground="#dc2626")
        except Exception:
            pass

    def _apply_live(self):
        if self._apply_in_progress:
            return
        self._apply_in_progress = True
        try:
            needs_reboot, fields = self._apply_to_manager()
            self.manager.save_env()
            changed = ", ".join([k for k in fields.keys()])
            logging.info(f"🔄 Applying connection changes ({changed}) → auto-restart")
            self.manager.reboot_bot("Settings changed that require reconnect")

        finally:
            self._apply_in_progress = False

    def _save_env(self):
        # Save without forcing restart
        try:
            # Keep current manager values (already synced on last Apply)
            self.manager.save_env()
            messagebox.showinfo("Saved", f"Settings saved to {self.manager.cfg_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save .env: {e}")

    def _clear_logs(self):
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.configure(state="disabled")

    def _copy_logs(self):
        try:
            text = self.txt_log.get("1.0", tk.END)
            self.clipboard_clear()
            self.clipboard_append(text)
            messagebox.showinfo("Copied", "Logs copied to clipboard.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy logs: {e}")

    def _open_log_folder(self):
        folder = os.getcwd()
        if sys.platform.startswith("win"):
            os.startfile(folder)
        else:
            messagebox.showinfo("Folder", f"Logs are in: {folder}")

    def _signal_handler(self, *_):
        self._stop()
        self.destroy()

if __name__ == "__main__":
    app = SofiApp()
    app.mainloop()
