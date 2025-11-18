# sofi_bot_gui_V3.py
# -*- coding: utf-8 -*-
"""
Sofi Farming Bot — with RAID + Top.gg Vote + improved UI

Key points:
- Discum gateway + EasyOCR + Selenium (Top.gg vote) + Tkinter UI.
- Auto-start; initial 'sd' on READY; first 'sgr' a few seconds later.
- While raid active -> pause 'sd'.
- Vote runs every 12h (with jitter) in a separate thread; never reboots bot if it fails.
- Elite detector preserved; normal/SMART modes preserved; likes parser supports 1.9K / 3M, etc.
- Confirmation detection only for final grab/fight messages.

DISCLAIMER: Self-bots violate Discord ToS. Use at your own risk.

PyInstaller (one line):
pyinstaller --noconfirm --onefile --name "SofiBotGUI" --icon "app.ico" --clean --noupx --collect-all easyocr --collect-all torch --collect-all torchvision --collect-all torchaudio --collect-all cv2 --collect-all PIL --collect-all numpy --hidden-import discum --hidden-import websocket --hidden-import chromedriver_autoinstaller --hidden-import urllib3.contrib.socks --hidden-import dotenv --distpath "dist" --workpath "build" sofi_bot_gui_V3.py
"""

import os
import re
import cv2
import sys
import time
import json
import math
import queue
import random
import logging
import threading
import requests
import numpy as np

from io import BytesIO
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Discum / OCR / env
import discum
import easyocr
from dotenv import load_dotenv

# Selenium for Top.gg vote
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# UI
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

# Requests with retries
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass, field
APP_TITLE = "Sofi Farming Bot — V3 (raid + vote)"
LOG_FILE = "Sofi_bot.log"
# ---- Discord Bot IDs ----
SOFI_BOT_ID = "853629533855809596"
# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("websocket").setLevel(logging.CRITICAL)
logging.getLogger("discum.gateway.gateway").setLevel(logging.ERROR)

# ---------- HTTP with retries ----------
def make_http() -> requests.Session:
    retry = Retry(
        total=5, connect=5, read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=False,
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    s = requests.Session()
    ad = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
    s.mount("https://", ad)
    s.mount("http://", ad)
    s.headers.update({"User-Agent": "SofiBot/1.0 (+requests)"})
    return s

HTTP = make_http()
READER = None  # EasyOCR singleton

# ---------- Elite detector (lean heuristics, same spirit as before) ----------
ELITE_CFG = dict(
    ring_ratio=0.045, edge_margin=0.05, sat_min=45, val_min=60,
    hue_bins=36, edge_nonzero_bins_min=8, edge_peak_max=0.55,
    edge_cstd_min=40.0, edge_count_min=120,
)

def _segment_card_mask(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 40, 120)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, -5)
    mix = cv2.bitwise_or(th, edges)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mix = cv2.morphologyEx(mix, cv2.MORPH_CLOSE, k, iterations=2)
    mix = cv2.morphologyEx(mix, cv2.MORPH_OPEN,  k, iterations=1)
    cnts, _ = cv2.findContours(mix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h,w), np.uint8)
    if not cnts: return mask
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def _make_frame_strips(raw, bottom_exclude=0.22, inset_ratio=0.012, thick_ratio=0.032):
    h, w = raw.shape[:2]
    ys, xs = np.where(raw>0)
    if xs.size==0: return np.zeros_like(raw), {}
    x0,x1 = int(xs.min()), int(xs.max())
    y0,y1 = int(ys.min()), int(ys.max())
    cw, ch = (x1-x0+1), (y1-y0+1)
    inset = max(2, int(round(min(cw,ch)*inset_ratio)))
    thick = max(2, int(round(min(cw,ch)*thick_ratio)))
    ycap  = y0 + int(round((1.0 - bottom_exclude)*ch))
    def clamp(a, lo, hi): return max(lo, min(hi, a))
    top  = (clamp(x0+inset,0,w-1), clamp(y0+inset,0,h-1),
            clamp(x1-inset-(x0+inset),0,w-1), clamp((y0+inset+thick)-(y0+inset),0,h-1))
    left = (clamp(x0+inset,0,w-1), clamp(y0+inset,0,h-1),
            clamp((x0+inset+thick)-(x0+inset),0,w-1), clamp((ycap-inset)-(y0+inset),0,h-1))
    right= (clamp(x1-inset-thick+1,0,w-1), clamp(y0+inset,0,h-1),
            clamp(thick,0,w-1), clamp((ycap-inset)-(y0+inset),0,h-1))
    union = np.zeros_like(raw, np.uint8)
    for (x,y,ww,hh) in (top,left,right):
        if ww>1 and hh>1: union[y:y+hh, x:x+ww]=255
    return union, {"top":top,"left":left,"right":right}

def _edge_metrics(Hdeg,S,V,h,w,side,cfg):
    thick_x = max(2, int(w*cfg["ring_ratio"]))
    thick_y = max(2, int(h*cfg["ring_ratio"]))
    mx = max(1, int(w*cfg["edge_margin"]))
    my = max(1, int(h*cfg["edge_margin"]))
    if side=="top":    rr,cc = slice(0,thick_y), slice(mx,w-mx)
    elif side=="bot":  rr,cc = slice(h-thick_y,h), slice(mx,w-mx)
    elif side=="left": rr,cc = slice(my,h-my), slice(0,thick_x)
    else:              rr,cc = slice(my,h-my), slice(w-thick_x,w)
    m = (S[rr,cc] >= cfg["sat_min"]) & (V[rr,cc] >= cfg["val_min"])
    hue = Hdeg[rr,cc][m].astype(np.float32)
    if hue.size==0: return dict(ok=False, nonzero=0, peak=1.0, cstd=0.0, n=0)
    hist,_ = np.histogram(hue, bins=cfg["hue_bins"], range=(0,360))
    total = float(hist.sum())
    peak = float(hist.max()/max(1.0,total))
    nonzero = int((hist>0).sum())
    ang = np.deg2rad(hue); s,c = np.sin(ang).sum(), np.cos(ang).sum()
    R = np.sqrt(s*s+c*c)/max(1.0,hue.size); R=float(np.clip(R,1e-6,0.999999))
    cstd = float(np.degrees(np.sqrt(-2.0*np.log(R))))
    ok = (hue.size>=cfg["edge_count_min"] and nonzero>=cfg["edge_nonzero_bins_min"]
          and peak<=cfg["edge_peak_max"] and cstd>=cfg["edge_cstd_min"])
    return dict(ok=ok, nonzero=nonzero, peak=peak, cstd=cstd, n=int(hue.size))

def is_elite(bgr):
    if bgr is None or bgr.size==0: return False
    raw = _segment_card_mask(bgr)
    ring, _ = _make_frame_strips(raw, bottom_exclude=0.22)
    if ring.sum()==0: return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hdeg = hsv[...,0].astype(np.float32)*2.0; S=hsv[...,1]; V=hsv[...,2]
    mtop = _edge_metrics(Hdeg,S,V,*bgr.shape[:2], "top", ELITE_CFG)
    mbot = _edge_metrics(Hdeg,S,V,*bgr.shape[:2], "bot", ELITE_CFG)
    mlef = _edge_metrics(Hdeg,S,V,*bgr.shape[:2], "left", ELITE_CFG)
    mrig = _edge_metrics(Hdeg,S,V,*bgr.shape[:2], "right", ELITE_CFG)
    okc = sum([mtop["ok"],mbot["ok"],mlef["ok"],mrig["ok"]])
    if not ((mtop["ok"] and mbot["ok"]) or (mlef["ok"] and mrig["ok"]) or okc>=3):
        return False
    score = (mtop["cstd"]+mbot["cstd"]+mlef["cstd"]+mrig["cstd"])/4.0
    return score>=60.0

# ---------- OCR ----------
def preprocess(image):
    if isinstance(image, Image.Image): image = np.array(image)
    if image.ndim==3: image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def extract_generation_with_easyocr(img_gray):
    global READER
    if READER is None: raise RuntimeError("OCR READER not initialized")
    if isinstance(img_gray, Image.Image): img_gray = np.array(img_gray)
    target_w = 300
    sc = target_w / img_gray.shape[1]
    im = cv2.resize(img_gray, (target_w, int(img_gray.shape[0]*sc)), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    words = READER.readtext(im, detail=0, batch_size=2)

    def clean_generations(tokens):
        out = {}
        for t in tokens:
            if len(out)>=3: break
            orig = (t or "").strip()
            if not orig: continue
            up = orig.upper()
            m6 = re.match(r"^6G(\d{1,4})$", up)
            if m6:
                g = int(m6.group(1)); out[f"G{g}"]=g; continue
            s = re.sub(r"[^A-Za-z0-9]", "", orig)
            s = (s.replace("i","1").replace("I","1")
                   .replace("o","0").replace("O","0")
                   .replace("g","9").replace("s","5").replace("S","5")
                   .replace("B","8").replace("l","1"))
            if s and s[0] in "0659" and not s.upper().startswith("G"): s="G"+s[1:]
            m = re.match(r"^G(\d{1,4})$", s.upper())
            if m:
                g = int(m.group(1)); out[f"G{g}"]=g
        return out

    def name_series(tokens):
        if not tokens or len(tokens)<2: return None, None
        gi=-1
        for i,t in enumerate(tokens):
            if clean_generations([t]): gi=i; break
        if gi==-1 or gi+1>=len(tokens): return None, None
        name = (tokens[gi+1] or "").strip()
        series = (tokens[gi+2] or "").strip() if gi+2<len(tokens) else None
        return name, series

    gens = clean_generations(words)
    name, series = name_series(words)
    return gens, name, series

def split_three_cards(image: Image.Image):
    w = image.width; each = w//3
    crops=[]
    for i in range(3):
        L = i*each; R = (i+1)*each if i<2 else w
        crops.append(image.crop((L,0,R,image.height)))
    return crops

def extract_cards(image: Image.Image):
    info={}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs=[]
        for i, card in enumerate(split_three_cards(image)):
            futs.append(ex.submit(_proc_one_card, i, card))
        for f in futs:
            i,data = f.result()
            info[i]=data
    return info

def _proc_one_card(i, card_img):
    rgb = np.array(card_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elite = is_elite(bgr)
    gens, name, series = extract_generation_with_easyocr(preprocess(card_img))
    logging.info(f"✅ Card {i}: Gen={list((gens or {}).keys()) or '∅'}, Name='{name or ''}', Series='{series or ''}', Elite={elite}")
    return i, {"generations": gens or {}, "name": name or "", "series": series or "", "elite": bool(elite)}

def likes_from_label(label: str):
    s = (label or "").strip().replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*([kKmM]?)", s)
    if not m: return None
    val = float(m.group(1)); suf = m.group(2).lower()
    if suf=="k": val *= 1_000.0
    if suf=="m": val *= 1_000_000.0
    return int(round(val))
@dataclass
class RaidState:
    active: bool = False
    message_id: str | None = None
    channel_id: str | None = None
    guild_id: str | None = None
    phase: str = "idle"          # idle | waiting_first | waiting_second | fighting
    last_action_ts: float = 0.0
    last_seen_ts: float = 0.0
    turn_counter: int = 0
    # component ids we discover on each frame
    start_button_id: str | None = None
    apply_button_id: str | None = None
    select_custom_id: str | None = None
    select_values: list[str] = field(default_factory=list)

# ---------- Selenium voter ----------
class TopGGVoter:
    def __init__(self, url, chrome_path="", profile_dir="", headless=False, timeout=35):
        self.url = url
        self.chrome_path = chrome_path.strip()
        self.profile = profile_dir.strip()
        self.headless = headless
        self.timeout = timeout

    def _driver(self):
        # Ensure chromedriver matches installed Chrome
        driver_path = chromedriver_autoinstaller.install()

        opts = webdriver.ChromeOptions()

        # Validate Chrome binary path if provided
        if self.chrome_path:
            if os.path.isfile(self.chrome_path):
                opts.binary_location = self.chrome_path
            else:
                logging.warning(f"[vote] CHROME_PATH does not exist: {self.chrome_path}. Using system default.")
                self.chrome_path = ""

        # Validate Chrome profile path if provided
        if self.profile:
            if os.path.isdir(self.profile):
                opts.add_argument(f"--user-data-dir={self.profile}")
            else:
                logging.warning(f"[vote] Chrome profile dir does not exist: {self.profile}. Ignoring.")
                self.profile = ""

        # Reduce crash likelihood
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-extensions")
        opts.add_argument("--no-first-run")
        opts.add_argument("--no-default-browser-check")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        if self.headless:
            opts.add_argument("--headless=new")

        try:
            svc = ChromeService(executable_path=driver_path)
            d = webdriver.Chrome(service=svc, options=opts)
            d.set_page_load_timeout(self.timeout)
            return d
        except WebDriverException as e:
            # This is where "session not created: Chrome instance exited" usually comes from
            logging.warning(f"[vote] WebDriver init failed: {e}")
            raise

    def _dismiss_consents(self, d):
        try:
            w = WebDriverWait(d, 8)
            candidates = [
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept')]")),
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'I agree')]")),
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'OK')]")),
            ]
            for c in candidates:
                try:
                    btn = w.until(c)
                    btn.click()
                    break
                except TimeoutException:
                    continue
        except Exception:
            pass

    def _click_vote(self, d):
        w = WebDriverWait(d, 20)
        paths = [
            "//button[.//span[contains(., 'Vote')]]",
            "//button[contains(., 'Vote')]",
            "//a[contains(., 'Vote')]",
            "//div[@role='button' and contains(., 'Vote')]",
        ]
        for xp in paths:
            try:
                el = w.until(EC.element_to_be_clickable((By.XPATH, xp)))
                el.click()
                logging.info("[vote] Vote clicked")
                return True
            except TimeoutException:
                continue
        logging.warning("[vote] Vote button not found")
        return False

    def vote_once(self):
        d = None
        try:
            d = self._driver()
            d.get(self.url)
            time.sleep(1.0)
            self._dismiss_consents(d)
            return self._click_vote(d)
        except WebDriverException as e:
            msg = str(e)
            logging.warning(f"[vote] failed: {msg}")
            if "session not created" in msg.lower():
                logging.warning("[vote] Chrome/driver session could not be created. "
                                "Check CHROME_PATH / CHROME_PROFILE or reinstall Chrome.")
            return False
        except Exception as e:
            logging.warning(f"[vote] error: {e}")
            return False
        finally:
            if d is not None:
                try:
                    d.quit()
                except Exception:
                    pass

# ---------- Bot Manager ----------
class SofiBot:
    SOFI_ID = "853629533855809596"

    def __init__(self, env_path=".env"):
        self.http = HTTP
        self.env_path = env_path
        load_dotenv(self.env_path)
        self.raid_state = RaidState()
        # Required
        self.TOKEN      = os.getenv("DISCORD_TOKEN","")
        self.GUILD_ID   = os.getenv("GUILD_ID","")
        self.CHANNEL_ID = os.getenv("CHANNEL_ID","")
        self.USER_ID    = os.getenv("USER_ID","")
        self.SOFI_BOT_ID = SOFI_BOT_ID
        # OCR
        self.USE_GPU = (os.getenv("OCR_USE_GPU","0")=="1")

        # Mode/priorities
        self.MODE = os.getenv("MODE","smart").strip().lower()
        if self.MODE not in ("smart","normal"): self.MODE="smart"
        self.NORM_P1 = os.getenv("NORM_P1","high_likes").strip().lower()
        self.NORM_P2 = os.getenv("NORM_P2","low_gen").strip().lower()
        self.NORM_P3 = os.getenv("NORM_P3","no_gen").strip().lower()
        self.series_pref = os.getenv("SERIES_NAME","").strip()
        self._normalize_priorities()

        # Smart scoring knobs
        self.SCORE_BONUS_SERIES = float(os.getenv("SCORE_BONUS_SERIES","0.12"))
        self.SCORE_BONUS_NOGEN  = float(os.getenv("SCORE_BONUS_NOGEN","0.03"))
        self.T_LIKE = int(os.getenv("T_LIKE","10"))
        self.T_GEN  = int(os.getenv("T_GEN","40"))
        self.WL_MIN = float(os.getenv("WL_MIN","0.15"))
        self.WL_SPAN= float(os.getenv("WL_SPAN","0.70"))
        self.LIKES_LOG_DAMP = (os.getenv("LIKES_LOG_DAMP","1")=="1")

        # Timings
        self.SD_INTERVAL_SEC = int(os.getenv("SD_INTERVAL_SEC","480"))
        self.CLAIM_CONFIRM_TIMEOUT = int(os.getenv("CLAIM_CONFIRM_TIMEOUT","16"))
        self.POST_ACORN_NORMAL_DELAY = float(os.getenv("POST_ACORN_NORMAL_DELAY","0.8"))
        self.PROCESS_COOLDOWN_SECONDS = int(os.getenv("PROCESS_COOLDOWN_SECONDS","240"))

        # Watchdogs
        self.WATCHDOG_TIMEOUT = int(os.getenv("WATCHDOG_TIMEOUT","600"))
        self.GATEWAY_READY_TIMEOUT = int(os.getenv("GATEWAY_READY_TIMEOUT","120"))

        # Vote settings
        self.ENABLE_VOTE = (os.getenv("ENABLE_VOTE","1")=="1")
        self.VOTE_INTERVAL_H = float(os.getenv("VOTE_INTERVAL_H","12"))
        self.VOTE_JITTER_MIN = int(os.getenv("VOTE_JITTER_MIN","60"))
        self.VOTE_JITTER_MAX = int(os.getenv("VOTE_JITTER_MAX","300"))
        self.TOPGG_URL = os.getenv("TOPGG_URL","https://top.gg/bot/853629533855809596/vote")
        self.CHROME_PATH = os.getenv("CHROME_PATH","")
        default_prof = os.path.expandvars(r"%LOCALAPPDATA%/Google/Chrome/User Data")
        self.CHROME_PROFILE = os.getenv("CHROME_PROFILE", default_prof if os.path.isdir(default_prof) else "")

        # Raid settings
        self.ENABLE_RAID = (os.getenv("ENABLE_RAID","1")=="1")
        self.RAID_INTERVAL_H = float(os.getenv("RAID_INTERVAL_H","3"))
        self.RAID_JITTER_MIN = int(os.getenv("RAID_JITTER_MIN","90"))
        self.RAID_JITTER_MAX = int(os.getenv("RAID_JITTER_MAX","240"))

        # State
        self.bot=None
        self.stop_event = threading.Event()
        self.gateway_ready=False
        self.last_gateway_event=0.0
        self.gateway_start=0.0

        self.pending_claim = {"triggered":False, "timestamp":0.0, "user_id": self.USER_ID}
        self.last_processed_time_lock = Lock()
        self.last_processed_time=0.0

        self._raid_active = threading.Event()
        self._reboot_lock = Lock()
        self._last_reboot_ts=0.0
        self.REBOOT_MIN_INTERVAL=20.0

        self.sent_initial_sd=False
        self._run_token=0

        # Threads
        self.sd_thread = None
        self.watchdog_thread = None
        self.claim_timeout_thread = None
        self.raid_thread = None
        self.vote_thread = None
        self.bot_thread = None

        # --- RAID state ---
        self.raid_active = False
        self.raid_message_id = None
        self.raid_started_ts = 0.0
        self.raid_last_action = 0.0
        self.RAID_ACTION_COOLDOWN = 1.0  # small delay between clicks
        self.RAID_MOVE_DELAY = 2.0       # wait after Apply before next choice
        self.RAID_START_DELAY = 3.0      # delay after sending 'sgr' before clicking Start
        self.RAID_MAX_DURATION = 600.0   # safety stop (10 min)
        self.RAID_AUTO_START_ON_READY = True  # send a raid on startup
        # raid state machine: "idle" -> "first_clicked" -> "second_clicked"
        self.raid_state = "idle"
    def _normalize_priorities(self):
        allowed = ["high_likes","low_gen","no_gen","series"]
        picks = [p for p in [self.NORM_P1,self.NORM_P2,self.NORM_P3] if p in allowed]
        for a in allowed:
            if a not in picks: picks.append(a)
        self.NORM_P1,self.NORM_P2,self.NORM_P3 = picks[:3]

    # ---------- lifecycle ----------
    def start(self):
        global READER
        if self.bot_thread and self.bot_thread.is_alive(): return
        logging.info(f"🔎 Initializing EasyOCR (gpu={self.USE_GPU}) …")
        if READER is None:
            READER = easyocr.Reader(["en"], gpu=self.USE_GPU, verbose=False)
            _ = READER.readtext(np.zeros((50,50,3),dtype=np.uint8), detail=0)
        self.stop_event.clear()
        self.bot = discum.Client(token=self.TOKEN, log=False)
        self._install_handlers()
        self.gateway_ready=False
        self.last_gateway_event=0.0
        self.gateway_start=time.time()
        self.sent_initial_sd=False
        self._run_token += 1

        if not (self.sd_thread and self.sd_thread.is_alive()):
            self.sd_thread = threading.Thread(target=self._sd_loop, daemon=True); self.sd_thread.start()
        if not (self.watchdog_thread and self.watchdog_thread.is_alive()):
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True); self.watchdog_thread.start()
        if not (self.claim_timeout_thread and self.claim_timeout_thread.is_alive()):
            self.claim_timeout_thread = threading.Thread(target=self._claim_timeout_loop, daemon=True); self.claim_timeout_thread.start()
        if self.ENABLE_RAID and not (self.raid_thread and self.raid_thread.is_alive()):
            self.raid_thread = threading.Thread(target=self._raid_loop, daemon=True); self.raid_thread.start()
        if self.ENABLE_VOTE and not (self.vote_thread and self.vote_thread.is_alive()):
            self.vote_thread = threading.Thread(target=self._vote_loop, daemon=True); self.vote_thread.start()

        self.bot_thread = threading.Thread(target=self.bot.gateway.run, kwargs={"auto_reconnect":True}, daemon=True)
        self.bot_thread.start()

    def stop(self, internal=False):
        self.stop_event.set()
        try:
            if self.bot and getattr(self.bot, "gateway", None): self.bot.gateway.close()
        except Exception:
            pass
        logging.info("✅ Bot stopped.")

    def reboot(self, reason):
        now=time.time()
        if (now-self._last_reboot_ts)<self.REBOOT_MIN_INTERVAL:
            logging.warning(f"⏳ Reboot suppressed (too soon). Reason: {reason}"); return
        if not self._reboot_lock.acquire(blocking=False): return
        try:
            self._last_reboot_ts=now
            logging.error(f"🔄 REBOOTING — {reason}")
            self.stop(internal=True)
            time.sleep(1.0)
            self.start()
        finally:
            self._reboot_lock.release()

    # ---------- loops ----------
    def _sd_loop(self):
        token = self._run_token
        while not self.stop_event.is_set():
            wait = max(300.0, float(self.SD_INTERVAL_SEC) + random.uniform(2, 12))
            logging.info(f"⏳ Next 'sd' in {wait:.1f}s")
            if self.stop_event.wait(wait):
                break
            if token != self._run_token:
                return

            # Do not send 'sd' while a raid is active (single source of truth = RaidState)
            if isinstance(self.raid_state, RaidState) and self.raid_state.active:
                logging.info("🛡️ Raid active — skipping 'sd'")
                continue

            ready = (self.gateway_ready and self.bot and getattr(self.bot.gateway, "session_id", None))
            if not ready:
                logging.warning("⚠️ Gateway not READY at 'sd' time → reboot")
                self.reboot("Gateway not READY when sending sd")
                return
            try:
                self.bot.sendMessage(self.CHANNEL_ID, "sd")
                logging.info("📤 Sent 'sd'")
            except Exception:
                logging.exception("⚠️ Failed to send 'sd'")

    def _vote_loop(self):
        # run once at startup then every 12h + jitter
        self._perform_vote_once()
        while not self.stop_event.is_set():
            base = self.VOTE_INTERVAL_H*3600.0
            jitter = random.uniform(self.VOTE_JITTER_MIN, self.VOTE_JITTER_MAX)
            wait = base + jitter
            logging.info(f"[vote] Next vote in ~{int(wait)}s")
            if self.stop_event.wait(wait): break
            self._perform_vote_once()

    def _perform_vote_once(self):
        if not self.ENABLE_VOTE: return
        try:
            logging.info("[vote] Attempting Top.gg vote…")
            voter = TopGGVoter(
                url=self.TOPGG_URL,
                chrome_path=self.CHROME_PATH,
                profile_dir=self.CHROME_PROFILE,
                headless=False,
                timeout=35,
            )
            ok = voter.vote_once()
            if ok: logging.info("[vote] Vote OK — next in ~12h")
            else:  logging.info("[vote] Vote FAILED — will retry next window")
        except Exception as e:
            logging.warning(f"[vote] error: {e}")

    def _raid_loop(self):
        # initial raid a few seconds after READY is set by handler
        # here we only schedule the periodic ones
        while not self.stop_event.is_set():
            base = self.RAID_INTERVAL_H*3600.0
            jitter = random.uniform(self.RAID_JITTER_MIN, self.RAID_JITTER_MAX)
            wait = base + jitter
            logging.info(f"[raid] Next raid in ~{int(wait)}s")
            if self.stop_event.wait(wait): break
            self._trigger_raid()

    def _trigger_raid(self, startup=False):
        """Send 'sgr'; Sofi's response will be handled by the raid driver."""
        try:
            if not (self.gateway_ready and self.bot and getattr(self.bot.gateway, "session_id", None)):
                logging.warning("[raid] Gateway not ready; skipping raid")
                return
            self.bot.sendMessage(self.CHANNEL_ID, "sgr")
            logging.info("📤 Sent 'sgr'")
        except Exception as e:
            logging.warning(f"[raid] send 'sgr' failed: {e}")

    def _watchdog_loop(self):
        while not self.stop_event.is_set():
            if self.stop_event.wait(5): break
            last = self.last_gateway_event or (time.time()-30)
            elapsed = time.time()-last
            if elapsed>self.WATCHDOG_TIMEOUT:
                self.reboot(f"No gateway events for {elapsed:.1f}s"); return
            if (not self.gateway_ready) and self.gateway_start and ((time.time()-self.gateway_start)>self.GATEWAY_READY_TIMEOUT):
                self.reboot("Gateway did not reach READY in time"); return

    def _claim_timeout_loop(self):
        while not self.stop_event.is_set():
            if self.stop_event.wait(0.5): break
            if self.pending_claim["triggered"]:
                if (time.time()-self.pending_claim["timestamp"])>self.CLAIM_CONFIRM_TIMEOUT:
                    logging.warning("⚠️ No Sofi confirmation received (timeout). Reset.")
                    self.pending_claim["triggered"]=False
    def _start_raid_now(self):
        """Convenience helper used at startup to start a raid."""
        try:
            self._trigger_raid(startup=True)
        except Exception:
            logging.exception("Failed to send 'sgr' for startup raid")
     # ---------- RAID component parsing ----------
    @staticmethod
    def _find_buttons(components):
        """Return list of (custom_id, label) for all real buttons in a Sofi component tree."""
        out = []
        for row in components or []:
            for c in row.get("components", []):
                if c.get("type") == 2 and c.get("custom_id"):  # button
                    out.append((c["custom_id"], (c.get("label") or "")))
        return out

    @staticmethod
    def _find_selects(components):
        """Return list of select components (type==3)."""
        sels = []
        for row in components or []:
            for c in row.get("components", []):
                if c.get("type") == 3:
                    sels.append(c)
        return sels
    def click_discord_button(self, custom_id, channel_id, guild_id, message):
        """Wrapper around discum's button click; robust to missing flags."""
        try:
            self.bot.click(
                applicationID=message["author"]["id"],
                channelID=str(channel_id),
                guildID=str(guild_id) if guild_id else None,
                messageID=message["id"],
                messageFlags=message.get("flags", 0),
                data={"component_type": 2, "custom_id": custom_id},
            )
            logging.info(f"➡️ Clicked button {custom_id}")
        except Exception as e:
            logging.warning(f"Button click failed: {e}")
    def _reset_raid_state(self, reason=""):
        rs = self.raid_state
        if isinstance(rs, RaidState) and rs.active:
            logging.info(f"🧹 Reset raid state ({reason})")
        self.raid_state = RaidState()

    def _iter_components(self, components):
        for row in (components or []):
            for comp in row.get("components", []):
                yield comp

    def _click_button_by_label(self, label_contains, components, channel_id, guild_id, message):
        """Find a BUTTON (type=2) whose label contains the given text (case-insensitive) and click it."""
        want = (label_contains or "").lower()
        for comp in self._iter_components(components):
            if comp.get("type") != 2:
                continue
            lab = (comp.get("label") or "").lower()
            if want in lab:
                cid = comp.get("custom_id")
                if cid:
                    self.click_discord_button(cid, channel_id, guild_id, message)
                    return True
        return False

    def _extract_select_menu(self, components):
        """Return (custom_id, [values]) for the first STRING_SELECT (type=3)."""
        for comp in self._iter_components(components):
            if comp.get("type") == 3:  # string select
                cid = comp.get("custom_id")
                values = []
                for opt in comp.get("options", []):
                    val = opt.get("value")
                    if val:
                        values.append(val)
                if cid and values:
                    return cid, values
        return None, []

    def _select_menu_choice(self, custom_id, value, channel_id, guild_id, message):
        """Send an interaction for a select menu choice."""
        try:
            # Discum: use click with component_type 3 and "values"
            self.bot.click(
                applicationID=message["author"]["id"],
                channelID=channel_id,
                guildID=message.get("guild_id"),
                messageID=message["id"],
                messageFlags=message["flags"],
                data={"component_type": 3, "custom_id": custom_id, "values": [value]},
            )
            logging.info(f"🎯 Selected move value '{value}'")
            return True
        except Exception as e:
            logging.warning(f"Select click failed: {e}")
            return False

    def _raid_select_random_move_and_apply(self, components, channel_id, guild_id, message):
        """Pick a random option from any select menu(s) and click the nearest Apply button."""
        selects = self._find_selects(components)
        if not selects:
            return False

        made_choice = False
        for sel in selects:
            cid = sel.get("custom_id")
            options = sel.get("options") or []
            if not cid or not options:
                continue
            choice = random.choice(options)
            val = choice.get("value")
            if not val:
                continue
            # send SELECT interaction
            try:
                self.bot.selectMenu(
                    applicationID=message["author"]["id"],
                    channelID=message["channel_id"],
                    guildID=message.get("guild_id"),
                    messageID=message["id"],
                    messageFlags=message["flags"],
                    data={"component_type": 3, "custom_id": cid, "values": [val]},
                )
                logging.info(f"🧩 Selected RAID move: {choice.get('label') or val}")
                made_choice = True
            except Exception as e:
                logging.warning(f"Select menu interaction failed: {e}")

        # click 'Apply' if present
        if made_choice:
            applied = self._click_button_by_label("apply", components, channel_id, guild_id, message)
            if applied:
                logging.info("✅ Applied raid move")
            else:
                logging.info("ℹ️ No 'Apply' button found yet.")
        return made_choice

    def _raid_should_stop(self, content: str) -> bool:
            """Detect raid end text. Adjust phrases as needed."""
            if not content:
                return False
            if "RAID: ENDED" in content.upper():
                return True
            low = content.lower()
            return (
                "raid ended" in low
                or "the raid has ended" in low
                or "victory!" in low
                or "defeat!" in low
            )
    def _find_button_by_label(self, label_substr: str, components):
        """Return (custom_id, label) of the first button whose label contains label_substr (case-insensitive)."""
        if not components: 
            return None, None
        needle = (label_substr or "").strip().lower()
        for row in components:
            for c in row.get("components", []):
                if c.get("type") != 2:  # not a button
                    continue
                lbl = str(c.get("label") or "")
                if needle in lbl.lower() and c.get("custom_id"):
                    return c["custom_id"], lbl
        return None, None
    def _maybe_drive_raid(self, payload):
        """
        Accepts either a discum Gateway response (with .raw) OR a plain Discord
        message dict. Drives Sofi raid UI until 'RAID: ENDED'.
        """
        try:
            # --- accept both resp and dict ---
            if hasattr(payload, "raw"):                # discum Gateway response
                if payload.raw.get("t") not in ("MESSAGE_CREATE", "MESSAGE_UPDATE"):
                    return
                d = payload.raw.get("d", {}) or {}
            elif isinstance(payload, dict):            # plain message dict
                d = payload
            else:
                return

            # make sure we have a valid raid state object
            if not isinstance(getattr(self, "raid_state", None), RaidState):
                self.raid_state = RaidState()

            author_id = str(d.get("author", {}).get("id", ""))
            if author_id != self.SOFI_BOT_ID:
                return

            channel_id = str(d.get("channel_id", ""))
            if self.CHANNEL_ID and channel_id != self.CHANNEL_ID:
                return

            content    = d.get("content", "") or ""
            components = d.get("components", []) or []

            rs = self.raid_state
            now = time.time()
            rs.last_seen_ts = now
            rs.channel_id   = channel_id
            rs.guild_id     = d.get("guild_id")
            rs.message_id   = d.get("id")

            # Finish condition
            if self._raid_should_stop(content):
                logging.info("🏁 RAID ended — stopping driver")
                self._reset_raid_state("ended")
                return

            # Panels with Start Raid
            if "SOFI: RAID" in content.upper():
                if self._click_button_by_label("start raid", components, channel_id, rs.guild_id, d):
                    rs.active = True
                    # flip between first and second screens
                    rs.phase = "waiting_second" if rs.phase == "waiting_first" else "waiting_first"
                    rs.last_action_ts = now
                    logging.info("▶️ Clicked 'Start Raid'")
                return

            # Fighting panel: select menu + Apply button
            select_cid, values = self._extract_select_menu(components)
            apply_id = None
            for comp in self._iter_components(components):
                if comp.get("type") == 2 and "apply" in (comp.get("label", "").lower()):
                    apply_id = comp.get("custom_id")
                    break

            if select_cid and values and apply_id:
                rs.active = True
                rs.phase  = "fighting"
                rs.select_custom_id = select_cid
                rs.select_values    = values
                rs.apply_button_id  = apply_id

                # avoid spamming interactions too quickly
                if (now - rs.last_action_ts) < 1.5:
                    return

                # pick a random move and apply
                choice = random.choice(values)
                if self._select_menu_choice(select_cid, choice, channel_id, rs.guild_id, d):
                    time.sleep(0.4)
                    self.click_discord_button(apply_id, channel_id, rs.guild_id, d)
                    rs.turn_counter += 1
                    rs.last_action_ts = time.time()
                    logging.info(f"🗡️ Applied move #{rs.turn_counter}")
                return

            # Safety: if active but stale for too long, reset
            if rs.active and (now - rs.last_action_ts) > 90:
                self._reset_raid_state("stale")
        except Exception as e:
            logging.warning(f"_maybe_drive_raid error: {e}")


    # ---------- gateway handlers ----------
    def _install_handlers(self):
        @self.bot.gateway.command
        def on_message_create_for_raid(resp):
            if not hasattr(resp, "raw") or resp.raw.get("t") != "MESSAGE_CREATE":
                return
            self._maybe_drive_raid(resp)

        @self.bot.gateway.command
        def on_message_update(resp):
            if not hasattr(resp, "raw") or resp.raw.get("t") != "MESSAGE_UPDATE":
                return
            self._maybe_drive_raid(resp)
        @self.bot.gateway.command
        def on_message_update(resp):
                    if not hasattr(resp, "raw") or resp.raw.get("t") != "MESSAGE_UPDATE":
                        return
                    self.last_gateway_event = time.time()
                    d = resp.raw["d"]
                    msg = {
                        "id":        d.get("id"),
                        "channel_id": d.get("channel_id"),
                        "guild_id":   d.get("guild_id"),
                        "content":    d.get("content") or "",
                        "components": d.get("components") or [],
                        # updates often omit author; assume Sofi so the raid driver can proceed once the message id is known
                        "author":     d.get("author") or {"id": SOFI_BOT_ID},
                        "flags":      d.get("flags", 0),
                    }
                    self._maybe_drive_raid(msg)

        @self.bot.gateway.command
        def on_message_create_for_raid(resp):
            if not hasattr(resp, "raw") or resp.raw.get("t") != "MESSAGE_CREATE":
                return
            d = resp.raw["d"]
            msg = {
                "id":        d.get("id"),
                "channel_id": d.get("channel_id"),
                "guild_id":   d.get("guild_id"),
                "content":    d.get("content") or "",
                "components": d.get("components") or [],
                "author":     d.get("author") or {},
                "flags":      d.get("flags", 0),
            }
            self._maybe_drive_raid(msg)
        @self.bot.gateway.command
        def any_event(resp):
            try:
                if hasattr(resp,"raw") and resp.raw.get("op")==0:
                    self.last_gateway_event=time.time()
            except Exception: pass

        @self.bot.gateway.command
        def on_ready(resp):
            if not resp.event.ready:
                return
            user = resp.parsed.auto().get("user")
            if user:
                logging.info(f"✅ READY as {user['username']}#{user['discriminator']}")
            self.gateway_ready = True
            self.last_gateway_event = time.time()
            self.gateway_start_time = 0.0

            # first sd
            if not self.sent_initial_sd:
                try:
                    self.bot.sendMessage(self.CHANNEL_ID, "sd")
                    self.sent_initial_sd = True
                    logging.info("📤 Sent 'sd'")
                except Exception:
                    logging.exception("failed to send initial sd")

            # delay raid start a bit so it doesn't collide with sd
            def _later():
                time.sleep(4.0)
                self._start_raid_now()
            threading.Thread(target=_later, daemon=True).start()

        @self.bot.gateway.command
        def on_message(resp):
            if not hasattr(resp, "raw") or resp.raw.get("t") != "MESSAGE_CREATE":
                return

            d = resp.raw["d"]
            m = resp.parsed.auto()
            author_id = str(d.get("author", {}).get("id"))
            channel_id = str(d.get("channel_id"))
            guild_id = str(d.get("guild_id")) if d.get("guild_id") else None
            content = d.get("content", "")
            components = d.get("components", [])
            atts = d.get("attachments", [])
            self.last_gateway_event = time.time()

            # only Sofi messages on the configured guild
            if author_id != SofiBot.SOFI_ID:
                return
            if self.GUILD_ID and guild_id != self.GUILD_ID:
                return

            # Final confirmation after clicking a button
            if self.pending_claim["triggered"]:
                eg = f"<@{self.USER_ID}> **grabbed** the"
                ef = f"<@{self.USER_ID}> fought off"
                if content.startswith(eg) or content.startswith(ef):
                    logging.info(f"✅ Claim confirmed: {content[:160]}…")
                    with self.last_processed_time_lock:
                        self.last_processed_time = time.time()
                    self.pending_claim["triggered"] = False
                    return

            # If a raid is active (as tracked by RaidState), do NOT process drops here.
            # Raid buttons / selects are handled in _maybe_drive_raid via other handlers.
            if isinstance(self.raid_state, RaidState) and self.raid_state.active:
                return

            # Normal Sofi drops: image + buttons
            if atts and components:
                for att in atts:
                    url = att.get("url")
                    if not url:
                        continue
                    if not url.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        continue
                    threading.Thread(
                        target=self._process_drop,
                        args=(url, components, channel_id, guild_id, m),
                        daemon=True,
                    ).start()

    # ---------- buttons/selects ----------
    def _click_button(self, custom_id, channel_id, guild_id, m):
        try:
            self.bot.click(
                applicationID=m["author"]["id"],
                channelID=channel_id,
                guildID=m.get("guild_id"),
                messageID=m["id"],
                messageFlags=m["flags"],
                data={"component_type":2,"custom_id":custom_id},
            )
            logging.info(f"➡️ Click {custom_id}")
        except Exception as e:
            logging.warning(f"click button error: {e}")

    def _click_select(self, custom_id, value, channel_id, guild_id, m):
        try:
            self.bot.click(
                applicationID=m["author"]["id"],
                channelID=channel_id,
                guildID=m.get("guild_id"),
                messageID=m["id"],
                messageFlags=m["flags"],
                data={"component_type":3,"custom_id":custom_id,"type":3,"values":[value]},
            )
            logging.info(f"🎛️ Select {value} on {custom_id}")
        except Exception as e:
            logging.warning(f"select error: {e}")

    def _click_normal_with_retry(self, button_id, channel_id, guild_id, m, reason):
        def do_click():
            try:
                self._click_button(button_id, channel_id, guild_id, m)
                self.pending_claim.update({"triggered":True, "timestamp":time.time(), "user_id": self.USER_ID})
                logging.info(f"🔘 NORMAL: {reason}")
            except Exception as e:
                logging.warning(f"normal click failed: {e}")
        do_click()
        t0=time.time()
        while self.pending_claim["triggered"] and (time.time()-t0)<self.CLAIM_CONFIRM_TIMEOUT:
            time.sleep(0.25)
        if self.pending_claim["triggered"]:
            logging.warning("⏱️ No confirmation. Retrying once …")
            self.pending_claim["triggered"]=False
            time.sleep(0.5)
            do_click()
            t1=time.time()
            while self.pending_claim["triggered"] and (time.time()-t1)<self.CLAIM_CONFIRM_TIMEOUT:
                time.sleep(0.25)
            if self.pending_claim["triggered"]:
                logging.error("❌ Still no confirmation after retry.")
                self.pending_claim["triggered"]=False

    # ---------- drop processing ----------
    def _process_drop(self, url, components, channel_id, guild_id, m):
        try:
            r = self.http.get(url, timeout=15); r.raise_for_status()
            pil = Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            logging.warning(f"Image fetch failed: {e}")
            return
        self._bouquet_then_pick(pil, components, channel_id, guild_id, m)

    def _bouquet_then_pick(self, pil, components, channel_id, guild_id, m):
        # parse first 3 buttons
        pos_buttons=[]
        for row in components:
            for b in row.get("components", []):
                if b.get("type")!=2 or not b.get("custom_id"): continue
                likes = likes_from_label(b.get("label") or "")
                pos_buttons.append({"id":b["custom_id"], "likes":likes, "label": b.get("label") or ""})
        pos_buttons=pos_buttons[:3]
        if not pos_buttons:
            logging.warning("No claimable buttons found"); return

        acorn_idxs  = [i for i,b in enumerate(pos_buttons) if b["likes"] is None]
        normal_idxs = [i for i,b in enumerate(pos_buttons) if b["likes"] is not None]

        info = extract_cards(pil)

        # acorns first (all)
        acorn_clicked=False
        for idx,pos in enumerate(acorn_idxs, start=1):
            try:
                self._click_button(pos_buttons[pos]["id"], channel_id, guild_id, m)
                logging.info(f"🌰 Acorn claimed @ {pos+1} (#{idx})")
                acorn_clicked=True
                time.sleep(0.25)
            except Exception as e:
                logging.warning(f"Acorn click failed: {e}")

        if not normal_idxs: return
        if acorn_clicked: time.sleep(self.POST_ACORN_NORMAL_DELAY)

        # assemble normals
        cards=[]
        for i in range(3):
            ii = info.get(i,{})
            gens = ii.get("generations",{}) or {}
            min_gen = min(gens.values()) if gens else None
            like_val = pos_buttons[i]["likes"] if i<len(pos_buttons) else 0
            cards.append({
                "pos": i,
                "likes": int(like_val if like_val is not None else 0),
                "gens": gens,
                "min_gen": min_gen,
                "has_gen": bool(gens),
                "name": ii.get("name") or "",
                "series": ii.get("series") or "",
                "elite": bool(ii.get("elite", False)),
            })
        normals = [cards[i] for i in normal_idxs]

        # fast paths
        epos = next((c["pos"] for c in normals if c["elite"]), None)
        if epos is not None:
            reason=f"ELITE @pos {epos+1}"
            self._click_normal_with_retry(pos_buttons[epos]["id"], channel_id, guild_id, m, reason)
            return
        low10=[c for c in normals if c["min_gen"] is not None and c["min_gen"]<10]
        if low10:
            low10.sort(key=lambda c:(c["min_gen"], -c["likes"], c["pos"]))
            ch=low10[0]
            reason=f"ABS gen<10 | G{ch['min_gen']} | likes={ch['likes']}"
            self._click_normal_with_retry(pos_buttons[ch["pos"]]["id"], channel_id, guild_id, m, reason)
            return

        # shared stats
        series_key=(self.series_pref or "").lower()
        like_vals=[c["likes"] for c in normals]
        max_l=max(like_vals) if like_vals else 0
        min_l=min(like_vals) if like_vals else 0
        like_gap=max_l-min_l
        gvals=[c["min_gen"] for c in normals if c["min_gen"] is not None]
        min_g=min(gvals) if gvals else None
        max_g=max(gvals) if gvals else None
        gen_gap=(max_g-min_g) if gvals else 0

        def score_smart(c):
            # dynamic weights
            L = like_gap/(like_gap+self.T_LIKE) if like_gap>0 else 0.0
            Gtight = 1.0 - (gen_gap/(gen_gap+self.T_GEN)) if gen_gap>0 else 1.0
            wL = max(0.1, min(0.9, self.WL_MIN + self.WL_SPAN*(0.5*(L+Gtight))))
            wG = 1.0 - wL
            # likes
            if max_l<=0: like_norm=0.0
            else:
                like_norm = (np.log1p(c["likes"]) / max(1e-9, np.log1p(max_l))) if self.LIKES_LOG_DAMP else (c["likes"]/max_l)
            # gen (lower is better)
            if (c["min_gen"] is None or min_g is None or max_g is None or max_g==min_g):
                gen_norm = 0.0
                nog = self.SCORE_BONUS_NOGEN if not c["has_gen"] else 0.0
            else:
                gen_norm = (max_g - c["min_gen"]) / max(1.0, (max_g - min_g))
                nog = 0.0
            series_bonus = self.SCORE_BONUS_SERIES if (series_key and series_key in (c["series"] or "").lower()) else 0.0
            return wL*like_norm + (1.0-wL)*gen_norm + series_bonus + nog

        def series_filter(pool):
            return [c for c in pool if series_key and series_key in (c["series"] or "").lower()]

        def tiebreak(pool):
            pool.sort(key=lambda c: (c["min_gen"] if c["min_gen"] is not None else 10**9, -c["likes"], c["pos"]))
            return pool[0]

        # logs
        header="pos | gen  | like | E | score"
        def fmt(c):
            g = f"G{c['min_gen']}" if c["min_gen"] is not None else "∅"
            sc = f"{c.get('score',0):.3f}" if "score" in c else "---"
            return f"{c['pos']+1:^3} | {g:>4} | {c['likes']:>4} | {'E' if c['elite'] else '-'} | {sc:>6}"
        logging.info("📋 Candidates:\n  "+header+"\n  "+"-"*len(header))
        for c in normals: logging.info("  "+fmt(c))

        if self.MODE=="smart":
            for c in normals: c["score"]=score_smart(c)
            logging.info("📊 SMART scores:")
            for c in normals: logging.info("  "+fmt(c))
            normals.sort(key=lambda c:(-c["score"], c["min_gen"] if c["min_gen"] is not None else 10**9, -c["likes"], c["pos"]))
            chosen=normals[0]
            reason=f"SMART {chosen['score']:.3f} | G{chosen['min_gen'] if chosen['min_gen'] is not None else '∅'} | likes={chosen['likes']}"
            self._click_normal_with_retry(pos_buttons[chosen["pos"]]["id"], channel_id, guild_id, m, reason)
            return

        # normal mode
        def pick_by(pref, cand):
            if not cand: return []
            if pref=="high_likes":
                mx=max([c["likes"] for c in cand]); return [c for c in cand if c["likes"]==mx]
            if pref=="low_gen":
                gens=[c["min_gen"] for c in cand if c["min_gen"] is not None]
                if not gens: return []
                mn=min(gens); return [c for c in cand if c["min_gen"]==mn]
            if pref=="no_gen":
                return [c for c in cand if (c["min_gen"] is None or not c["has_gen"])]
            if pref=="series":
                s=series_filter(cand); return s if s else []
            return cand

        for pref in (self.NORM_P1,self.NORM_P2,self.NORM_P3):
            pool = pick_by(pref, normals)
            if pool:
                chosen = tiebreak(pool)
                reason = f"NORMAL {self.NORM_P1}>{self.NORM_P2}>{self.NORM_P3} | G{chosen['min_gen'] if chosen['min_gen'] is not None else '∅'} | likes={chosen['likes']}"
                self._click_normal_with_retry(pos_buttons[chosen["pos"]]["id"], channel_id, guild_id, m, reason)
                return
        chosen = tiebreak(normals)
        reason = f"NORMAL fallback | G{chosen['min_gen'] if chosen['min_gen'] is not None else '∅'} | likes={chosen['likes']}"
        self._click_normal_with_retry(pos_buttons[chosen["pos"]]["id"], channel_id, guild_id, m, reason)

# ---------- UI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x720")
        self.minsize(900,640)
        self.bot = SofiBot()
        self._build_ui()
        # auto-start
        self.after(200, self.bot.start)

    def _build_ui(self):
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        nb = ttk.Notebook(self); nb.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # ---- Farm tab ----
        farm = ttk.Frame(nb); farm.columnconfigure(0, weight=1)
        nb.add(farm, text="Farming")

        frm1 = ttk.Labelframe(farm, text="Mode & Priorities"); frm1.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for i in range(6): frm1.columnconfigure(i, weight=1)
        ttk.Label(frm1, text="Mode").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value=self.bot.MODE)
        ttk.Combobox(frm1, textvariable=self.mode_var, values=["smart","normal"], width=10, state="readonly").grid(row=0, column=1, sticky="w")

        vals = ["high_likes","low_gen","no_gen","series"]
        ttk.Label(frm1, text="P1").grid(row=0, column=2, sticky="e")
        self.p1 = tk.StringVar(value=self.bot.NORM_P1); ttk.Combobox(frm1, textvariable=self.p1, values=vals, width=12, state="readonly").grid(row=0, column=3, sticky="w")
        ttk.Label(frm1, text="P2").grid(row=0, column=4, sticky="e")
        self.p2 = tk.StringVar(value=self.bot.NORM_P2); ttk.Combobox(frm1, textvariable=self.p2, values=vals, width=12, state="readonly").grid(row=0, column=5, sticky="w")
        ttk.Label(frm1, text="P3").grid(row=1, column=0, sticky="e")
        self.p3 = tk.StringVar(value=self.bot.NORM_P3); ttk.Combobox(frm1, textvariable=self.p3, values=vals, width=12, state="readonly").grid(row=1, column=1, sticky="w")
        ttk.Label(frm1, text="Series contains").grid(row=1, column=2, sticky="e")
        self.series = tk.StringVar(value=self.bot.series_pref); ttk.Entry(frm1, textvariable=self.series, width=22).grid(row=1, column=3, columnspan=3, sticky="we")

        frm2 = ttk.Labelframe(farm, text="Intervals"); frm2.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for i in range(6): frm2.columnconfigure(i, weight=1)
        ttk.Label(frm2, text="SD interval (s)").grid(row=0, column=0, sticky="e")
        self.sd_int = tk.IntVar(value=self.bot.SD_INTERVAL_SEC); ttk.Entry(frm2, textvariable=self.sd_int, width=10).grid(row=0,column=1,sticky="w")
        ttk.Label(frm2, text="Claim confirm timeout (s)").grid(row=0, column=2, sticky="e")
        self.cc_to = tk.IntVar(value=self.bot.CLAIM_CONFIRM_TIMEOUT); ttk.Entry(frm2, textvariable=self.cc_to, width=10).grid(row=0,column=3,sticky="w")

        # ---- Vote tab ----
        vote = ttk.Frame(nb); nb.add(vote, text="Vote (Top.gg)")
        vote.columnconfigure(1, weight=1)
        self.en_vote = tk.IntVar(value=1 if self.bot.ENABLE_VOTE else 0)
        ttk.Checkbutton(vote, text="Enable vote", variable=self.en_vote).grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Label(vote, text="Chrome binary").grid(row=1, column=0, sticky="e", padx=8)
        self.chrome_path = tk.StringVar(value=self.bot.CHROME_PATH); ttk.Entry(vote, textvariable=self.chrome_path).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Label(vote, text="Chrome user-data-dir").grid(row=2, column=0, sticky="e", padx=8)
        self.chrome_prof = tk.StringVar(value=self.bot.CHROME_PROFILE); ttk.Entry(vote, textvariable=self.chrome_prof).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(vote, text="Vote now", command=self._vote_now).grid(row=0, column=1, sticky="e", padx=8)

        # ---- Raid tab ----
        raid = ttk.Frame(nb); nb.add(raid, text="Raid")
        raid.columnconfigure(1, weight=1)
        self.en_raid = tk.IntVar(value=1 if self.bot.ENABLE_RAID else 0)
        ttk.Checkbutton(raid, text="Enable raid", variable=self.en_raid).grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Label(raid, text="Raid every (h)").grid(row=1, column=0, sticky="e", padx=8)
        self.raid_h = tk.DoubleVar(value=self.bot.RAID_INTERVAL_H); ttk.Entry(raid, textvariable=self.raid_h, width=8).grid(row=1, column=1, sticky="w")
        ttk.Button(raid, text="Start raid now", command=lambda: threading.Thread(target=self.bot._trigger_raid, kwargs={"startup":True}, daemon=True).start()).grid(row=0, column=1, sticky="e", padx=8)

        # ---- Connection tab ----
        conn= ttk.Frame(nb); nb.add(conn, text="Connection / IDs")
        for i in range(2): conn.columnconfigure(i, weight=1)
        self.token = tk.StringVar(value=self.bot.TOKEN)
        self.guild = tk.StringVar(value=self.bot.GUILD_ID)
        self.channel = tk.StringVar(value=self.bot.CHANNEL_ID)
        self.user = tk.StringVar(value=self.bot.USER_ID)
        row=0
        ttk.Label(conn, text="Discord Token").grid(row=row, column=0, sticky="e", padx=8, pady=6); ttk.Entry(conn, textvariable=self.token, show="•").grid(row=row, column=1, sticky="ew", padx=8)
        row+=1
        ttk.Label(conn, text="Guild ID").grid(row=row, column=0, sticky="e", padx=8, pady=6); ttk.Entry(conn, textvariable=self.guild).grid(row=row, column=1, sticky="ew", padx=8)
        row+=1
        ttk.Label(conn, text="Channel ID").grid(row=row, column=0, sticky="e", padx=8, pady=6); ttk.Entry(conn, textvariable=self.channel).grid(row=row, column=1, sticky="ew", padx=8)
        row+=1
        ttk.Label(conn, text="Your User ID").grid(row=row, column=0, sticky="e", padx=8, pady=6); ttk.Entry(conn, textvariable=self.user).grid(row=row, column=1, sticky="ew", padx=8)

        # ---- Actions + Logs ----
        bar = ttk.Frame(self); bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0,8))
        bar.columnconfigure(1, weight=1)
        ttk.Button(bar, text="Apply (live)", command=self._apply).grid(row=0, column=0, padx=4)
        ttk.Button(bar, text="Save .env", command=self._save_env).grid(row=0, column=1, padx=4, sticky="w")
        ttk.Button(bar, text="Stop", command=self.bot.stop).grid(row=0, column=2, padx=4)

        self.log = ScrolledText(self, height=16, wrap="word")
        self.log.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0,8))
        self.rowconfigure(2, weight=1)
        self._hook_logging()

    def _hook_logging(self):
        class TextHandler(logging.Handler):
            def __init__(self, widget): super().__init__(); self.widget=widget
            def emit(self, record):
                msg=self.format(record)
                try:
                    self.widget.after(0, self._append, msg)
                except Exception:
                    pass
            def _append(self, msg):
                self.widget.insert("end", msg+"\n"); self.widget.see("end")
        h = TextHandler(self.log)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s","%H:%M:%S"))
        logging.getLogger().addHandler(h)

    def _apply(self):
        # farming
        self.bot.MODE = self.mode_var.get().strip().lower()
        self.bot.NORM_P1 = self.p1.get().strip().lower()
        self.bot.NORM_P2 = self.p2.get().strip().lower()
        self.bot.NORM_P3 = self.p3.get().strip().lower()
        self.bot.series_pref = self.series.get().strip()
        self.bot._normalize_priorities()
        self.bot.SD_INTERVAL_SEC = int(self.sd_int.get())
        self.bot.CLAIM_CONFIRM_TIMEOUT = int(self.cc_to.get())

        # vote
        self.bot.ENABLE_VOTE = (self.en_vote.get()==1)
        self.bot.CHROME_PATH = self.chrome_path.get().strip()
        self.bot.CHROME_PROFILE = self.chrome_prof.get().strip()

        # raid
        self.bot.ENABLE_RAID = (self.en_raid.get()==1)
        self.bot.RAID_INTERVAL_H = float(self.raid_h.get())

        # IDs
        self.bot.TOKEN = self.token.get().strip()
        self.bot.GUILD_ID = self.guild.get().strip()
        self.bot.CHANNEL_ID = self.channel.get().strip()
        self.bot.USER_ID = self.user.get().strip()

        logging.info("✅ Applied settings.")

    def _save_env(self):
        lines = [
            f"DISCORD_TOKEN={self.token.get().strip()}",
            f"GUILD_ID={self.guild.get().strip()}",
            f"CHANNEL_ID={self.channel.get().strip()}",
            f"USER_ID={self.user.get().strip()}",
            f"OCR_USE_GPU={'1' if self.bot.USE_GPU else '0'}",
            f"MODE={self.mode_var.get().strip().lower()}",
            f"NORM_P1={self.p1.get().strip().lower()}",
            f"NORM_P2={self.p2.get().strip().lower()}",
            f"NORM_P3={self.p3.get().strip().lower()}",
            f"SERIES_NAME={self.series.get().strip()}",
            f"SD_INTERVAL_SEC={int(self.sd_int.get())}",
            f"CLAIM_CONFIRM_TIMEOUT={int(self.cc_to.get())}",
            f"ENABLE_VOTE={'1' if self.en_vote.get()==1 else '0'}",
            f"VOTE_INTERVAL_H={self.bot.VOTE_INTERVAL_H}",
            f"VOTE_JITTER_MIN={self.bot.VOTE_JITTER_MIN}",
            f"VOTE_JITTER_MAX={self.bot.VOTE_JITTER_MAX}",
            f"TOPGG_URL={self.bot.TOPGG_URL}",
            f"CHROME_PATH={self.chrome_path.get().strip()}",
            f"CHROME_PROFILE={self.chrome_prof.get().strip()}",
            f"ENABLE_RAID={'1' if self.en_raid.get()==1 else '0'}",
            f"RAID_INTERVAL_H={float(self.raid_h.get())}",
            f"RAID_JITTER_MIN={self.bot.RAID_JITTER_MIN}",
            f"RAID_JITTER_MAX={self.bot.RAID_JITTER_MAX}",
            "",
        ]
        with open(".env","w",encoding="utf-8") as f:
            f.write("\n".join(lines))
        logging.info("💾 Saved .env")

    def _vote_now(self):
        threading.Thread(target=self.bot._perform_vote_once, daemon=True).start()

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except KeyboardInterrupt:
        pass
