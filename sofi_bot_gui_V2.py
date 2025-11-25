# -*- coding: utf-8 -*-
"""
Sofi Farming Bot — GUI + Top.gg voter (Chrome profile reuse)

DISCLAIMER: Using self-bots violates Discord ToS. You assume all risk.
"""

import os, re, sys, time, random, threading, signal, logging, tempfile, shutil
from io import BytesIO
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor

# ---------------- lib deps ----------------
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from PIL import Image
import cv2
import easyocr
import discum
from dotenv import load_dotenv

# UI
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
# Theming (ttkthemes)
try:
    from ttkthemes import ThemedTk
    HAVE_TTKTHEMES = True
except Exception:
    HAVE_TTKTHEMES = False

# Selenium (Top.gg voting)
HAVE_SELENIUM = True
try:
    import contextlib
    import chromedriver_autoinstaller
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
except Exception:
    HAVE_SELENIUM = False

# ---------------- logging ----------------
LOG_FILE = "Sofi_bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logging.getLogger("websocket").setLevel(logging.CRITICAL)
logging.getLogger("discum.gateway.gateway").setLevel(logging.ERROR)

# ---------------- HTTP ----------------
def http_session() -> requests.Session:
    retry = Retry(
        total=5, connect=5, read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=False,
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16))
    s.mount("http://",  HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16))
    s.headers.update({"User-Agent": "SofiBot/2.0 (+requests)"})
    return s

def pil_from_url(s: requests.Session, url: str, timeout=15.0) -> Optional[Image.Image]:
    try:
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        logging.warning(f"Image fetch failed: {e}")
        return None

# ---------------- OCR (global) ----------------
READER = None
def preproc(gray_or_rgb):
    if isinstance(gray_or_rgb, Image.Image):
        gray_or_rgb = np.array(gray_or_rgb)
    if gray_or_rgb.ndim == 3:
        return cv2.cvtColor(gray_or_rgb, cv2.COLOR_RGB2GRAY)
    return gray_or_rgb

def _clean_gen_text(texts: List[str]) -> Dict[str, int]:
    out = {}
    for t in texts:
        if len(out) >= 3: break
        t = (t or "").strip()
        if not t: continue
        # Special case: 6G123
        m6 = re.match(r"^6G(\d{1,4})$", t.upper())
        if m6:
            out[f"G{m6.group(1)}"] = int(m6.group(1))
            continue
        z = re.sub(r"[^a-zA-Z0-9]", "", t)
        z = (z.replace("i","1").replace("I","1")
               .replace("o","0").replace("O","0")
               .replace("g","9").replace("s","5").replace("S","5")
               .replace("B","8").replace("l","1"))
        if z and z[0] in ("0","6","5","9") and not z.upper().startswith("G"):
            z = "G"+z[1:]
        m = re.match(r"^G(\d{1,4})$", z.upper())
        if m:
            out[f"G{m.group(1)}"] = int(m.group(1))
    return out

def _extract_name_series(texts: List[str]):
    if not texts or len(texts) < 2: return None, None
    gi = -1
    for i,t in enumerate(texts):
        if _clean_gen_text([t]): gi = i; break
    if gi == -1: return None, None
    name  = (texts[gi+1] if gi+1 < len(texts) else None) or ""
    series= (texts[gi+2] if gi+2 < len(texts) else None) or ""
    return name.strip(), series.strip()

def extract_card_for_index(i: int, img: Image.Image):
    try:
        # Convert to RGB array
        rgb = np.array(img) if isinstance(img, Image.Image) else img

        # V2 uses 'preproc', not '_preproc_image'
        gray = preproc(rgb)

        # Prepare BGR for elite detector
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Resize for OCR
        resized = cv2.resize(
            gray,
            (300, int(gray.shape[0] * (300.0 / gray.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )

        # OCR text
        texts = READER.readtext(resized, detail=0, batch_size=2)

        # Extract gens / name / series
        gens = _clean_gen_text(texts)
        name, series = _extract_name_series(texts)

        logging.info(
            f"✅ Card {i}: Gen={list(gens.keys()) or '∅'}, "
            f"Name='{name or ''}', Series='{series or ''}'"
        )

        # IMPORTANT: V2 EXPECTS (i, dict)
        return i, {
            "generations": gens,
            "name": name or "",
            "series": series or ""
        }

    except Exception as e:
        logging.warning(f"card {i} OCR error: {e}")
        return i, {"generations": {}, "name": "", "series": ""}


def extract_cards(img: Image.Image) -> Dict[int, dict]:
    w = img.width//3
    def crop(k):
        L = k*w
        R = L+w if k<2 else img.width
        return img.crop((L,0,R,img.height))
    out = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        for fut in [ex.submit(extract_card_for_index, k, crop(k)) for k in range(3)]:
            i, d = fut.result()
            out[i] = d
            logging.info(f"✅ Card {i}: Gen={list((d['generations'] or {}).keys()) or '∅'}, Name='{d['name']}', Series='{d['series']}'")
    return out

# ---------------- Likes parser ----------------
def parse_likes(label: str) -> Optional[int]:
    # Supports "1.9K", "2K", "3M" (rare), and "2,345"
    s = (label or "").replace(",", "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)([kKmM]?)", s)
    if not m:
        return None
    val = float(m.group(1))
    suf = m.group(2).lower()
    if suf == "k": val *= 1000
    elif suf == "m": val *= 1_000_000
    return int(round(val))

# ---------------- Selenium helpers ----------------
def _find_chrome_win() -> Optional[str]:
    for p in [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]:
        if os.path.isfile(p): return p
    return shutil.which("chrome") or shutil.which("chrome.exe")

def _is_writable(p: str) -> bool:
    try:
        os.makedirs(p, exist_ok=True)
        tf = os.path.join(p, ".wtest")
        with open(tf,"w") as f: f.write("ok")
        os.remove(tf)
        return True
    except Exception:
        return False
    
class TopGGVoter:
    def __init__(
        self,
        url: str,
        profile_dir: str = None,
        profile_name: str = None,
        chrome_binary: str = None,
        wait_secs: int = 10,
        attach_debugger: bool = False,
        debugger_address: str | None = None,
    ):
        self.url = url
        self.profile_dir = (profile_dir or "").strip()
        self.profile_name = (profile_name or "").strip()
        self.chrome_binary = (chrome_binary or "").strip()
        self.wait_secs = max(1, int(wait_secs))
        self.attach_debugger = bool(attach_debugger)
        self.debugger_address = (debugger_address or "").strip()

    def _driver(self):
        # Ensure driver exists (Selenium still needs it even for attach)
        try:
            chromedriver_autoinstaller.install()
        except Exception as e:
            logging.warning(f"[vote] chromedriver_autoinstaller issue: {e}")

        opts = webdriver.ChromeOptions()

        # --- ATTACH MODE (no profile flags!) ---
        if self.attach_debugger and self.debugger_address:
            opts.debugger_address = self.debugger_address
            logging.info(f"[vote] Attaching to Chrome at {self.debugger_address} (debugger).")
            try:
                drv = webdriver.Chrome(options=opts)
                drv.set_window_size(1200, 900)
                return drv
            except Exception as e:
                logging.warning(f"[vote] attach failed: {e}")
                # Do NOT fall back to launching with the locked profile; just bubble up.
                raise

        # --- LAUNCH MODE (used only if not attaching) ---
        if not self.chrome_binary and sys.platform.startswith("win"):
            auto = _find_chrome_win()
            if auto:
                self.chrome_binary = auto
        if self.chrome_binary and os.path.isfile(self.chrome_binary):
            opts.binary_location = self.chrome_binary

        if self.profile_dir:
            if not os.path.isdir(self.profile_dir):
                logging.warning(f"[vote] profile dir not found: {self.profile_dir}")
            elif not _is_writable(self.profile_dir):
                logging.warning(f"[vote] profile dir not writable: {self.profile_dir}")
            else:
                opts.add_argument(f"--user-data-dir={self.profile_dir}")
        if self.profile_name:
            opts.add_argument(f"--profile-directory={self.profile_name}")

        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--no-first-run")
        opts.add_argument("--no-default-browser-check")
        opts.add_argument("--disable-notifications")
        opts.add_argument("--remote-debugging-port=0")
        opts.add_argument("--remote-allow-origins=*")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        drv = webdriver.Chrome(options=opts)
        drv.set_window_size(1200, 900)
        return drv

    def _goto(self, url):
        self.driver.set_page_load_timeout(self.page_load_timeout)
        self.driver.get(url)
        try:
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except Exception:
            pass

    def _accept_cookies_if_present(self):
        wait = WebDriverWait(self.driver, 3)
        with contextlib.suppress(Exception):
            for by, sel in [
                (By.XPATH, "//button[contains(., 'Accept')]"),
                (By.XPATH, "//button[contains(., 'I agree')]"),
                (By.XPATH, "//button[contains(., 'Got it')]"),
            ]:
                btn = wait.until(EC.element_to_be_clickable((by, sel)))
                btn.click()
                time.sleep(0.2)
                break

    def _find_vote_button(self, timeout=10):
        wait = WebDriverWait(self.driver, timeout)
        locators = [
            (By.XPATH, "//button[.//span[contains(translate(., 'VOTE', 'vote'), 'vote')]]"),
            (By.XPATH, "//button[contains(translate(., 'VOTE', 'vote'), 'vote')]"),
            (By.CSS_SELECTOR, "button.Button__StyledButton-sc-1h3903b-0"),
            (By.XPATH, "//button[contains(@class,'Button') and not(@disabled)]"),
        ]
        for by, sel in locators:
            try:
                el = wait.until(EC.element_to_be_clickable((by, sel)))
                return el
            except TimeoutException:
                continue
        return None

    def _looks_logged_out(self):
        # Simple heuristics: login prompts or OAuth buttons visible
        if self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Log in')]"):
            return True
        if self.driver.find_elements(By.XPATH, "//a[contains(@href,'/login')]"):
            return True
        return False

    def _has_vote_confirmation(self):
        d = self.driver
        checks = [
            (By.XPATH, "//*[contains(translate(., 'THANKS FOR VOTING', 'thanks for voting'), 'thanks for voting')]"),
            (By.XPATH, "//*[contains(translate(., 'YOU ALREADY VOTED', 'you already voted'), 'you already voted')]"),
            (By.XPATH, "//*[contains(translate(., 'VOTE AGAIN IN', 'vote again in'), 'vote again in')]"),
            (By.XPATH, "//button[contains(., 'Voted') or contains(., 'Voted!') or @disabled='true' or @aria-disabled='true']"),
        ]
        for by, sel in checks:
            if d.find_elements(by, sel):
                return True
        return False

    def _extract_cooldown_seconds(self):
        """
        Try to read remaining cooldown like "Vote again in 11:43".
        Returns seconds or None if not found.
        """
        text = self.driver.page_source.lower()
        m = re.search(r"vote again in\s+(\d{1,2}):(\d{2})", text)
        if not m:
            return None
        mm, ss = int(m.group(1)), int(m.group(2))
        return mm * 60 + ss

    def _try_click(self, drv, xps, timeout=10):
        for xp in xps:
            try:
                el = WebDriverWait(drv, timeout).until(EC.element_to_be_clickable((By.XPATH, xp)))
                drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                el.click()
                return True
            except Exception:
                continue
        return False

    def vote_once(self, tag: str = "") -> bool:
        if not HAVE_SELENIUM:
            logging.warning("[vote] Selenium not installed")
            return False
        drv = None
        try:
            drv = self._driver()
            drv.get(self.url)
            # cookies (best-effort)
            try:
                WebDriverWait(drv, 6).until(
                    EC.any_of(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept')]")),
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'I agree')]")),
                    )
                ).click()
            except Exception:
                pass
            time.sleep(self.wait_secs)
            # If already logged, Vote should be visible
            if self._try_click(drv, ["//button[normalize-space()='Vote']",
                                     "//a[normalize-space()='Vote']",
                                     "//*[@data-qa='vote-button' or @data-testid='vote-button']"], timeout=8):
                logging.info("[vote] Vote clicked on Top.gg")
                return True
            # If not, try quick login flow (user should already be logged in via profile; this is fallback)
            self._try_click(drv, ["//a[normalize-space()='Log In']","//button[normalize-space()='Log In']"], timeout=6)
            self._try_click(drv, ["//button[contains(., 'Discord')]", "//a[contains(., 'Discord')]"], timeout=8)
            # After redirect back, attempt vote again
            time.sleep(5)
            if self._try_click(drv, ["//button[normalize-space()='Vote']",
                                     "//a[normalize-space()='Vote']",
                                     "//*[@data-qa='vote-button' or @data-testid='vote-button']"], timeout=8):
                logging.info("[vote] Vote clicked on Top.gg")
                return True
            logging.warning("[vote] Vote button not found")
            return False
        except Exception as e:
            logging.warning(f"[vote] failed: {e}")
            return False
        finally:
            try:
                if drv: drv.quit()
            except Exception:
                pass



# ---- background thread loop (non-blocking) ----
def _vote_loop(self):
        voter = TopGGVoter(
            self.VOTE_URL,
            self.VOTE_PROFILE_DIR or None,
            self.VOTE_PROFILE_NAME or "Default",
            self.VOTE_CHROME_BIN or None,
            wait_secs=10,
            attach_debugger=True,
            debugger_address="127.0.0.1:9222",
        )

        def do_vote(tag=""):
            try:
                ok = voter.vote_once(tag or "")
            except Exception as e:
                logging.warning(f"[vote] error: {e}")
                ok = False
            self.watchdog_grace_until = max(self.watchdog_grace_until, time.time() + 240)
            return ok

        # --- Startup vote ---
        logging.info("[vote] Attempting Top.gg vote (startup)…")
        ok = do_vote("(startup)")
        wait = self.VOTE_INTERVAL_H * 3600 + random.uniform(0, self.VOTE_JITTER_MIN * 60)
        self.next_vote_at = time.time() + wait
        logging.info(f"[vote] Vote {'OK' if ok else 'FAILED'} — next in ~{int(wait)}s")

        # --- Main vote loop ---
        while not self.stop_evt.is_set():
            rem = self.next_vote_at - time.time()
            if rem <= 0:
                ok = do_vote()
                wait = self.VOTE_INTERVAL_H * 3600 + random.uniform(0, self.VOTE_JITTER_MIN * 60)
                self.next_vote_at = time.time() + wait
                logging.info(f"[vote] Vote {'OK' if ok else 'FAILED'} — next in ~{int(wait)}s")

            self.stop_evt.wait(min(30.0, max(1.0, rem)))
class SofiBotManager:
    SOFI_BOT_ID = "853629533855809596"

    def __init__(self, env_path=".env"):
        self.env_path = env_path
        load_dotenv(env_path)

        # .env
        self.TOKEN    = os.getenv("DISCORD_TOKEN","")
        self.GUILD_ID = os.getenv("GUILD_ID","")
        self.CHANNEL  = os.getenv("CHANNEL_ID","")
        self.USER_ID  = os.getenv("USER_ID","")

        self.SD_INTERVAL = int(os.getenv("SD_INTERVAL_SEC","480"))
        self.MODE   = os.getenv("MODE","smart").strip().lower()
        self.P1     = os.getenv("NORM_P1","high_likes")
        self.P2     = os.getenv("NORM_P2","low_gen")
        self.P3     = os.getenv("NORM_P3","series")
        self.SERIES = os.getenv("SERIES_NAME","").strip()

        # smart scoring tunables
        self.T_LIKE = int(os.getenv("T_LIKE","10"))
        self.T_GEN  = int(os.getenv("T_GEN","40"))
        self.WL_MIN = float(os.getenv("WL_MIN","0.15"))
        self.WL_SPAN= float(os.getenv("WL_SPAN","0.70"))
        self.LIKES_LOG_DAMP = (os.getenv("LIKES_LOG_DAMP","1")=="1")

        # vote
        self.VOTE_ENABLED = (os.getenv("TOPGG_VOTE_ENABLED","0")=="1")
        self.VOTE_URL     = os.getenv("TOPGG_VOTE_URL","https://top.gg/bot/853629533855809596/vote")
        self.VOTE_PROFILE_DIR  = os.getenv("TOPGG_CHROME_PROFILE_DIR","")
        self.VOTE_PROFILE_NAME = os.getenv("TOPGG_CHROME_PROFILE_NAME","")
        self.VOTE_CHROME_BIN   = os.getenv("TOPGG_CHROME_BINARY","")
        self.VOTE_INTERVAL_H   = float(os.getenv("TOPGG_VOTE_INTERVAL_HOURS","12"))
        self.VOTE_JITTER_MIN   = float(os.getenv("TOPGG_VOTE_JITTER_MINUTES","7"))

        # state
        self.http = http_session()
        self.bot  = None
        self.stop_evt = threading.Event()
        self.gateway_ready = False
        self.last_dispatch = 0.0
        self.gateway_connect_started = 0.0

        self.pending_claim = {"triggered": False, "timestamp": 0.0, "user_id": self.USER_ID}
        self.CLAIM_CONFIRM_TIMEOUT = int(os.getenv("CLAIM_CONFIRM_TIMEOUT","15"))
        self.POST_ACORN_NORMAL_DELAY = float(os.getenv("POST_ACORN_NORMAL_DELAY","0.8"))
        self.last_processed_time = 0.0
        self.last_processed_lock = threading.Lock()
        self.PROCESS_COOLDOWN = 240

        # voting guard
        self.in_voting = threading.Event()
        self._voting_lock = threading.Lock()
        self.next_vote_at = 0.0

        # watchdog
        self.WATCHDOG_TIMEOUT = int(os.getenv("WATCHDOG_TIMEOUT","600"))
        self.GATEWAY_READY_TIMEOUT = int(os.getenv("GATEWAY_READY_TIMEOUT","120"))
        # if we keep sending 'sd' but never see a Sofi drop, reboot after this many seconds
        self.SOFI_STALL_TIMEOUT = int(os.getenv("SOFI_STALL_TIMEOUT","900"))
        self.watchdog_grace_until = 0.0
        # threads
        self.t_gateway = None
        self.t_sd = None
        self.t_watch = None
        self.t_claim_to = None
        self.t_vote = None
        self.next_sd_at = 0.0           # unix ts for next 'sd'
        self.SD_BASE_INTERVAL = 480.0   # 8 minutes
        self.SD_JITTER_SEC   = 8.0      # small human-ish jitter
        # reboot guard
        self._reboot_lock = threading.Lock()
        self._last_reboot = 0.0
        self.REBOOT_MIN_INTERVAL = 20.0
        self.last_sd_ts = 0.0
        # Track last time we *saw* a Sofi drop fully (with components+image)
        self.last_sofi_drop_ts = 0.0
        # Track last time we *sent* an 'sd' command
        self.last_sd_sent_ts = 0.0
        self.MIN_SD_SGR_GAP = float(os.getenv("SOFI_MIN_SD_SGR_GAP", "6.0"))   # seconds
        self.STARTUP_SD_DELAY  = float(os.getenv("SOFI_STARTUP_SD_DELAY", "1.0"))

    # ---------- lifecycle ----------

        
    def start(self):
        global READER
        logging.info(f"🔎 Initializing EasyOCR (gpu=False) …")
        if READER is None:
            READER = easyocr.Reader(["en"], gpu=False, verbose=False)
            _ = READER.readtext(np.zeros((50,50,3),dtype=np.uint8), detail=0)

        self.stop_evt.clear()
        self.bot = discum.Client(token=self.TOKEN, log=False)
        self._install_handlers()

        self.gateway_ready = False
        self.last_dispatch = 0.0
        self.gateway_connect_started = time.time()
        self.watchdog_grace_until = time.time() + 180  # grace at (re)start

        # threads
        if not self.t_sd or not self.t_sd.is_alive():
            self.t_sd = threading.Thread(target=self._sd_loop, daemon=True); self.t_sd.start()
        if not self.t_watch or not self.t_watch.is_alive():
            self.t_watch = threading.Thread(target=self._watchdog_loop, daemon=True); self.t_watch.start()
        if not self.t_claim_to or not self.t_claim_to.is_alive():
            self.t_claim_to = threading.Thread(target=self._claim_timeout_loop, daemon=True); self.t_claim_to.start()
        if not hasattr(self, "t_vote") or not (self.t_vote and self.t_vote.is_alive()):
            self.t_vote = threading.Thread(target=_vote_loop, args=(self,), daemon=True)
            self.t_vote.start()

        def run_gw():
            logging.info("🔌 Connecting to Discord gateway …")
            try:
                self.bot.gateway.run(auto_reconnect=True)
            except Exception as e:
                logging.critical(f"Gateway run error: {e}")
        if not self.t_gateway or not self.t_gateway.is_alive():
            self.t_gateway = threading.Thread(target=run_gw, daemon=True); self.t_gateway.start()
        # send first 'sd' right away
        self.next_sd_at = time.time() + 1.0

    def stop(self):
        logging.info("🛑 Stopping bot …")
        self.stop_evt.set()
        try:
            if self.bot and self.bot.gateway:
                self.bot.gateway.close()
        except Exception:
            pass
        for t in (self.t_sd, self.t_watch, self.t_claim_to, self.t_vote):
            if t and t.is_alive():
                try: t.join(timeout=2.0)
                except Exception: pass
        if self.t_gateway and self.t_gateway.is_alive():
            try: self.t_gateway.join(timeout=3.0)
            except Exception: pass
        self.t_gateway = self.t_sd = self.t_watch = self.t_claim_to = self.t_vote = None
        logging.info("✅ Bot stopped.")

    def reboot(self, reason: str):
        now = time.time()
        if now - self._last_reboot < self.REBOOT_MIN_INTERVAL:
            logging.warning(f"⏳ Reboot suppressed (too soon). Reason: {reason}")
            return
        if not self._reboot_lock.acquire(False):
            return
        try:
            self._last_reboot = now
            logging.error(f"🔄 REBOOT — {reason}")
            self.stop()
            time.sleep(1.0)
            self.start()
        finally:
            self._reboot_lock.release()

    # ---------- threads ----------
    def _sd_loop(self):
     while not self.stop_evt.is_set():
        # If gateway not ready, wait a hair and retry
        if not (self.gateway_ready):
            logging.info("📤 waiting for gateway ready")
            self.stop_evt.wait(1.0)
            continue

        # Schedule bootstrap if not set
        if self.next_sd_at <= 0:
            self.next_sd_at = time.time() + 1.0
        ready = (self.gateway_ready and self.bot and getattr(self.bot.gateway, "session_id", None))    
        now = time.time()
        if now >= self.next_sd_at:
            try:
                if ready :
                    self.bot.sendMessage(self.CHANNEL, "sd")
                    logging.info("📤 drop 'sd' (sofi)")
                    now_ts = time.time()
                    self.last_sd_ts = now_ts
                    self.last_sd_sent_ts = now_ts
                else:
                    self.reboot("gateway not ready before sending Sd command")
                    return
                    # Treat our own sd as activity to keep watchdog calm
            except Exception as e:
                logging.warning(f"sd send error: {e}")

            # schedule next 'sd'
            wait = float(self.SD_INTERVAL) + random.uniform(0, self.SD_JITTER_SEC)
            self.next_sd_at = time.time() + wait

        # sleep until next tick
        self.stop_evt.wait(0.5)

    def _watchdog_loop(self):
        while not self.stop_evt.is_set():
            time.sleep(3)
            # 0) Critical threads must be alive
            tgw = self.t_gateway
            
            if tgw is not None and not tgw.is_alive():
                self.reboot("gateway thread died")
                return

            tsd = self.t_sd
            if tsd is not None and not tsd.is_alive():
                self.reboot("sd loop thread died")
                return
            
            # global grace window
            if time.time() < self.watchdog_grace_until:
                continue

            # skip *entirely* while voting
            if self.in_voting.is_set():
                continue

            # also skip READY timeout while voting just finished but READY not yet set
            if (not self.gateway_ready) and (time.time() < self.watchdog_grace_until):
                continue

            # READY timeout
            if (not self.gateway_ready) and self.gateway_connect_started:
                if time.time() - self.gateway_connect_started > self.GATEWAY_READY_TIMEOUT:
                    self.reboot("never reached READY")
                    return

            # no events for too long
            last = self.last_dispatch or 0.0
            now  = time.time()
            if last and (now - last) > self.WATCHDOG_TIMEOUT:
                self.reboot("no gateway events")
                return

            # Sofi stall: we keep sending 'sd' but we never see a Sofi drop
            if self.last_sd_sent_ts:
                # if we have never seen a drop OR the last drop is older than the last 'sd'
                if (not self.last_sofi_drop_ts) or (self.last_sofi_drop_ts < self.last_sd_sent_ts):
                    if (now - self.last_sd_sent_ts) > self.SOFI_STALL_TIMEOUT:
                        self.reboot("sd sent but no Sofi drop seen")
                        return


    def _claim_timeout_loop(self):
        while not self.stop_evt.is_set():
            time.sleep(0.5)
            if self.pending_claim["triggered"]:
                if (time.time() - self.pending_claim["timestamp"]) > self.CLAIM_CONFIRM_TIMEOUT:
                    logging.warning("⏱️ Claim confirmation timeout. Resetting trigger.")
                    self.pending_claim["triggered"] = False

    # ---------- gateway handlers ----------
    def _install_handlers(self):
        @self.bot.gateway.command
        def any_event(resp):
            try:
                self.last_dispatch = time.time()
            except Exception:
                pass

        @self.bot.gateway.command
        def ready(resp):
            if not resp.event.ready: return
            u = resp.parsed.auto().get("user")
            if u:
                logging.info(f"✅ READY as {u['username']}#{u['discriminator']}")
            self.gateway_ready = True
            self.next_sd_at = time.time() + self.STARTUP_SD_DELAY

            if self.next_sd_at <= 0:
                self.next_sd_at = time.time() + 1.0
            self.gateway_connect_started = 0.0
           

        @self.bot.gateway.command
        def on_message(resp):
            if not hasattr(resp,"raw") or resp.raw.get("t") != "MESSAGE_CREATE":
                return
            d = resp.raw["d"]
            m = resp.parsed.auto()

            author_id = str(d.get("author", {}).get("id"))
            if author_id != SofiBotManager.SOFI_BOT_ID:
                return

            channel_id = str(d.get("channel_id"))
            guild_id   = str(d.get("guild_id")) if d.get("guild_id") else None
            content    = d.get("content", "")
            comps      = d.get("components", []) or []
            atts       = d.get("attachments", []) or []
            embeds     = d.get("embeds", []) or []
            logging.info(
                f"[sofi-debug] ch={channel_id} comps={len(comps)} "
                f"atts={len(atts)} embeds={len(embeds)} content={content[:80]!r}"
            )
                # confirmation
            if self.pending_claim["triggered"]:
                uid = self.pending_claim["user_id"]
                content_l = content.lower()

                # Sofi confirmation patterns (robust)
                patterns = [
                    f"<@{uid}>",                  # must mention the user
                    "grabbed",                    # Sofi confirmation verb
                    "fought off",                 # Rare boss cards
                    "successfully grabbed",       # Sometimes used
                    "claimed the",                # Some variants
                    "took the",                   # Some servers
                ]

                if all(p in content_l for p in [f"<@{uid}>", "grab"]):
                    matched = True
                else:
                    matched = any(p in content_l for p in patterns)

                if matched:
                    logging.info(f"✅ Claim confirmed: {content[:200]}…")
                    with self.last_processed_lock:
                        self.last_processed_time = time.time()
                    self.pending_claim["triggered"] = False
                    return

            # --- Collect image URLs from attachments + embeds ---
            img_urls = []

            # 1) Attachments (old behavior)
            for a in atts:
                fn  = (a.get("filename", "") or "").lower()
                url = a.get("url")
                if url and any(fn.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
                    img_urls.append(url)

            # 2) Embeds: image + thumbnail
            for e in embeds:
                img = e.get("image") or {}
                url = img.get("url")
                if url:
                    img_urls.append(url)

                thumb = e.get("thumbnail") or {}
                turl = thumb.get("url")
                if turl:
                    img_urls.append(turl)

            # Require buttons + at least one image URL
            if not comps or not img_urls:
                return
            # Mark that we *saw* a Sofi drop with components+image
            self.last_sofi_drop_ts = time.time()

            logging.info(f"🎴 Sofi drop seen: comps={len(comps)} imgs={len(img_urls)}")

            def worker(url):
                pil = pil_from_url(self.http, url)
                if not pil:
                    return
                try:
                    self._pick_from_image(pil, comps, channel_id, guild_id, m)
                except Exception as e:
                    logging.warning(f"process drop error: {e}")

            # Limit to first 3 images (3 cards)
            for url in img_urls[:3]:
                threading.Thread(target=worker, args=(url,), daemon=True).start()

    # ---------- buttons & claims ----------
    def _btn_click(self, button_id, channel_id, guild_id, m, tag=""):
        try:
            self.bot.click(
                applicationID=m["author"]["id"],
                channelID=channel_id,
                guildID=m.get("guild_id"),
                messageID=m["id"],
                messageFlags=m["flags"],
                data={"component_type": 2, "custom_id": button_id},
            )
            if tag: logging.info(tag)
        except Exception as e:
            logging.warning(f"button click error: {e}")

    def _normal_click_with_confirm(self, button_id, channel_id, guild_id, m, reason):
        self._btn_click(button_id, channel_id, guild_id, m, tag=f"🔘 NORMAL click: {reason}")
        self.pending_claim.update(triggered=True, timestamp=time.time())
        # wait confirm
        t0 = time.time()
        while self.pending_claim["triggered"] and (time.time()-t0) < self.CLAIM_CONFIRM_TIMEOUT:
            time.sleep(0.25)
        if self.pending_claim["triggered"]:
            logging.warning("⏱️ No confirmation. Retrying once …")
            self.pending_claim["triggered"] = False
            time.sleep(0.4)
            self._btn_click(button_id, channel_id, guild_id, m, tag=f"🔘 NORMAL click (retry): {reason}")
            t1 = time.time()
            self.pending_claim.update(triggered=True, timestamp=t1)
            while self.pending_claim["triggered"] and (time.time()-t1) < self.CLAIM_CONFIRM_TIMEOUT:
                time.sleep(0.25)
            if self.pending_claim["triggered"]:
                logging.error("❌ Still no confirmation after retry.")
                self.pending_claim["triggered"] = False

    def _blocked(self, ignore=False) -> bool:
        if ignore: return False
        with self.last_processed_lock:
            cooldown = self.last_processed_time and (time.time()-self.last_processed_time < self.PROCESS_COOLDOWN)
            pend = self.pending_claim.get("triggered", False)
        if pend:
            logging.info("⏳ Skip — waiting previous claim confirmation.")
            return True
        if cooldown:
            logging.info("⏳ Skip — cooldown not expired.")
            return True
        return False
    def _is_elite_button(self, btn: dict) -> bool:
        """
        Sofi 'elite' card now uses a yellow heart on the claim button.
        We treat any yellow-heart unicode (with possible skin-tone variants)
        as elite. If the component has an emoji object, prefer that.
        """
        emo = btn.get("emoji") or {}
        name = (emo.get("name") or "").strip()
        YELLOW_HEARTS = {"💛", "💛🏻", "💛🏼", "💛🏽", "💛🏾", "💛🏿"}
        if name in YELLOW_HEARTS:
            return True
        label = (btn.get("label") or "").strip()
        return any(h in label for h in YELLOW_HEARTS)

    def _pick_from_image(self, img: Image.Image, components, channel_id, guild_id, m):
        # parse buttons (ignore links; ignore beyond first 3)
        pos_btns = []
        for row in components:
            for b in row.get("components", []):
                # Only real buttons with a custom_id
                if b.get("type") != 2 or not b.get("custom_id"):
                    continue

                label = str(b.get("label") or "")
                likes = parse_likes(label)  # None => acorn/event
                is_elite = self._is_elite_button(b)

                pos_btns.append({
                    "id": b["custom_id"],
                    "likes": likes,
                    "label": label,
                    "elite": is_elite,      # <- NEW
                })

        pos_btns = pos_btns[:3]
        if not pos_btns:
            logging.warning("No claimable buttons found."); return

        acorn_pos  = [i for i,b in enumerate(pos_btns) if b["likes"] is None]
        normal_pos = [i for i,b in enumerate(pos_btns) if b["likes"] is not None]

        cards = extract_cards(img)

        # 1) acorns first (claim all), non-blocking
        acorn_clicked = False
        for idx, p in enumerate(acorn_pos, start=1):
            try:
                self._btn_click(pos_btns[p]["id"], channel_id, guild_id, m, tag=f"🌰 Acorn click #{idx} (pos {p+1})")
                acorn_clicked = True
                time.sleep(0.25)
            except Exception as e:
                logging.warning(f"acorn click failed pos {p+1}: {e}")

        if not normal_pos:
            logging.info("🌰 All buttons are acorns — done.")
            return

        if acorn_clicked:
            time.sleep(self.POST_ACORN_NORMAL_DELAY)

        # assemble normal candidates
        cands = []
        for i in normal_pos:
            info = cards.get(i, {})
            gens = info.get("generations",{}) or {}
            min_gen = min(gens.values()) if gens else None
            cands.append({
                "pos": i,
                "likes": pos_btns[i]["likes"] or 0,
                "min_gen": min_gen,
                "has_gen": bool(gens),
                "series": info.get("series","") or "",
                "elite": bool(pos_btns[i].get("elite", False)),
            })

        # 2) ELITE
        e = next((c for c in cands if c["elite"]), None)
        if e:
            if not self._blocked(ignore=acorn_clicked):
                self._normal_click_with_confirm(pos_btns[e["pos"]]["id"], channel_id, guild_id, m, "ELITE")
            return

        # 3) ABS GEN < 10
        sub10 = [c for c in cands if (c["min_gen"] is not None and c["min_gen"] < 10)]
        if sub10:
            sub10.sort(key=lambda c: (c["min_gen"], -c["likes"], c["pos"]))
            ch = sub10[0]
            if not self._blocked(ignore=acorn_clicked):
                self._normal_click_with_confirm(pos_btns[ch["pos"]]["id"], channel_id, guild_id, m,
                                                f"ABS gen<10 | G{ch['min_gen']} | likes={ch['likes']}")
            return

        # 4) Smart/Normal
        series_key = (self.SERIES or "").lower()

        if self.MODE == "smart":
            likes_vals = [c["likes"] for c in cands]
            max_likes = max(likes_vals) if likes_vals else 0
            gen_vals  = [c["min_gen"] for c in cands if c["min_gen"] is not None]
            min_gen, max_gen = (min(gen_vals), max(gen_vals)) if gen_vals else (None, None)
            like_gap = (max(likes_vals)-min(likes_vals)) if likes_vals else 0
            gen_gap  = (max_gen-min_gen) if gen_vals else 0

            L = like_gap / (like_gap + self.T_LIKE) if like_gap>0 else 0.0
            Gtight = 1.0 - (gen_gap/(gen_gap + self.T_GEN)) if gen_gap>0 else 1.0
            wL = max(0.1, min(0.9, self.WL_MIN + self.WL_SPAN * (0.5*(L+Gtight))))
            wG = 1.0 - wL
            logging.info(f"⚖️ SMART weights: likes={wL:.2f}, gen={wG:.2f}  (Δlikes={like_gap}, Δgen={gen_gap})")

            def score(c):
                # likes (log damped by default)
                if max_likes>0:
                    like_norm = (np.log1p(c["likes"])/max(1e-9, np.log1p(max_likes))) if self.LIKES_LOG_DAMP else (c["likes"]/max_likes)
                else:
                    like_norm = 0.0
                if (c["min_gen"] is None or min_gen is None or max_gen is None or max_gen==min_gen):
                    gen_norm = 0.0
                    nog_bonus = 0.03 if not c["has_gen"] else 0.0
                else:
                    # lower gen -> higher score
                    gen_norm = (max_gen - c["min_gen"]) / max(1.0, (max_gen - min_gen))
                    nog_bonus = 0.0
                ser_bonus = 0.12 if (series_key and series_key in c["series"].lower()) else 0.0
                return wL*like_norm + wG*gen_norm + ser_bonus + nog_bonus

            for c in cands: c["score"]=score(c)
            cands.sort(key=lambda c: (-c["score"],
                                      c["min_gen"] if c["min_gen"] is not None else 10**9,
                                      -c["likes"], c["pos"]))
            ch = cands[0]
            if not self._blocked(ignore=acorn_clicked):
                self._normal_click_with_confirm(pos_btns[ch["pos"]]["id"], channel_id, guild_id, m,
                                                f"SMART score={ch['score']:.3f} | G{ch['min_gen'] if ch['min_gen'] is not None else '∅'} | likes={ch['likes']}")
            return

        # Normal mode priorities
        def apply_pref(pref, pool):
            if pref == "high_likes":
                mx = max(c["likes"] for c in pool) if pool else None
                return [c for c in pool if c["likes"]==mx] if mx is not None else []
            if pref == "low_gen":
                g = [c["min_gen"] for c in pool if c["min_gen"] is not None]
                if not g: return []
                mn = min(g)
                return [c for c in pool if c["min_gen"]==mn]
            if pref == "no_gen":
                return [c for c in pool if (c["min_gen"] is None or not c["has_gen"])]
            if pref == "series":
                if not series_key: return []
                return [c for c in pool if series_key in c["series"].lower()]
            return pool

        def tie_break(pool):
            pool.sort(key=lambda c: (
                c["min_gen"] if c["min_gen"] is not None else 10**9,
                -c["likes"], c["pos"]
            ))
            return pool[0]

        logging.info(f"🎯 NORMAL: {self.P1} > {self.P2} > {self.P3}")
        for pref in (self.P1, self.P2, self.P3):
            cand = apply_pref(pref, cands)
            if cand:
                ch = tie_break(cand)
                if not self._blocked(ignore=acorn_clicked):
                    self._normal_click_with_confirm(pos_btns[ch["pos"]]["id"], channel_id, guild_id, m,
                                                    f"NORMAL {self.P1}>{self.P2}>{self.P3} | G{ch['min_gen'] if ch['min_gen'] is not None else '∅'} | likes={ch['likes']}")
                return
        # fallback
        ch = tie_break(cands)
        if not self._blocked(ignore=acorn_clicked):
            self._normal_click_with_confirm(pos_btns[ch["pos"]]["id"], channel_id, guild_id, m,
                                            f"FALLBACK | G{ch['min_gen'] if ch['min_gen'] is not None else '∅'} | likes={ch['likes']}")

# ---------------- UI ----------------
class TextLogHandler(logging.Handler):
    def __init__(self, widget: ScrolledText):
        super().__init__()
        self.widget = widget
        self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    def emit(self, record):
        msg = self.format(record) + "\n"
        tag = ("CRIT" if record.levelno>=logging.CRITICAL else
               "ERR" if record.levelno>=logging.ERROR else
               "WARN" if record.levelno>=logging.WARNING else
               "INFO")
        try:
            self.widget.after(0, self._append, msg, tag)
        except Exception:
            pass
    def _append(self, msg, tag):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, msg, tag)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")


class SofiGUI:
    def __init__(self):
        if HAVE_TTKTHEMES:
            # Use ThemedTk
            self.root = ThemedTk(theme="arc")
        else:
            # Fallback to normal Tk
            self.root = tk.Tk()

        self.root.title("Sofi Farming Bot — with Top.gg voter")
        self.root.geometry("1080x820")
        self.root.minsize(980, 700)

        self.mgr = SofiBotManager()
        self._build_ui()
        self._wire_logging()
        self._load_from_env()
        self.root.after(150, self._start_now)

        # Bind kill signals
        signal.signal(signal.SIGINT, self._sig)
        signal.signal(signal.SIGTERM, self._sig)

    def run(self):
        self.root.mainloop()

    # ---- UI build ----
    def _build_ui(self):
        root = self.root

        style = ttk.Style()
        try:
            style.theme_use("aqua")
        except Exception:
            pass

        pad = {"padx":8, "pady":6}

        # Connection
        frm_conn = ttk.LabelFrame(root, text="Connection & Runtime")
        frm_conn.pack(fill="x", **pad)

        self.var_token = tk.StringVar()
        self.var_guild = tk.StringVar()
        self.var_chan  = tk.StringVar()
        self.var_user  = tk.StringVar()
        self.var_sd_int= tk.IntVar(value=480)

        ttk.Label(frm_conn, text="Discord Token").grid(row=0, column=0, sticky="w")
        ent_token = ttk.Entry(frm_conn, textvariable=self.var_token, width=64, show="•")
        ent_token.grid(row=0, column=1, sticky="ew", columnspan=3)
        self._show = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_conn, text="Show", variable=self._show, command=lambda: ent_token.config(show="" if self._show.get() else "•")).grid(row=0, column=4, sticky="w")

        ttk.Label(frm_conn, text="Guild ID").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_conn, textvariable=self.var_guild, width=24).grid(row=1, column=1, sticky="w")

        ttk.Label(frm_conn, text="Channel ID").grid(row=1, column=2, sticky="w")
        ttk.Entry(frm_conn, textvariable=self.var_chan, width=24).grid(row=1, column=3, sticky="w")

        ttk.Label(frm_conn, text="Your User ID").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm_conn, textvariable=self.var_user, width=24).grid(row=2, column=1, sticky="w")

        ttk.Label(frm_conn, text="'sd' Interval (sec)").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(frm_conn, from_=60, to=3600, increment=10, textvariable=self.var_sd_int, width=10).grid(row=2, column=3, sticky="w")

        for c in (1,3):
            frm_conn.grid_columnconfigure(c, weight=1)

        # Mode
        frm_mode = ttk.LabelFrame(root, text="Mode & Preferences")
        frm_mode.pack(fill="x", **pad)

        self.var_mode = tk.StringVar(value="smart")
        ttk.Label(frm_mode, text="Mode").grid(row=0, column=0, sticky="w")
        self.cmb_mode = ttk.Combobox(frm_mode, values=("smart","normal"), textvariable=self.var_mode, state="readonly", width=12)
        self.cmb_mode.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(frm_mode, text="Series (optional)").grid(row=0, column=2, sticky="w")
        self.var_series = tk.StringVar()
        ttk.Entry(frm_mode, textvariable=self.var_series, width=28).grid(row=0, column=3, sticky="w", padx=4)

        opts = ("high_likes","low_gen","no_gen","series")
        self.var_p1 = tk.StringVar(value="high_likes")
        self.var_p2 = tk.StringVar(value="low_gen")
        self.var_p3 = tk.StringVar(value="series")
        ttk.Label(frm_mode, text="Normal P1").grid(row=1, column=0, sticky="w")
        ttk.Combobox(frm_mode, values=opts, textvariable=self.var_p1, state="readonly", width=12).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(frm_mode, text="Normal P2").grid(row=1, column=2, sticky="w")
        ttk.Combobox(frm_mode, values=opts, textvariable=self.var_p2, state="readonly", width=12).grid(row=1, column=3, sticky="w")
        ttk.Label(frm_mode, text="Normal P3").grid(row=1, column=4, sticky="w")
        ttk.Combobox(frm_mode, values=opts, textvariable=self.var_p3, state="readonly", width=12).grid(row=1, column=5, sticky="w")

        # Voting
        frm_vote = ttk.LabelFrame(root, text="Top.gg Vote (every 12h)")
        frm_vote.pack(fill="x", **pad)

        self.var_vote_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_vote, text="Enable Top.gg Vote", variable=self.var_vote_enabled).grid(row=0, column=0, sticky="w")

        ttk.Label(frm_vote, text="Chrome Profile Dir").grid(row=0, column=1, sticky="e")
        self.var_prof_dir = tk.StringVar()
        ttk.Entry(frm_vote, textvariable=self.var_prof_dir, width=40).grid(row=0, column=2, sticky="w")

        ttk.Label(frm_vote, text="Profile Name").grid(row=1, column=1, sticky="e")
        self.var_prof_name = tk.StringVar()
        ttk.Entry(frm_vote, textvariable=self.var_prof_name, width=20).grid(row=1, column=2, sticky="w")

        ttk.Label(frm_vote, text="Chrome Binary").grid(row=2, column=1, sticky="e")
        self.var_chrome_bin = tk.StringVar()
        ttk.Entry(frm_vote, textvariable=self.var_chrome_bin, width=40).grid(row=2, column=2, sticky="w")

        self.var_vote_url = tk.StringVar(value="https://top.gg/bot/853629533855809596/vote")
        ttk.Label(frm_vote, text="Vote URL").grid(row=3, column=1, sticky="e")
        ttk.Entry(frm_vote, textvariable=self.var_vote_url, width=40).grid(row=3, column=2, sticky="w")

        self.var_vote_hrs = tk.DoubleVar(value=12.0)
        ttk.Label(frm_vote, text="Interval (hours)").grid(row=4, column=1, sticky="e")
        ttk.Spinbox(frm_vote, from_=1, to=24, increment=0.5, textvariable=self.var_vote_hrs, width=10).grid(row=4, column=2, sticky="w")

        self.var_vote_jitter = tk.DoubleVar(value=7.0)
        ttk.Label(frm_vote, text="Jitter (min)").grid(row=4, column=3, sticky="e")
        ttk.Spinbox(frm_vote, from_=0, to=30, increment=1.0, textvariable=self.var_vote_jitter, width=10).grid(row=4, column=4, sticky="w")

        self.lbl_next_vote = ttk.Label(frm_vote, text="Next vote: —")
        self.lbl_next_vote.grid(row=0, column=3, columnspan=2, sticky="w")

        ttk.Button(frm_vote, text="Vote Now (test)", command=self._open_voter_chrome).grid(row=1, column=3, sticky="w", padx=4)

        for c in (2,):
            frm_vote.grid_columnconfigure(c, weight=1)

        # Controls
        frm_ctl = ttk.Frame(root); frm_ctl.pack(fill="x", **pad)
        ttk.Button(frm_ctl, text="Apply (Reboot)", command=self._apply).pack(side="left", padx=6)
        ttk.Button(frm_ctl, text="Save .env", command=self._save_env).pack(side="left", padx=6)
        ttk.Button(frm_ctl, text="Stop", command=self._stop).pack(side="left", padx=6)
        ttk.Button(frm_ctl, text="Clear Logs", command=self._clear).pack(side="left", padx=6)
        ttk.Button(frm_ctl, text="Copy Logs", command=self._copy).pack(side="left", padx=6)

        # Logs
        frm_log = ttk.LabelFrame(root, text="Logs")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt = ScrolledText(frm_log, height=18)
        self.txt.pack(fill="both", expand=True)
        # tags
        self.txt.tag_config("INFO", foreground="#e0e0e0")
        self.txt.tag_config("WARN", foreground="#ffcc00")
        self.txt.tag_config("ERR",  foreground="#ff6666")
        self.txt.tag_config("CRIT", foreground="#ff3333")
        self.txt.config(bg="#111", fg="#e0e0e0", insertbackground="#e0e0e0")

        # small timer for next vote label
        self.root.after(1000, self._tick_vote_label)


    def _wire_logging(self):
        lh = TextLogHandler(self.txt)
        logging.getLogger().addHandler(lh)

    # ---- env I/O ----
    def _load_from_env(self):
        self.var_token.set(self.mgr.TOKEN)
        self.var_guild.set(self.mgr.GUILD_ID)
        self.var_chan.set(self.mgr.CHANNEL)
        self.var_user.set(self.mgr.USER_ID)
        self.var_sd_int.set(self.mgr.SD_INTERVAL)

        self.var_mode.set(self.mgr.MODE)
        self.var_series.set(self.mgr.SERIES)
        self.var_p1.set(self.mgr.P1); self.var_p2.set(self.mgr.P2); self.var_p3.set(self.mgr.P3)

        self.var_vote_enabled.set(self.mgr.VOTE_ENABLED)
        self.var_prof_dir.set(self.mgr.VOTE_PROFILE_DIR)
        self.var_prof_name.set(self.mgr.VOTE_PROFILE_NAME)
        self.var_chrome_bin.set(self.mgr.VOTE_CHROME_BIN)
        self.var_vote_url.set(self.mgr.VOTE_URL)
        self.var_vote_hrs.set(self.mgr.VOTE_INTERVAL_H)
        self.var_vote_jitter.set(self.mgr.VOTE_JITTER_MIN)

    def _save_env(self):
        data = {
            "DISCORD_TOKEN": self.var_token.get(),
            "GUILD_ID": self.var_guild.get(),
            "CHANNEL_ID": self.var_chan.get(),
            "USER_ID": self.var_user.get(),
            "SD_INTERVAL_SEC": str(self.var_sd_int.get()),
            "MODE": self.var_mode.get(),
            "NORM_P1": self.var_p1.get(),
            "NORM_P2": self.var_p2.get(),
            "NORM_P3": self.var_p3.get(),
            "SERIES_NAME": self.var_series.get(),
            "T_LIKE": "10", "T_GEN":"40", "WL_MIN":"0.15", "WL_SPAN":"0.70", "LIKES_LOG_DAMP":"1",
            "TOPGG_VOTE_ENABLED": "1" if self.var_vote_enabled.get() else "0",
            "TOPGG_VOTE_URL": self.var_vote_url.get(),
            "TOPGG_CHROME_PROFILE_DIR": self.var_prof_dir.get(),
            "TOPGG_CHROME_PROFILE_NAME": self.var_prof_name.get(),
            "TOPGG_CHROME_BINARY": self.var_chrome_bin.get(),
            "TOPGG_VOTE_INTERVAL_HOURS": str(self.var_vote_hrs.get()),
            "TOPGG_VOTE_JITTER_MINUTES": str(self.var_vote_jitter.get()),
            "WATCHDOG_TIMEOUT": "600",
            "GATEWAY_READY_TIMEOUT": "120",
            "CLAIM_CONFIRM_TIMEOUT": "15",
            "POST_ACORN_NORMAL_DELAY": "0.8",
        }
        with open(self.mgr.env_path, "w", encoding="utf-8") as f:
            for k,v in data.items():
                f.write(f"{k}={v}\n")
        logging.info(f"💾 Saved .env to {self.mgr.env_path}")

    # ---- controls ----
    def _apply(self):
        self._save_env()
        # propagate to manager
        self.mgr.__init__(self.mgr.env_path)
        self.mgr.reboot("Apply settings")

    def _start_now(self):
        self.mgr.start()

    def _stop(self):
        self.mgr.stop()

    def _clear(self):
        self.txt.configure(state="normal")
        self.txt.delete("1.0", tk.END)
        self.txt.configure(state="disabled")

    def _copy(self):
        try:
            txt = self.txt.get("1.0", tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            logging.info("📋 Logs copied to clipboard.")
        except Exception:
            pass

    def _open_voter_chrome(self):
        prof_dir = self.var_prof_dir.get().strip()
        prof_name = self.var_prof_name.get().strip() or "Default"
        chrome = self.var_chrome_bin.get().strip() or r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        if not prof_dir:
            messagebox.showwarning("Chrome", "Set Chrome Profile Dir first."); return
        try:
            import subprocess
            subprocess.Popen([chrome,
                            f"--remote-debugging-port=9222",
                            f"--user-data-dir={prof_dir}",
                            f"--profile-directory={prof_name}"])
            logging.info("[vote] Launched voter Chrome at 127.0.0.1:9222.")
        except Exception as e:
            logging.error(f"[vote] Failed to launch Chrome: {e}")
    def _vote_now(self):
        if not HAVE_SELENIUM:
            messagebox.showwarning("Vote", "Selenium not installed.")
            return
        if not self.var_vote_enabled.get():
            messagebox.showinfo("Vote", "Enable voting first.")
            return
        # one-off vote in a short thread that respects voting guards
        def do():
            got = self.mgr._voting_lock.acquire(False)
            if not got:
                logging.info("[vote] Already in progress.")
                return
            self.mgr.in_voting.set()
            try:
                voter = TopGGVoter(
                    self.var_vote_url.get(),
                    self.var_prof_dir.get(),
                    self.var_prof_name.get(),
                    self.var_chrome_bin.get(),
                    wait_secs=10,
                )
                ok = voter.vote_once()
                # do not change schedule; this is a manual test
                logging.info(f"[vote] Manual vote {'OK' if ok else 'FAILED'}.")
            finally:
                self.mgr.in_voting.clear()
                try: self.mgr._voting_lock.release()
                except Exception: pass
        threading.Thread(target=do, daemon=True).start()

    def _tick_vote_label(self):
        if self.mgr.VOTE_ENABLED and self.mgr.next_vote_at>0:
            rem = int(self.mgr.next_vote_at - time.time())
            if rem < 0: rem = 0
            hh = rem//3600; mm = (rem%3600)//60
            self.lbl_next_vote.config(text=f"Next vote in: {hh}h {mm}m")
        else:
            self.lbl_next_vote.config(text="Next vote: —")
            self.root.after(1000, self._tick_vote_label)


    def _sig(self, *args):
        try:
            self.mgr.stop()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass

if __name__ == "__main__":
    app = SofiGUI()
    app.run()
