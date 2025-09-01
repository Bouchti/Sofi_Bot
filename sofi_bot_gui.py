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
from dotenv import load_dotenv, dotenv_values
from rapidfuzz import fuzz
import easyocr
import discum
from discum.utils.button import Buttoner

# --- UI (built-in, free) ---
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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

def extract_card_for_index(index, card_crop):
    try:
        processed = preprocess_image(card_crop)
        gens, name, series = extract_generation_with_easyocr(processed)
        return index, (gens, name, series)
    except Exception as e:
        logging.warning(f"Failed to process card index {index}: {e}")
        return index, ({}, "", "")

def extract_card_generations(image):
    card_width = image.width // 3
    card_info = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i in range(3):
            left = i * card_width
            right = left + card_width
            crop = image.crop((left, 0, right, image.height))
            futures.append(executor.submit(extract_card_for_index, i, crop))
        for future in futures:
            i, (gens, name, series) = future.result()
            if gens or name:
                card_info[i] = {"generations": gens or {}, "name": name or "", "series": series or ""}
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
        self.sent_initial_sd = False
        self.bot_identity_logged = False

        # config defaults
        load_dotenv(self.cfg_path)
        self.TOKEN = os.getenv("DISCORD_TOKEN", "")
        self.GUILD_ID = os.getenv("GUILD_ID", "")
        self.CHANNEL_ID = os.getenv("CHANNEL_ID", "")
        self.USER_ID = os.getenv("USER_ID", "")
        self.SD_INTERVAL_SEC = int(os.getenv("SD_INTERVAL_SEC", "480"))
        self.USE_GPU = (os.getenv("OCR_USE_GPU", "0") == "1")

        # leaderboard
        self.TOP_CHARACTERS_LIST, self.TOP_CHARACTERS_SET = load_top_characters(self.leaderboard_path)

    # --------- Threads ---------
    def message_watchdog(self):
        while not self.stop_event.is_set():
            time.sleep(10)
            elapsed = time.time() - self.last_message_received
            if elapsed > self.WATCHDOG_TIMEOUT:
                logging.error(f"No message received for {elapsed:.1f}s. Restarting bot process...")
                self.restart_process()

    def reset_claim_if_timed_out(self):
        while not self.stop_event.is_set():
            time.sleep(1)
            if self.pending_claim["triggered"] and time.time() - (self.pending_claim["timestamp"] or 0) > 7:
                logging.warning("No Sofi confirmation received. Resetting claim trigger.")
                self.pending_claim["triggered"] = False

    def periodic_sd_sender(self):
        """Wait until initial 'sd' is sent (in on_ready), then send every 8 minutes."""
        # wait until on_ready sends the first 'sd' or we stop
        while not self.stop_event.is_set() and not self.sent_initial_sd:
            time.sleep(0.2)

        while not self.stop_event.is_set():
            wait_time = 480 + random.uniform(0, 10)  # ~8min + jitter
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
                    logging.error("Gateway appears frozen. Restarting process...")
                    self.restart_process()
            except Exception as e:
                logging.error(f"keep_alive check failed: {e}")

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
        claimed_indexes = []

        # first pass: claim any no-generation card quickly
        for i in range(card_count):
            generations = card_info.get(i, {}).get("generations", {})
            if not generations:
                self.click_discord_button(button_ids[i], channel_id, guild_id, m)
                card_claimed_time = time.time()
                elapsed_time = card_claimed_time - image_received_time
                time.sleep(3)
                logging.info(f"Claimed card {i+1} (no generation) in {elapsed_time:.2f}s")
                claimed_indexes.append(i)

        # choose best among remaining
        best_match_index = None
        best_score = -1
        best_likes = -1
        best_match_name = None
        best_match_series = None
        lowest_gen_index = None
        lowest_gen_value = float("inf")
        gen_below_10_index = None

        for i in range(card_count):
            if i in claimed_indexes or i not in card_info:
                continue
            info = card_info[i]
            name = info.get("name")
            series = info.get("series")
            generations = info.get("generations", {})

            if name and series:
                match = find_best_character_match(name, series, self.TOP_CHARACTERS_LIST)
                if match:
                    score, _, likes, matched_name, matched_series = match
                    if score is not None and score >= 60:
                        if score > best_score or (score == best_score and likes > best_likes):
                            best_score = score
                            best_likes = likes
                            best_match_index = i
                            best_match_name = matched_name
                            best_match_series = matched_series

            if generations:
                gen_value = min(generations.values())
                if gen_value < 10 and gen_below_10_index is None:
                    gen_below_10_index = i
                if gen_value < lowest_gen_value:
                    lowest_gen_value = gen_value
                    lowest_gen_index = i

        chosen_index = (
            gen_below_10_index
            if gen_below_10_index is not None
            else best_match_index
            if best_match_index is not None
            else lowest_gen_index
            if lowest_gen_index is not None
            else next((i for i in range(card_count) if i not in claimed_indexes), None)
        )

        now = time.time()
        generations = card_info.get(chosen_index, {}).get("generations", {})

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

        if chosen_index is not None:
            self.click_discord_button(button_ids[chosen_index], channel_id, guild_id, m)
            self.pending_claim["timestamp"] = now
            self.pending_claim["triggered"] = True
            self.pending_claim["user_id"] = self.USER_ID
            card_claimed_time = time.time()
            elapsed_time = card_claimed_time - image_received_time

            if best_match_index == chosen_index:
                logging.info(f"Claimed card {chosen_index+1} (best match) in {elapsed_time:.2f}s")
            elif gen_below_10_index == chosen_index:
                logging.info(f"Claimed card {chosen_index+1} (generation < 10) in {elapsed_time:.2f}s")
            else:
                logging.info(f"Claimed card {chosen_index+1} (lowest generation) in {elapsed_time:.2f}s")
        else:
            logging.warning("No card chosen to claim.")

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
                for attempt in range(1, 6):  # small retry loop in case session_id races
                    try:
                        if self.bot and self.bot.gateway.session_id:
                            self.bot.sendMessage(self.CHANNEL_ID, "sd")
                            self.sent_initial_sd = True
                            logging.info("📤 Sent initial 'sd' command right after on_ready.")
                            break
                        else:
                            logging.debug("Gateway ready but session_id not set yet; retrying…")
                    except Exception:
                        logging.exception("⚠️ Failed to send initial 'sd'")
                    time.sleep(0.5)  # brief retry delay

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

            with self.last_drop_id_lock:
                # update last drop id (not displayed, but kept for parity)
                pass

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
        # warm-up: speeds up first pass, and downloads model if needed
        _ = READER.readtext(np.zeros((100, 100, 3), dtype=np.uint8), detail=0)

        # Init Discum
        self.bot = discum.Client(token=self.TOKEN, log=False)
        self._install_handlers()

        self.stop_event.clear()
        self.pending_claim["user_id"] = self.USER_ID
        self.bot_identity_logged = False

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
        logging.info("Bot stopped.")

    def restart_process(self):
        # Graceful stop and re-exec current program with same args
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
        }
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            for k, v in data.items():
                f.write(f"{k}={v}\n")
        logging.info(f"Saved settings to {self.cfg_path}")

# --------------------------
# Tkinter UI
# --------------------------
class TextHandler(logging.Handler):
    """Route logs to Tkinter safely."""
    def __init__(self, widget: ScrolledText):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record) + "\n"
        try:
            self.widget.after(0, self._append, msg)
        except Exception:
            pass

    def _append(self, msg):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, msg)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")

class SofiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sofi Farming Bot (Discum + OCR)")
        self.geometry("980x650")
        self.minsize(900, 600)

        # Bot manager
        self.manager = SofiBotManager()

        # UI
        self._build_ui()
        self._wire_logging()

        # preload fields
        self._load_fields_from_manager()
        self._start()

        # OS signals to close cleanly if run as script
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

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
    app = SofiApp()
    app.mainloop()
