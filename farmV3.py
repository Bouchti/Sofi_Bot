import os
import time
import logging
import discum
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import requests
import random
import cv2
from rapidfuzz import fuzz, process
import easyocr
import re
import threading
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Lock

last_drop_id_lock = Lock()
last_processed_time_lock = Lock()
# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")
CHANNEL_ID = os.getenv("CHANNEL_ID")
USER_ID = os.getenv("USER_ID")

pending_claim = {
    "message_id": None,
    "timestamp": None,
    "user_id": USER_ID,  # Replace with your exact Discord username (not tag)
    "triggered": False
}

# Track last time the bot processed a Sofi drop
last_processed_time = 0
PROCESS_COOLDOWN_SECONDS = 240  # 4 minutes
last_message_received = time.time()
WATCHDOG_TIMEOUT = 510  # seconds

# Initialize logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger("websocket").setLevel(logging.CRITICAL)
logging.getLogger("discum.gateway.gateway").setLevel(logging.ERROR)

# Initialize global variables
LAST_DROP_MESSAGE_ID = "0"
#first message ready
IS_READY = False
reader = easyocr.Reader(['en'], gpu=True, verbose=False)
# Warm-up with a dummy image to preload the model
_ = reader.readtext(np.zeros((100, 100, 3), dtype=np.uint8), detail=0)
# Initialize requests session
session = requests.Session()

# Create a stop event for threads
stop_event = threading.Event()

bot_identity_logged = False  # Global flag

def message_watchdog():
    global last_message_received  # <- include this
    while not stop_event.is_set():
        time.sleep(10)
        elapsed = time.time() - last_message_received
        if elapsed > WATCHDOG_TIMEOUT:
            logging.error(f"🛑 No message received for {elapsed:.1f} seconds. Restarting bot...")
            stop_event.set()
            time.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)


def reset_claim_if_timed_out():
    while True:
        time.sleep(1)
        if pending_claim["triggered"] and time.time() - pending_claim["timestamp"] > 7:
            logging.warning("⚠️ No Sofi confirmation received. Resetting claim trigger.")
            pending_claim["triggered"] = False

threading.Thread(target=reset_claim_if_timed_out, daemon=True).start()

def periodic_sd_sender(bot, stop_event):
    while not stop_event.is_set():
        wait_time = 480 + random.uniform(0, 10)
        logging.info(f"⏳ Waiting {wait_time} seconds before sending 'sd'...")
        stop_event.wait(wait_time)
        if stop_event.is_set():
            break

        # 🔐 Make sure session is active
        if not bot.gateway.session_id:
            logging.warning("⚠️ Gateway session is missing. Skipping 'sd' command.")
            continue

        try:
            bot.sendMessage(CHANNEL_ID, "sd")
            logging.info("📤 Sent 'sd' command.")
        except Exception as e:
            logging.exception("⚠️ Failed to send 'sd'")
def click_discord_button(custom_id, channel_id, guild_id):
    with last_drop_id_lock:
        message_id = LAST_DROP_MESSAGE_ID

    if not message_id or len(message_id) < 15:
        logging.warning("❌ Not sending interaction: Invalid or missing message ID.")
        return

    url = "https://discord.com/api/v9/interactions"
    headers = {
        "Authorization": TOKEN,
        "Content-Type": "application/json",
    }

    payload = {
        "type": 3,
        "guild_id": guild_id,
        "channel_id": channel_id,  # 👈 Use dynamic channel_id
        "message_id": message_id ,
        "application_id": "853629533855809596",  # Sofi
        "session_id": bot.gateway.session_id,
        "data": {
            "component_type": 2,
            "custom_id": custom_id
        }
    }
    logging.debug(f"""
📤   Preparing interaction payload:
     Message ID : {LAST_DROP_MESSAGE_ID}
     Channel ID : {channel_id}
     Guild ID   : {guild_id}
     Session ID : {bot.gateway.session_id}
    """)
    response = session.post(url, json=payload, headers=headers)
    if response.status_code == 204:
        logging.info(f"✅ Successfully clicked button with ID: {custom_id}")
    else:
        logging.warning(f"❌ Failed to click button. Status: {response.status_code}, Response: {response.text}")

def preprocess_string(s):
    tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', s).upper().split()
    return ' '.join(sorted(tokens))

def load_top_characters():
    try:
        with open("sofi_leaderboard.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)

            if isinstance(raw_data, list):
                entries = raw_data
            else:
                entries = raw_data.get("data", [])

            processed = []
            for entry in entries:
                char_name = entry.get("character")
                series = entry.get("series")
                likes = entry.get("likes", 0)

                if not char_name or not series:
                    logging.warning("⚠️ Entry missing 'character' or 'series' key, skipping entry.")
                    continue

                combined = preprocess_string(f"{char_name} - {series}")
                processed.append({
                    "full_name": combined,
                    "likes": likes
                })

            processed_set = {entry["full_name"] for entry in processed}
            return processed, processed_set
    except FileNotFoundError:
        logging.warning("⚠️ 'sofi_leaderboard.json' not found.")
        return [], set()
    
TOP_CHARACTERS_LIST, TOP_CHARACTERS_SET = load_top_characters()

def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # nothing else

def extract_generation_with_easyocr(image):
    start_time = time.time()

    def clean_generation_text(ocr_results):
        cleaned_generations = {}
        logging.debug(f"🔍 Raw OCR Results: {ocr_results}")

        for text in ocr_results:
            if len(cleaned_generations) >= 3:
                break

            original_text = text.strip()
            text = re.sub(r'[^a-zA-Z0-9]', '', original_text)

            # Apply replacements only for generation
            text = text.replace("i", "1").replace("I", "1") \
                       .replace("o", "0").replace("O", "0") \
                       .replace("g", "9").replace("s", "5").replace("S", "5") \
                       .replace("B", "8").replace("l", "1")

            if text.startswith(("0", "6", "5", "9")) and not text.upper().startswith("G"):
                text = "G" + text[1:]

             # Fix leading digit + G (e.g., 6G1914 -> G1914)
            if re.match(r'^\dG\d+$', text.upper()):
                text = "G" + text[2:]

            match = re.match(r'^G(\d{1,4})$', text.upper())
            if match:
                gen_number = f"G{match.group(1)}"
                cleaned_generations[gen_number] = int(match.group(1))
                logging.debug(f"✅ Final Processed: {original_text} → {gen_number}")
            else:
                logging.debug(f"❌ Rejected: {original_text} → {text}")

        logging.debug(f"🚀 FINAL NORMALIZED GENERATIONS: {cleaned_generations}")
        return cleaned_generations

    def extract_card_name_and_series(ocr_results):
        logging.debug(f"🔍 Raw OCR Results: {ocr_results}")
        try:
            if not ocr_results or len(ocr_results) < 2:
                return "", ""

            gen_index = -1
            for i, text in enumerate(ocr_results):
                if clean_generation_text([text]):
                    gen_index = i
                    break

            if gen_index == -1:
                return "", ""

            name = ocr_results[gen_index + 1].strip() if gen_index + 1 < len(ocr_results) else ""
            series = ocr_results[gen_index + 2].strip() if gen_index + 2 < len(ocr_results) else ""

            logging.debug(f"🧹 Card Name: {name}")
            logging.debug(f"🧹 Series Name: {series}")
            return name, series

        except Exception as e:
            logging.warning(f"⚠️ Error extracting name and series: {e}")
            return "", ""

    try:
        # Convert PIL to array
        if isinstance(image, Image.Image):
            image = np.array(image)

        target_width = 300
        scale_ratio = target_width / image.shape[1]
        image = cv2.resize(image, (target_width, int(image.shape[0] * scale_ratio)), interpolation=cv2.INTER_AREA)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        extracted_text = reader.readtext(image, detail=0, batch_size=2)
        logging.debug(f"🔍 EasyOCR Raw Extracted: {extracted_text}")

        generations = clean_generation_text(extracted_text)
        name, series = extract_card_name_and_series(extracted_text)

        time_taken = time.time() - start_time
        logging.debug(f"⏱️ Time taken for extract_generation_with_easyocr: {time_taken:.2f} seconds")

        return generations, name, series

    except Exception as e:
        logging.warning(f"⚠️ OCR failed in extract_generation_with_easyocr: {e}")
        return {}, "", ""

def extract_card_for_index(index, card_crop):
    try:
        processed = preprocess_image(card_crop)
        result = extract_generation_with_easyocr(processed)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("extract_generation_with_easyocr did not return (gens, name, series)")
        return index, result
    except Exception as e:
        logging.warning(f"⚠️ Failed to process card index {index}: {e}")
        return index, ({}, "", "")

def extract_card_generations(image):
    card_width = image.width // 3
    card_info = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(3):
            left = i * card_width
            right = left + card_width
            crop = image.crop((left, 0, right, image.height))
            futures.append(executor.submit(extract_card_for_index, i, crop))

        for future in futures:
            try:
                i, (gens, name, series) = future.result()
                if gens or name:
                    card_info[i] = {
                        'generations': gens or {},
                        'name': name or "",
                        'series': series or ""
                    }
                    logging.debug(f"✅ Card {i}: Gen: {gens}, Name: {name}, Series: {series}")
                else:
                    logging.debug(f"❌ No valid data for card {i}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to extract data for card: {e}")

    logging.debug(f"🚀 FINAL CARD INFO: {card_info}")
    return card_info

def find_best_character_match(name, series):
    if not name:
        return None, 0, 0, None, None

    try:
        raw = f"{name} {series}".strip()
        normalized = preprocess_string(raw)

        full_names = []
        index_map = []

        for idx, entry in enumerate(TOP_CHARACTERS_LIST):
            if not isinstance(entry, dict):
                logging.warning(f"⚠️ Skipping non-dict entry in leaderboard: {entry}")
                continue

            character = entry.get("character", "")
            series_val = entry.get("series", "")
            full_name = entry.get("full_name")

            # Fallback: construct full_name safely
            if not full_name:
                parts = []
                if character:
                    parts.append(character)
                if series_val:
                    parts.append(series_val)
                full_name = " ".join(parts).strip()

            if not full_name:
                logging.warning(f"⚠️ Skipping entry due to missing name and series: {entry}")
                continue

            full_names.append(preprocess_string(full_name))
            index_map.append(idx)

        if not full_names:
            logging.warning("⚠️ No valid entries in leaderboard to match against.")
            return None, 0, 0, None, None

        best_match = process.extractOne(
            normalized,
            full_names,
            scorer=fuzz.token_sort_ratio
        )

        if best_match:
            match_string, score, match_idx = best_match
            leaderboard_idx = index_map[match_idx]
            matched_entry = TOP_CHARACTERS_LIST[leaderboard_idx]

            likes = matched_entry.get("likes", 0)
            matched_name = matched_entry.get("full_name", match_string)
            matched_series = matched_entry.get("series", "Unknown")

            return score, leaderboard_idx, likes, matched_name, matched_series

    except Exception as e:
        logging.warning(f"⚠️ Match failed for {name} - {series}: {e}")

    return None, 0, 0, None, None

def click_bouquet_then_best_from_image(pil_image, buttons_components, image_received_time, channel_id, guild_id):
    logging.info("🧠 Starting processing of the Sofi card image...")
    card_count = 3
    button_ids = [
        btn["custom_id"]
        for component_row in buttons_components
        for btn in component_row.get("components", [])
        if btn["type"] == 2 and btn["style"] in (1, 2)
    ]

    if len(button_ids) != 3:
        logging.warning("⚠️ Expected 3 buttons but found: %s", len(button_ids))
        return

    card_info = extract_card_generations(pil_image)
    logging.debug(f"Card info: {card_info}")

    claimed_indexes = []

    for i in range(card_count):
        generations = card_info.get(i, {}).get('generations', {})
        if not generations:
            click_discord_button(button_ids[i], channel_id, guild_id)
            card_claimed_time = time.time()
            elapsed_time = card_claimed_time - image_received_time
            time.sleep(3)
            logging.info(f"✅ Claimed card {i+1} (🌸 no generation 🌸) ⏱️ {elapsed_time:.2f}s")
            claimed_indexes.append(i)

    best_match_index = None
    best_score = -1
    best_likes = -1
    best_match_name = None
    best_match_series = None
    lowest_gen_index = None
    lowest_gen_value = float('inf')

    for i in range(card_count):
        if i in claimed_indexes or i not in card_info:
            continue

        info = card_info[i]
        name = info.get('name')
        series = info.get('series')
        generations = info.get('generations', {})

        if name and series:
            # Cut the length of series to match the OCR-truncated version
            truncated_series = series.strip()
            for entry in TOP_CHARACTERS_LIST:
                if entry.get("character", "") == name:
                    expected_series = entry["series"]
                    match = find_best_character_match(name, expected_series[:len(truncated_series)])
                    if match:
                        score, _, likes, matched_name, matched_series = match
                        if score is not None and score >= 85:
                            if score > best_score or (score == best_score and likes > best_likes):
                                best_score = score
                                best_likes = likes
                                best_match_index = i
                                best_match_name = matched_name
                                best_match_series = matched_series
                        else:
                            logging.debug(f"🔸 Skipped weak match: {matched_name} (score: {score:.2f})")

        if generations:
            gen_value = min(generations.values())
            if gen_value < lowest_gen_value:
                lowest_gen_value = gen_value
                lowest_gen_index = i

    if best_match_index is not None:
        logging.info(f"🔎 Best match: {best_match_name} from {best_match_series} with ❤️ {best_likes} likes (score: {best_score})")

    chosen_index = (
        best_match_index if best_match_index is not None
        else lowest_gen_index if lowest_gen_index is not None
        else next((i for i in range(card_count) if i not in claimed_indexes), None)
    )

    now = time.time()
    generations = card_info.get(chosen_index, {}).get('generations', {})

    if generations:
      with last_processed_time_lock:
        cooldown_active = (
            last_processed_time and
            (now - last_processed_time < PROCESS_COOLDOWN_SECONDS)
        )
        pending_active = pending_claim.get("triggered", False)

        if generations and (pending_active or cooldown_active):
            if pending_active:
                logging.info("⏱️ Skipping claim — waiting for confirmation of previous claim.")
            elif cooldown_active:
                logging.info("⏱️ Skipping claim — cooldown period not yet expired.")
            return

    if chosen_index is not None:
        click_discord_button(button_ids[chosen_index], channel_id, guild_id)
        pending_claim["timestamp"] = now
        pending_claim["triggered"] = True
        pending_claim["user_id"] = USER_ID

        card_claimed_time = time.time()
        elapsed_time = card_claimed_time - image_received_time

        if best_match_index == chosen_index:
            logging.info(f"✅ Claimed card {chosen_index+1} (⭐ best match ⭐)")
        else:
            logging.info(f"✅ Claimed card {chosen_index+1} (🥱 lowest generation 🥱)")

        logging.info(f"⏱️ Time to claim: {elapsed_time:.2f} seconds")
    else:
        logging.warning("⚠️ No card was chosen to be claimed.")



# Create Discum client
bot = discum.Client(token=TOKEN, log=False)


def keep_alive(bot):
    previous_latency = None
    failure_count = 0

    while not stop_event.is_set():
        time.sleep(30)
        try:
            latency = bot.gateway.latency
            logging.debug(f"📶 Gateway latency: {latency}")

            if latency is None or latency == previous_latency:
                failure_count += 1
                logging.debug(f"⚠️ Latency unchanged or missing ({failure_count}x).")
            else:
                failure_count = 0

            previous_latency = latency

            if failure_count >= 3:
                logging.error("🛑 Gateway appears frozen. Restarting process...")
                stop_event.set()
                time.sleep(1)
                # Restart the entire Python process
                os.execv(sys.executable, [sys.executable] + sys.argv)

        except Exception as e:
            logging.error(f"❌ keep_alive check failed: {e}")

@bot.gateway.command
def on_ready(resp):
    # Prefer Discum's native ready event
    if resp.event.ready:
        user = resp.parsed.auto().get('user')
        if user:
            logging.info(f"✅ Bot connected as {user['username']}#{user['discriminator']}")
        else:
            logging.info("✅ Bot connected, but user info is missing.")
        return



SOFI_BOT_ID = "853629533855809596"  # Official Sofi bot ID
@bot.gateway.command
def on_message(resp):
    global LAST_DROP_MESSAGE_ID, last_message_received, last_processed_time, pending_claim,  bot_identity_logged

    if not hasattr(resp, 'raw') or resp.raw.get('t') != "MESSAGE_CREATE":
        return

    data = resp.raw['d']
    author_id = str(data.get("author", {}).get("id"))
    channel_id = str(data.get('channel_id'))
    guild_id = str(data.get("guild_id"))
    content = data.get("content", "")
    username = data.get("author", {}).get("username", "")
    discriminator = data.get("author", {}).get("discriminator", "0000")

      # ✅ Log bot identity once (assuming message comes from us or SOFI bot)
    if not bot_identity_logged and author_id == USER_ID:
        logging.info(f"✅ Bot connected as {username} (ID: {author_id})")
        pending_claim["user_id"] = author_id  # Set this here as well
        bot_identity_logged = True

    # 🚫 Ignore messages not from Sofi bot
    if author_id != SOFI_BOT_ID:
        return

    # Update the last message received time
    last_message_received = time.time()

    # ✅ Check if it's a claim confirmation
    if pending_claim.get("triggered"):
        logging.debug(f"🕵️ Sofi message content: {content}")
        expected_grab = f"<@{pending_claim['user_id']}> **grabbed** the"
        expected_fight = f"<@{pending_claim['user_id']}> fought off"
        starts_with_grab = content.startswith(expected_grab)
        starts_with_fight = content.startswith(expected_fight)

        logging.debug(f"🧾 Checking if content starts with '{expected_grab}': {starts_with_grab}")
        logging.debug(f"🧾 Checking if content starts with '{expected_fight}': {starts_with_fight}")
        logging.debug(f"🧾 Final IF result: {starts_with_grab or starts_with_fight}")

        if starts_with_grab or starts_with_fight:
            logging.info(f"🎉 Claim confirmed: {content}")
            with last_processed_time_lock:
                last_processed_time = time.time()
            pending_claim["triggered"] = False
            return
        
    attachments = data.get("attachments", [])
    components = data.get("components", [])

    if not attachments or not components:
        return

    if components and len(components) > 0:
        with last_drop_id_lock:
            LAST_DROP_MESSAGE_ID = data.get("id")
            logging.debug(f"💾 LAST_DROP_MESSAGE_ID set to {LAST_DROP_MESSAGE_ID}")

    def process_sofi_drop(attachment_url, components, channel_id, guild_id):
        try:
            response = session.get(attachment_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_received_time = time.time()
            click_bouquet_then_best_from_image(image, components, image_received_time, channel_id, guild_id)
        except Exception as e:
            logging.warning(f"⚠️ Failed to process image: {e}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        for attachment in attachments:
            filename = attachment.get('filename', '')
            if any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                thread = threading.Thread(
                    target=process_sofi_drop,
                    args=(attachment.get('url'), components, channel_id, guild_id),
                    daemon=True
                )
                thread.start()

def signal_handler(sig, frame):
    """Handle termination signals to gracefully shutdown."""
    logging.info("🛑 Termination signal received. Shutting down...")
    stop_event.set()
    sys.exit(0)

# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Start the periodic sender using a simple thread
sd_thread = threading.Thread(target=periodic_sd_sender, args=(bot, stop_event), daemon=True)
sd_thread.start()

# Start gateway keep-alive checker
ka_thread = threading.Thread(target=keep_alive, args=(bot,), daemon=True)
ka_thread.start()

watchdog_thread = threading.Thread(target=message_watchdog, daemon=True)
watchdog_thread.start()

# Run the bot
logging.info("🔌 Connecting to Discord gateway...")

try:
    bot.gateway.run(auto_reconnect=True)
except Exception as e:
    logging.critical(f"❌ Failed to connect to Discord gateway: {e}")
    logging.critical("🔁 Please check your DISCORD_TOKEN, internet connection, or Discord API status.")
    sys.exit(1)