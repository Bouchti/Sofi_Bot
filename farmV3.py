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
# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")
CHANNEL_ID = os.getenv("CHANNEL_ID")
USER_ID = os.getenv("USER_ID")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("websocket").setLevel(logging.CRITICAL)
logging.getLogger("discum.gateway.gateway").setLevel(logging.ERROR)

# Initialize global variables
LAST_DROP_MESSAGE_ID = "0"
reader = easyocr.Reader(['en'], gpu=True)

# Initialize requests session
session = requests.Session()

# Create a stop event for threads
stop_event = threading.Event()

def periodic_sd_sender(bot, stop_event):
    """Send the 'sd' command periodically."""
    while not stop_event.is_set():
        wait_time = 480 + random.uniform(0, 10)  # 8 minutes + 0–10 seconds
        logging.info(f"⏳ Waiting {wait_time} seconds before sending 'sd'...")
        stop_event.wait(wait_time)
        if stop_event.is_set():
            break
        try:
            bot.sendMessage(CHANNEL_ID, "sd")
            logging.info("📤 Sent 'sd' command.")
        except Exception as e:
            logging.exception("⚠️ Failed to send 'sd'")

def click_discord_button(custom_id):
    """Click a Discord button using its custom ID."""
    url = "https://discord.com/api/v9/interactions"
    headers = {
        "Authorization": TOKEN,
        "Content-Type": "application/json",
    }
    payload = {
        "type": 3,
        "guild_id": GUILD_ID,
        "channel_id": CHANNEL_ID,
        "message_id": LAST_DROP_MESSAGE_ID,
        "application_id": "853629533855809596",  # Sofi's App ID
        "session_id": bot.gateway.session_id,
        "data": {
            "component_type": 2,
            "custom_id": custom_id
        }
    }
    response = session.post(url, json=payload, headers=headers)
    if response.status_code == 204:
        logging.info(f"✅ Successfully clicked button with ID: {custom_id}")
    else:
        logging.warning(f"❌ Failed to click button. Status: {response.status_code}, Response: {response.text}")

def preprocess_string(s):
    """Preprocess a string by removing punctuation, converting to uppercase, and sorting tokens."""
    tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', s).upper().split()
    return ' '.join(sorted(tokens))

def load_top_characters():
    """Load top characters from a file."""
    try:
        with open("top_sofi_characters.txt", "r", encoding="utf-8") as f:
            top_characters_list = [preprocess_string(name.strip()) for name in f.readlines()]
            return top_characters_list, set(top_characters_list)
    except FileNotFoundError:
        logging.warning("⚠️ 'top_sofi_characters.txt' not found. AI card evaluation will be skipped.")
        return [], set()

TOP_CHARACTERS_LIST, TOP_CHARACTERS_SET = load_top_characters()

def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # nothing else

def extract_card_names_from_ocr_results(ocr_results):
    """Extract card name from OCR results if index 0 becomes a valid generation after processing."""
    logging.debug(f"🔍 Raw OCR Results: {ocr_results}")

    if len(ocr_results) < 2:
        logging.warning("⚠️ Not enough OCR results to extract name.")
        return []

    # Try to clean and validate index 0 as a generation
    processed_gen = clean_generation_text([ocr_results[0]])
    if not processed_gen:
        logging.warning("⚠️ Index 0 is not a valid generation after processing.")
        return []

    # If generation is valid, assume index 1 is the name
    card_name = ocr_results[1].strip()
    logging.debug(f"🧹 Card Name: {card_name}")
    return [card_name]

def clean_generation_text(ocr_results):
    """Clean and normalize generation text extracted from OCR."""
    cleaned_generations = {}
    logging.debug(f"🔍 Raw OCR Results: {ocr_results}")
    for text in ocr_results:
        if len(cleaned_generations) >= 3:
            break
        original_text = text
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        text = text.replace("i", "1").replace("I", "1").replace("o", "0").replace("O", "0").replace("g", "9").replace("s", "5").replace("S", "5").replace("B", "8").replace("l", "1")
        if text.startswith("0") or text.startswith("6") or text.startswith("5") or text.startswith("9"):
            text = "G" + text[1:]
        match = re.match(r'^(G?)(\d{1,4})$', text)
        if match:
            prefix, gen_number = match.groups()
            if not prefix:
                gen_number = "G" + gen_number
            else:
                gen_number = prefix + gen_number
            if len(gen_number) > 5:
                gen_number = gen_number[:5]
            cleaned_generations[gen_number] = int(gen_number[1:])
            logging.debug(f"✅ Final Processed: {original_text} → {gen_number}")
    logging.debug(f"🚀 FINAL NORMALIZED GENERATIONS: {cleaned_generations}")
    return cleaned_generations

def extract_generation_with_easyocr(image):
    """Extract generation numbers and card names from an image using EasyOCR."""
    # Convert PIL Image to NumPy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)

    start_time = time.time()
    target_width = 400
    scale_ratio = target_width / image.shape[1]
    image = cv2.resize(image, (target_width, int(image.shape[0] * scale_ratio)), interpolation=cv2.INTER_AREA)
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if np.std(image) < 50:  # low contrast image    
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    extracted_text = reader.readtext(image, detail=0)
    logging.debug(f"🔍 EasyOCR Raw Extracted: {extracted_text}")

    # Extract generations and card names
    generations = clean_generation_text(extracted_text)
    card_names = extract_card_names_from_ocr_results(extracted_text)
    time_taken = time.time() - start_time
    logging.debug(f"⏱️ Time taken for extract_generation_with_easyocr: {time_taken:.2f} seconds")
    
    # Handle empty results
    if not generations and not card_names:
        logging.warning("⚠️ No valid generation numbers or card names found.")
    
    return generations, card_names

def extract_card_for_index(index, card_crop):
    processed = preprocess_image(card_crop)
    return index, extract_generation_with_easyocr(processed)

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
            i, (gens, names) = future.result()
            if gens or names:
                card_info[i] = {'generations': gens, 'names': names}
                logging.debug(f"✅ Card {i}: Generations: {gens}, Names: {names}")
            else:
                logging.debug(f"❌ No valid data for card {i}")

    logging.debug(f"🚀 FINAL CARD INFO: {card_info}")
    return card_info

def find_best_character_match(card_names):
    logging.debug(f"🔍 Starting character match for names: {card_names}")
    matches = []
    for name in card_names:
        try:
            normalized_name = preprocess_string(name)
            best_match = process.extractOne(
                normalized_name, 
                TOP_CHARACTERS_LIST, 
                scorer=fuzz.token_sort_ratio
            )
            if best_match:
                match_name, score = best_match[0], best_match[1]
                if score >= 95:                  
                    rank = TOP_CHARACTERS_LIST.index(match_name)
                    logging.info(f"⭐ Match found: {name} -> {match_name} with score {score} and rank {rank}")
                    matches.append((name, match_name, score, rank))
                else:
                    logging.debug(f"⚠️ Low score match: {name} -> {match_name} with score {score}")
            else:
                logging.info(f"❌ No match found for: {name}")
        except Exception as e:
            logging.error(f"Error matching {name}: {e}")
    return matches

def click_bouquet_then_best_from_image(pil_image, buttons_components, image_received_time):
    """Process the Sofi card image and click the appropriate buttons for cards without generation numbers."""
    logging.info("🧠 Starting processing of the Sofi card image...")
    card_count = 3
    button_ids = [btn["custom_id"] for component_row in buttons_components for btn in component_row.get("components", []) if btn["type"] == 2 and btn["style"] in (1, 2)]

    if len(button_ids) != 3:
        logging.warning("⚠️ Expected 3 buttons but found: %s", len(button_ids))
        return

    card_info = extract_card_generations(pil_image)
    logging.debug(f"Card info: {card_info}")

    claimed_indexes = []

    # Claim cards without generation numbers
    for i in range(card_count):
        generations = card_info.get(i, {}).get('generations', {})
        if not generations:
            click_discord_button(button_ids[i])
            # Record the time when the card is claimed
            card_claimed_time = time.time()
        
        # Calculate the elapsed time
            elapsed_time = card_claimed_time - image_received_time
            logging.info(f"⏱️ Time from image received to card claimed: {elapsed_time:.2f} seconds")
            # you can reduce this sleep but sofi bot can ignore your next clicks
            time.sleep(3)
            logging.info(f"✅ Claimed card {i+1} (no generation number)")
            claimed_indexes.append(i)

    # Loop only over remaining unclaimed cards
    best_match_index = None
    best_rank = float('inf')
    lowest_gen_index = None
    lowest_gen_value = float('inf')

    for i in range(card_count):
        if i in claimed_indexes or i not in card_info:
            continue

        names = card_info[i].get('names', [])
        generations = card_info[i].get('generations', {})

        # Find best character match
        matches = find_best_character_match(names)
        for _, match_name, score, rank in matches:
            if score >= 95 and rank < best_rank:
                best_rank = rank
                best_match_index = i

        # Track the card with the lowest generation number
        if generations:
            gen_value = min(generations.values())
            if gen_value < lowest_gen_value:
                lowest_gen_value = gen_value
                lowest_gen_index = i

    # Choose the best match or the card with the lowest generation number
    if best_match_index is not None:
        chosen_index = best_match_index
    elif lowest_gen_index is not None:
        chosen_index = lowest_gen_index
    else:
        # Fallback to first unclaimed card
        unclaimed = [i for i in range(card_count) if i not in claimed_indexes]
        chosen_index = unclaimed[0] if unclaimed else None

    if chosen_index is not None:
        click_discord_button(button_ids[chosen_index])
        if best_match_index is not None:
            logging.info(f"✅ Claimed card {chosen_index+1} (⭐best match⭐)")
        else:
            logging.info(f"✅ Claimed card {chosen_index+1} (🥱lowest generation🥱)")
        # Record the time when the card is claimed
        card_claimed_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = card_claimed_time - image_received_time
        logging.info(f"⏱️ Time from image received to card claimed: {elapsed_time:.2f} seconds")
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
                failure_count = 0  # Reset on healthy update

            previous_latency = latency

            if failure_count >= 3:
                logging.error("🛑 Gateway appears frozen. Reconnecting...")
                bot.gateway.close()
                time.sleep(3)
                bot.gateway.run(auto_reconnect=True)
                failure_count = 0  # Reset after reconnect

        except Exception as e:
            logging.error(f"❌ keep_alive check failed: {e}")

@bot.gateway.command
def on_message(resp):
    """Handle incoming messages from Discord."""
    global LAST_DROP_MESSAGE_ID
    logging.debug("🛎️ Event received")
    if not hasattr(resp, 'raw') or 't' not in resp.raw or resp.raw['t'] != "MESSAGE_CREATE":
        return
    data = resp.raw['d']
    channel_id = str(data.get('channel_id'))
    author_id = str(data.get('author', {}).get('id'))
    content = data.get('content', '')
    logging.debug(f"📨 Message from {author_id} in channel {channel_id} - Content: {content}")
    if author_id == "853629533855809596":
        attachments = data.get('attachments', [])
        components = data.get('components', [])
        LAST_DROP_MESSAGE_ID = data.get("id")
        for attachment in attachments:
            filename = attachment.get('filename', '')
            if any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                try:
                    logging.debug(f"📥 Found image attachment: {filename}")
                    response = session.get(attachment.get('url'))
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                     # Record the time when the image is received
                    image_received_time = time.time()
                    # Pass the image_received_time to the function
                    click_bouquet_then_best_from_image(image, components, image_received_time)
                except Exception as e:
                    logging.warning(f"⚠️ Failed to process image: {e}")
                return
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

# Run the bot
bot.gateway.run(auto_reconnect=True)