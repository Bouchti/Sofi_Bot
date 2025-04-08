import os
import time
import logging
import discum
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import requests
import threading
import random
from farmV2 import detect_bouquet_card, extract_card_generations, extract_card_names, find_best_character_match
LAST_DROP_MESSAGE_ID = "0"  # Track latest drop message to extract buttons from
# === Load environment variables ===
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")
CHANNEL_ID = os.getenv("CHANNEL_ID")
USER_ID = os.getenv("USER_ID")

logging.basicConfig(level=logging.DEBUG)

# Function to send the "sd" command periodically
def periodic_sd_sender():
    while True:
        wait_time = 480 + random.randint(0, 10)  # 8 minutes + 0–10 seconds
        logging.info(f"⏳ Waiting {wait_time} seconds before sending 'sd'...")
        time.sleep(wait_time)

        try:
            bot.sendMessage(CHANNEL_ID, "sd")
            logging.info("📤 Sent 'sd' command.")
        except Exception as e:
            logging.warning(f"⚠️ Failed to send 'sd': {e}")

# Start the periodic sender in a background thread
threading.Thread(target=periodic_sd_sender, daemon=True).start()


def click_discord_button(custom_id):
    url = "https://discord.com/api/v9/interactions"
    headers = {
        "Authorization": TOKEN,
        "Content-Type": "application/json",
    }

    session_id = bot.gateway.session_id  # ✅ Use the Discum session ID

    payload = {
        "type": 3,
        "guild_id": GUILD_ID,
        "channel_id": CHANNEL_ID,
        "message_id": LAST_DROP_MESSAGE_ID,
        "application_id": "853629533855809596",  # Sofi's App ID
        "session_id": session_id,  # ✅ REQUIRED by Discord
        "data": {
            "component_type": 2,
            "custom_id": custom_id
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 204:
        logging.info(f"✅ Successfully clicked button with ID: {custom_id}")
    else:
        logging.warning(f"❌ Failed to click button. Status: {response.status_code}, Response: {response.text}")


# === Placeholder for bouquet/gen logic ===
def click_bouquet_then_best_from_image(pil_image, buttons_components):
    logging.info("🧠 Starting processing of the Sofi card image...")
    card_count = 3
    card_width = pil_image.width // card_count
    if pil_image.width > 900:
        pil_image = pil_image.resize((900, int(pil_image.height * 900 / pil_image.width)))
    bouquet_card_index = detect_bouquet_card(pil_image)

    # Prepare button custom_ids from message component buttons
    button_ids = []
    for component_row in buttons_components:
        for btn in component_row.get("components", []):
            if btn["type"] == 2 and btn["style"] in (1, 2):  # type=2 is button
                button_ids.append(btn["custom_id"])

    if len(button_ids) != 3:
        logging.warning("⚠️ Expected 3 buttons but found: %s", len(button_ids))
        return

    if bouquet_card_index is not None:
        click_discord_button(button_ids[bouquet_card_index])
        logging.info(f"🌸 Claimed bouquet card at index {bouquet_card_index}")
        return

    # Remaining indices
    remaining_indices = [i for i in range(3) if i != bouquet_card_index]
    card_gens = extract_card_generations(pil_image)
    card_gens = {k: v for k, v in card_gens.items() if k in remaining_indices}

    best_match_index = None
    best_rank = float('inf')
    no_gen_card_index = None

    for i in remaining_indices:
        left = i * card_width
        right = (i + 1) * card_width
        card_crop = pil_image.crop((left, 0, right, pil_image.height))
        names = extract_card_names(np.array(card_crop))
        matches = find_best_character_match(names)

        for _, match_name, score, rank in matches:
            if score >= 95 and rank < best_rank:
                best_rank = rank
                best_match_index = i

        if i not in card_gens and no_gen_card_index is None:
            no_gen_card_index = i

    if no_gen_card_index is not None:
        chosen_index = no_gen_card_index
    elif best_match_index is not None:
        chosen_index = best_match_index
    elif card_gens:
        flattened_gens = [(index, gen) for index, gens in card_gens.items() for gen in gens.items()]
        sorted_card_gens = sorted(flattened_gens, key=lambda x: x[1][1])
        chosen_index = sorted_card_gens[0][0] if sorted_card_gens else remaining_indices[0]
    else:
        chosen_index = remaining_indices[0]

    click_discord_button(button_ids[chosen_index])
    logging.info(f"✅ Claimed secondary card at index {chosen_index}")

# === Create Discum client ===
bot = discum.Client(token=TOKEN, log=False)

logging.getLogger("websocket").setLevel(logging.CRITICAL)
logging.getLogger("discum.gateway.gateway").setLevel(logging.ERROR)

@bot.gateway.command
def on_message(resp):
    global LAST_DROP_MESSAGE_ID  # <-- THIS FIXES IT
    logging.debug("🛎️ Event received")

    if not hasattr(resp, 'raw') or 't' not in resp.raw or resp.raw['t'] != "MESSAGE_CREATE":
        return

    data = resp.raw['d']
    channel_id = str(data.get('channel_id'))
    author_id = str(data.get('author', {}).get('id'))
    content = data.get('content', '')

    logging.debug(f"📨 Message from {author_id} in channel {channel_id} - Content: {content}")

    if channel_id != str(CHANNEL_ID):
        return

    if author_id == "853629533855809596":
        attachments = data.get('attachments', [])
        components = data.get('components', [])
        LAST_DROP_MESSAGE_ID = data.get("id")  # 🔁 This now sets the GLOBAL var correctly

        for attachment in attachments:
            filename = attachment.get('filename', '')
            if any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                try:
                    logging.info(f"📥 Found image attachment: {filename}")
                    response = requests.get(attachment.get('url'))
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    click_bouquet_then_best_from_image(image, components)
                except Exception as e:
                    logging.warning(f"⚠️ Failed to process image: {e}")
                return
            
bot.gateway.run(auto_reconnect=True)