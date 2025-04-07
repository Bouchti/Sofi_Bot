import logging
import time
import mss
import pyautogui
import pytesseract
import cv2
import numpy as np
import pygetwindow as gw
from PIL import Image
import easyocr
import re
import random
from io import BytesIO
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

# Configuration
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
with mss.mss() as sct:
    monitor = sct.monitors[1]  # Select second monitor
BUTTON_OFFSETS = [0, 475, 775]
CARD_DROP_X = monitor["left"] + 420
CARD_DROP_Y = 800
CARD_DROP_WIDTH = 600
CARD_DROP_HEIGHT = 165
CLAIM_BUTTON_TEMPLATE_PATH = "claim_button.png"
CLAIM_BUTTON_TEMPLATE = cv2.imread(CLAIM_BUTTON_TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
# Load templates for bouquet detection
bouquet_card_template = cv2.imread("bouquet_icon_template.png", cv2.IMREAD_UNCHANGED)
bouquet_button_template = cv2.imread("bouquet_button.png", cv2.IMREAD_UNCHANGED)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Logging configuration
# Logging configuration
logging.basicConfig(level=logging.DEBUG)  # Set to WARNING to reduce log output
def capture_cards_only():
    with mss.mss() as sct:
        region = {
            "left": CARD_DROP_X,
            "top": CARD_DROP_Y-200,
            "width": CARD_DROP_WIDTH,
            "height": CARD_DROP_HEIGHT+130
        }
        screenshot = sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save(f"cards_only.png")
        return img

# Ensure the bouquet card template is loaded correctly
if bouquet_card_template is None:
    logging.error("❌ Bouquet card template not found or failed to load.")
else:
    # Convert the template to grayscale
    bouquet_card_template = cv2.cvtColor(bouquet_card_template, cv2.COLOR_BGR2GRAY)

# Function to detect bouquet card and return its index (0, 1, 2)
def detect_bouquet_card(screenshot):
    screenshot_np = np.array(screenshot)
    card_width = CARD_DROP_WIDTH // 3
    card_height = CARD_DROP_HEIGHT + 130
    matches = []

    # Convert the screenshot to grayscale
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

    for i in range(3):
        left = i * card_width 
        right = left + card_width 
        card_img = screenshot_gray[0:card_height, left:right]

        # Perform template matching
        res = cv2.matchTemplate(card_img, bouquet_card_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        logging.debug(f"🔍 Bouquet match value for card {i+1}: {max_val}")
        if max_val > 0.9:
            matches.append((i, max_val))

    if matches:
        best_match = max(matches, key=lambda x: x[1])
        logging.info(f"💐 Bouquet card detected at position {best_match[0] + 1} with confidence {best_match[1]:.2f}")
        return best_match[0]
    return None

# Function to click bouquet button
def click_bouquet_button(card_index, buttons):
    button_x, button_y = buttons[card_index]
    button_center_x = button_x + 40
    button_center_y = button_y + 15 
    logging.info(f"🌹 Clicking bouquet card button at position: {buttons[card_index]}")
    pyautogui.moveTo(button_center_x, button_center_y, duration=0.2)
    time.sleep(1)
    pyautogui.click()
    logging.info("✅ Bouquet claimed successfully.")


def click_bouquet_then_best():
    logging.info("\n🔍 Starting smart claim process...")
    screenshot = capture_cards_only()
    buttons = find_buttons()
    bouquet_card_index = detect_bouquet_card(screenshot)

    if bouquet_card_index is not None:
        click_bouquet_button(bouquet_card_index, buttons)
        time.sleep(2)  # Small delay to ensure bouquet is claimed

    remaining_indices = [i for i in range(3) if i != bouquet_card_index]

    # Extract generations and remove bouquet card from gen dictionary
    card_gens = extract_card_generations(screenshot)
    card_gens = {k: v for k, v in card_gens.items() if k in remaining_indices}

    best_match_index = None
    best_rank = float('inf')
    no_gen_card_index = None
    card_width = CARD_DROP_WIDTH // 3

    for i in remaining_indices:
        left = i * card_width
        right = left + card_width
        card_image = screenshot.crop((left, 0, right, CARD_DROP_HEIGHT+130))
        names = extract_card_names(np.array(card_image))
        matches = find_best_character_match(names)

        for _, match_name, score, rank in matches:
            if score >= 95 and rank < best_rank:
                best_rank = rank
                best_match_index = i

        if i not in card_gens and no_gen_card_index is None:
            no_gen_card_index = i

    # Decide which card to click
    if no_gen_card_index is not None:
        button_index = no_gen_card_index
    elif best_match_index is not None:
        button_index = best_match_index
    elif card_gens:
        flattened_gens = [(index, gen) for index, gens in card_gens.items() for gen in gens.items()]
        sorted_card_gens = sorted(flattened_gens, key=lambda x: x[1][1])
        button_index = sorted_card_gens[0][0] if sorted_card_gens else remaining_indices[0]
    else:
        button_index = remaining_indices[0]  # fallback

    # Final click
    button_x, button_y = buttons[button_index]
    button_center_x = button_x + 40
    button_center_y = button_y + 15 
    logging.info(f"🎯 Clicking best card at position {button_index + 1}: ({button_center_x}, {button_center_y})")
    pyautogui.moveTo(button_center_x, button_center_y, duration=0.2)
    time.sleep(1)
    pyautogui.click()
    logging.info(f"✅ Claimed secondary card (index {button_index + 1})")


def preprocess_string(s):
    tokens = re.sub(r'[^a-zA-Z0-9]', ' ', s).upper().split()
    return ' '.join(sorted(tokens))

# Load top characters from file
def preprocess_string(s):
    # Remove punctuation, convert to uppercase, and sort tokens
    tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', s).upper().split()
    return ' '.join(sorted(tokens))

try:
    with open("top_sofi_characters.txt", "r", encoding="utf-8") as f:
        TOP_CHARACTERS_LIST = [preprocess_string(name.strip()) for name in f.readlines()]
        TOP_CHARACTERS_SET = set(TOP_CHARACTERS_LIST)
except FileNotFoundError:
    TOP_CHARACTERS_LIST = []
    TOP_CHARACTERS_SET = set()
    logging.warning("⚠️ 'top_mal_characters.txt' not found. AI card evaluation will be skipped.")


# AI-based card name extraction and matching
reader = easyocr.Reader(['en'], gpu=True)


def scrape_top_characters(limit=3000):
    characters = []
    seen = set()

    for offset in range(0, limit, 50):
        url = f"https://myanimelist.net/character.php?limit={offset}"
        logging.info(f"🔍 Scraping: {url}")
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, "html.parser")

            name_elements = soup.select("a.fw-b")
            for tag in name_elements:
                name = tag.text.strip().upper()
                if name and name not in seen:
                    characters.append(name)
                    seen.add(name)
        except Exception as e:
            logging.warning(f"⚠️ Error scraping {url}: {e}")

        time.sleep(1)  # Respectful delay

    if characters:
        with open("top_mal_characters.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(characters))  # Preserves order
        logging.info(f"✅ Saved {len(characters)} characters to top_mal_characters.txt")
    else:
        logging.warning("⚠️ No characters were scraped.")

    return characters

def extract_card_names(image):
    results = reader.readtext(image, detail=0)
    cleaned = []
    for r in results:
        r_clean = re.sub(r'[^a-zA-Z0-9 ]', '', r).strip()
        if len(r_clean) > 1:
            # Try inserting space between lowercase followed by uppercase
            spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', r_clean)
            cleaned.append(spaced.upper())
    logging.debug(f"🔍 Extracted Raw Card Names: {results}")
    logging.debug(f"🧹 Cleaned Names: {cleaned}")
    return cleaned


def find_best_character_match(card_names):
    matches = []
    if len(card_names) > 1:
        name = card_names[1]
        try:
            normalized_name = preprocess_string(name)
            best_match = process.extractOne(
                normalized_name, 
                TOP_CHARACTERS_LIST, 
                scorer=fuzz.token_sort_ratio
            )
            if best_match:
                match_name, score = best_match[0], best_match[1]
                if score >= 85:                  
                    rank = TOP_CHARACTERS_LIST.index(match_name)
                    logging.debug(f"🔍 Match found: {name} -> {match_name} with score {score} and rank {rank}")
                    matches.append((name, match_name, score, rank))
                else:
                    logging.info(f"⚠️ Low score match: {name} -> {match_name} with score {score}")
            else:
                logging.info(f"❌ No match found for: {name}")
        except Exception as e:
            logging.error(f"Error matching {name}: {e}")
    return matches

def capture_discord_message():
    with mss.mss() as sct:
        region = {
            "left": CARD_DROP_X,
            "top": CARD_DROP_Y,
            "width": CARD_DROP_WIDTH,
            "height": CARD_DROP_HEIGHT
        }
        screenshot = sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save(f"cards.png")
        return img

def capture_buttons():
    with mss.mss() as sct:
        region = {
            "left": CARD_DROP_X,
            "top": CARD_DROP_Y+100,
            "width": CARD_DROP_WIDTH-100,
            "height": CARD_DROP_HEIGHT-30
        }
        screenshot = sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save(f"buttons.png")
        return img

def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(image)
    blurred = cv2.GaussianBlur(contrast_img, (3,3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    return sharpened

def clean_generation_text(ocr_results):
    cleaned_generations = {}
    logging.debug(f"🔍 Raw OCR Results: {ocr_results}")

    for text in ocr_results:
        if len(cleaned_generations) >= 3:
            break

        original_text = text
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        text = text.replace("i", "1").replace("I", "1").replace("o", "0").replace("O", "0").replace("g", "9").replace("s", "5").replace("S", "5").replace("B", "8").replace("l", "1")

        if text.startswith("0") or text.startswith("6" ) or text.startswith("5") or text.startswith("9"):
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
    start_time = time.time()
    if not hasattr(extract_generation_with_easyocr, "reader"):
        extract_generation_with_easyocr.reader = easyocr.Reader(['en'], gpu=True)

    max_width = 800
    if image.shape[1] > max_width:
        scale_ratio = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, int(image.shape[0] * scale_ratio)), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    extracted_text = extract_generation_with_easyocr.reader.readtext(image, detail=0)
    logging.debug(f"🔍 EasyOCR Raw Extracted: {extracted_text}")
    generations = clean_generation_text(extracted_text)
    time_taken = time.time() - start_time
    logging.info(f"⏱️ Time taken for extract_generation_with_easyocr: {time_taken:.2f} seconds")
    return generations

def extract_card_generations(image):
    """Extracts generation numbers for each card by cropping the image into three sections."""
    card_width = CARD_DROP_WIDTH // 3
    card_generations = {}

    for i in range(3):
        left = i * card_width
        right = left + card_width
        card_image = image.crop((left, 0, right, CARD_DROP_HEIGHT+130))
        card_image.save(f"card_{i}.png")

        # Preprocess the image in memory
        processed_image = preprocess_image(card_image)
        extracted_numbers = extract_generation_with_easyocr(processed_image)

        if extracted_numbers:
            card_generations[i] = extracted_numbers
        else:
            logging.debug(f"❌ No valid generation numbers found for card {i+1}.")

    return card_generations

def find_buttons():
    screenshot = capture_discord_message()
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    
    # Ensure the template is in grayscale
    if CLAIM_BUTTON_TEMPLATE is None:
        logging.error("❌ Claim button template not found or failed to load.")
        return [(CARD_DROP_X + offset, CARD_DROP_Y + 270) for offset in BUTTON_OFFSETS]
    
    template_gray = cv2.cvtColor(CLAIM_BUTTON_TEMPLATE, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(screenshot_cv, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(result >= threshold)
    detected_buttons = [(pt[0] + CARD_DROP_X, pt[1] + CARD_DROP_Y) for pt in zip(*loc[::-1])]
    
    # Filter out buttons that are too close to each other
    filtered_buttons = []
    min_distance = 50
    for button in detected_buttons:
        if all(np.linalg.norm(np.array(button) - np.array(existing_button)) > min_distance for existing_button in filtered_buttons):
            filtered_buttons.append(button)
    
    # Sort and select the top 3 buttons
    buttons = sorted(filtered_buttons, key=lambda x: x[0])[:3]
    if len(buttons) != 3:
        logging.warning(f"❌ Incorrect number of buttons detected ({len(buttons)} found). Adjust template or threshold.")
        buttons = [(CARD_DROP_X + offset, CARD_DROP_Y + 270) for offset in BUTTON_OFFSETS]
    else:
        logging.debug(f"✅ Detected {len(buttons)} claim buttons at positions: {buttons}")
    
    return buttons

def click_lowest_or_first_card():
    logging.info("\n🔍 Starting claim process...")
    screenshot = capture_discord_message()
    card_gens = extract_card_generations(screenshot)
    buttons = find_buttons()

    if len(buttons) != 3:
        logging.warning(f"❌ Incorrect number of buttons detected ({len(buttons)} found). Adjust template or threshold.")
        buttons = [(CARD_DROP_X + offset, CARD_DROP_Y + 270) for offset in BUTTON_OFFSETS]

    best_match_index = None
    best_rank = float('inf')
    no_gen_card_index = None

    card_width = CARD_DROP_WIDTH // 3
    for i in range(3):
        left = i * card_width
        right = left + card_width
        card_image = screenshot.crop((left, 0, right, CARD_DROP_HEIGHT))
        names = extract_card_names(np.array(card_image))
        matches = find_best_character_match(names)

        for _, match_name, score, rank in matches:
            if score >= 95 and rank < best_rank:
                best_rank = rank
                best_match_index = i

        # Check if the card has no generation numbers
        if i not in card_gens and no_gen_card_index is None:
            no_gen_card_index = i

    if no_gen_card_index is not None:
        logging.info(f"✅ No generation found on card {no_gen_card_index + 1}. Prioritizing this card.")
        button_index = no_gen_card_index
    elif best_match_index is not None:
        logging.info(f"🌟 Best match found on card {best_match_index + 1} with SOFI rank {best_rank + 1}")
        button_index = best_match_index
    elif card_gens:
        flattened_gens = [(index, gen) for index, gens in card_gens.items() for gen in gens.items()]
        sorted_card_gens = sorted(flattened_gens, key=lambda x: x[1][1])
        lowest_gen_card, lowest_gen_value = sorted_card_gens[0][1]
        logging.info(f"✅ Lowest Gen Card: {lowest_gen_card} (Gen {lowest_gen_value})")
        button_index = sorted_card_gens[0][0]
    else:
        logging.info("⚠️ No generation or name match found. Defaulting to first button.")
        button_index = 0

    button_x, button_y = buttons[button_index]
    button_center_x = button_x + 40
    button_center_y = button_y + 15 
    logging.info(f"🎯 Final Selection: Clicking button at position {buttons[button_index]}")
    logging.info(f"📌 Moving to: ({button_center_x}, {button_center_y}) for Button {button_index + 1}")
    pyautogui.moveTo(button_center_x, button_center_y, duration=0.2)
    time.sleep(1)
    pyautogui.click()
    logging.info(f"✅ Clicked the claim button for Button {button_index + 1}!\n")


def focus_discord():
    """Bring Discord to the front and focus on the 'Send a message' text area."""
    try:
        for window in gw.getWindowsWithTitle("Discord"):
            try:
                window.activate()
                logging.info("✅ Discord activated!")
            except Exception as e:
                logging.error(f"Error activating Discord window: {e}")
                # Alternative method: click on the taskbar icon
                logging.info("Attempting to click on the Discord taskbar icon.")
                pyautogui.click(x=100, y=1050)  # Adjust coordinates to your taskbar location
                time.sleep(1)

            text_area_x = monitor["left"] + 500
            text_area_y = monitor["top"] + 1000

            pyautogui.click(text_area_x, text_area_y)
            logging.info("✅ Focused on 'Send a message' text area.")
            time.sleep(1)
            return True
    except Exception as e:
        logging.error(f"Error focusing Discord: {e}")
    return False

while True:
    click_bouquet_then_best()
    break
    focus_discord()
    pyautogui.write("sd")
    pyautogui.press("enter")
    time.sleep(4)
    click_bouquet_then_best()
    sleep_duration = random.randint(480, 500)
    logging.info(f"Sleeping for {sleep_duration} seconds before the next action.")
    time.sleep(sleep_duration)