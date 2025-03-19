
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
# Configuration
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Detect and store the second monitor's position
with mss.mss() as sct:
    monitor = sct.monitors[2]  # Select second monitor
BUTTON_OFFSETS = [0, 475, 775]
CARD_DROP_X = monitor["left"] + 360  # Shift inside Discord window
CARD_DROP_Y = 600  
CARD_DROP_WIDTH = 600
CARD_DROP_HEIGHT = 400
CLAIM_BUTTON_TEMPLATE_PATH = "claim_button.png"
CLAIM_BUTTON_TEMPLATE = cv2.imread(CLAIM_BUTTON_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Logging configuration
logging.basicConfig(level=logging.INFO)

def capture_discord_message():
    """Capture a screenshot of the card drop area."""
    with mss.mss() as sct:
        region = {
            "left": CARD_DROP_X,
            "top": CARD_DROP_Y,
            "width": CARD_DROP_WIDTH,
            "height": CARD_DROP_HEIGHT
        }
        screenshot = sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save("debug_screenshot.png")  # Save for debugging
        return img
def preprocess_image(image_path):
    """Enhance the image for better OCR detection."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(image)
    blurred = cv2.GaussianBlur(contrast_img, (3,3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    processed_path = "debug_preprocessed.png"
    cv2.imwrite(processed_path, sharpened)
    return processed_path

def clean_generation_text(ocr_results):
    """Cleans and normalizes OCR results to extract exactly three valid generation numbers."""
    cleaned_generations = {}
    logging.info(f"🔍 Raw OCR Results: {ocr_results}")

    for text in ocr_results:
        if len(cleaned_generations) >= 3:
            break  # Stop processing once we have three valid generations

        original_text = text
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        text = text.replace("i", "1").replace("I", "1").replace("o", "0").replace("O", "0")

        if text.startswith("0") or text.startswith("6"):
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
            logging.info(f"✅ Final Processed: {original_text} → {gen_number}")
   
    logging.info(f"🚀 FINAL NORMALIZED GENERATIONS: {cleaned_generations}")
    return cleaned_generations

def extract_generation_with_easyocr(image_path):
    """Extracts generation numbers using EasyOCR with improved accuracy."""
    # Initialize EasyOCR reader with GPU support
    reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=True to enable GPU usage
    extracted_text = reader.readtext(image_path, detail=0)
    logging.info(f"🔍 EasyOCR Raw Extracted: {extracted_text}")
    generations = clean_generation_text(extracted_text)
    return generations

def extract_card_generations(image):
    """Extracts generation numbers for each card by cropping the image into three sections."""
    card_width = CARD_DROP_WIDTH // 3
    card_generations = {}

    for i in range(3):
        left = i * card_width
        right = left + card_width
        card_image = image.crop((left, 0, right, CARD_DROP_HEIGHT))
        card_image.save(f"debug_card_{i}.png")  # Save for debugging

        processed_image_path = preprocess_image(f"debug_card_{i}.png")
        logging.info(f"🔍 Sending preprocessed card {i} image to OCR for text extraction...")
        extracted_numbers = extract_generation_with_easyocr(processed_image_path)

        if extracted_numbers:
            card_generations[i] = extracted_numbers
        else:
            logging.info(f"❌ No valid generation numbers found for card {i}.")

    return card_generations

def find_buttons():
    """Detect and return claim button coordinates using image recognition."""
    screenshot = capture_discord_message()
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

    # Match template for claim button detection
    result = cv2.matchTemplate(screenshot_cv, CLAIM_BUTTON_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  # Adjusted threshold for better accuracy
    loc = np.where(result >= threshold)

    detected_buttons = [(pt[0] + CARD_DROP_X, pt[1] + CARD_DROP_Y) for pt in zip(*loc[::-1])]

    # Filter out detections that are too close to each other
    filtered_buttons = []
    min_distance = 50  # Minimum distance between buttons to consider them distinct
    for button in detected_buttons:
        if all(np.linalg.norm(np.array(button) - np.array(existing_button)) > min_distance for existing_button in filtered_buttons):
            filtered_buttons.append(button)

    buttons = sorted(filtered_buttons, key=lambda x: x[0])[:3]

    if len(buttons) != 3:
        logging.info(f"❌ Incorrect number of buttons detected ({len(buttons)} found). Adjust template or threshold.")
        buttons = [(CARD_DROP_X + offset, CARD_DROP_Y + 270) for offset in BUTTON_OFFSETS]
    else:
        logging.info(f"✅ Detected {len(buttons)} claim buttons at positions: {buttons}")

    return buttons

def click_lowest_or_first_card():
    """Find and click the lowest generation card, or first if no gen is found."""
    logging.info("\n🔍 Starting claim process...")
    screenshot = capture_discord_message()
    card_gens = extract_card_generations(screenshot)
    buttons = find_buttons()

    if len(buttons) != 3:
        logging.info(f"❌ Incorrect number of buttons detected ({len(buttons)} found). Adjust template or threshold.")
        buttons = [(CARD_DROP_X + offset, CARD_DROP_Y + 270) for offset in BUTTON_OFFSETS]

    # Identify the button index without a generation number
    cards_without_gen = [index for index in range(3) if index not in card_gens]
    if cards_without_gen:
        logging.info("⚠️ Cards without generation numbers detected. Prioritizing these cards.")
        button_index = cards_without_gen[0]
        lowest_gen_card = None
    else:
        # Flatten the dictionary to get a list of tuples (index, gen_number)
        flattened_gens = [(index, gen) for index, gens in card_gens.items() for gen in gens.items()]
        sorted_card_gens = sorted(flattened_gens, key=lambda x: x[1][1])
        logging.info(f"📌 Sorted Generations (Lowest First): {sorted_card_gens}")
        lowest_gen_card, lowest_gen_value = sorted_card_gens[0][1]
        logging.info(f"✅ Lowest Gen Card: {lowest_gen_card} (Gen {lowest_gen_value})")
        button_index = sorted_card_gens[0][0]

    button_x, button_y = buttons[button_index]
    button_center_x = button_x + 40
    button_center_y = button_y + 15
    logging.info(f"🎯 Final Selection: Clicking button at position {buttons[button_index]}")
    logging.info(f"📌 Moving to: ({button_center_x}, {button_center_y}) for Button {button_index + 1}")
    pyautogui.moveTo(button_center_x, button_center_y, duration=0.2)
    time.sleep(1)
    pyautogui.click()
    logging.info(f"✅ Clicked the claim button for card {lowest_gen_card if lowest_gen_card else 'Unknown'} at Button {button_index + 1}!\n")


def focus_discord():
    """Bring Discord to the front and focus on the 'Send a message' text area."""
    for window in gw.getWindowsWithTitle("Discord"):
        window.activate()
        logging.info("✅ Discord activated!")
        time.sleep(1)

        # Coordinates for the "Send a message" text area
        # These coordinates may need to be adjusted based on your screen resolution and Discord layout
        text_area_x = monitor["left"] + 500  # Example X coordinate
        text_area_y = monitor["top"] + 1000  # Example Y coordinate

        # Click on the "Send a message" text area
        pyautogui.click(text_area_x, text_area_y)
        logging.info("✅ Focused on 'Send a message' text area.")
        time.sleep(1)
        return True
    return False

# Auto-roll every 8 minutes
while True:
    click_lowest_or_first_card()
