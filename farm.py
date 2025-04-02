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

# Configuration
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
with mss.mss() as sct:
    monitor = sct.monitors[2]  # Select second monitor
BUTTON_OFFSETS = [0, 475, 775]
CARD_DROP_X = monitor["left"] + 360
CARD_DROP_Y = 600
CARD_DROP_WIDTH = 600
CARD_DROP_HEIGHT = 400
CLAIM_BUTTON_TEMPLATE_PATH = "claim_button.png"
CLAIM_BUTTON_TEMPLATE = cv2.imread(CLAIM_BUTTON_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Logging configuration
logging.basicConfig(level=logging.DEBUG)  # Set to WARNING to reduce log output

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
        return img

def preprocess_image(image):
    """Enhance the image for better OCR detection."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(image)
    blurred = cv2.GaussianBlur(contrast_img, (3,3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    return sharpened

def clean_generation_text(ocr_results):
    """Cleans and normalizes OCR results to extract exactly three valid generation numbers."""
    cleaned_generations = {}
    logging.debug(f"🔍 Raw OCR Results: {ocr_results}")

    for text in ocr_results:
        if len(cleaned_generations) >= 3:
            break

        original_text = text
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        text = text.replace("i", "1").replace("I", "1").replace("o", "0").replace("O", "0").replace("g", "9").replace("s", "5").replace("S", "5")

        if text.startswith("0") or text.startswith("6" ) or text.startswith("5"):
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
    """Extracts generation numbers using EasyOCR with improved accuracy."""
    # Record the start time for the function
    start_time = time.time()

    # Ensure the reader is initialized once if used multiple times
    if not hasattr(extract_generation_with_easyocr, "reader"):
        extract_generation_with_easyocr.reader = easyocr.Reader(['en'], gpu=True)

    # Resize image if necessary
    max_width = 800
    if image.shape[1] > max_width:
        scale_ratio = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, int(image.shape[0] * scale_ratio)))

    # Check if the image is grayscale and convert it to color if necessary
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Enhance contrast and reduce noise
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Perform OCR
    extracted_text = extract_generation_with_easyocr.reader.readtext(image, detail=0)
    logging.debug(f"🔍 EasyOCR Raw Extracted: {extracted_text}")
    generations = clean_generation_text(extracted_text)

    # Record the end time for the function
    end_time = time.time()

    # Calculate and log the time taken for the function
    time_taken = end_time - start_time
    logging.info(f"⏱️ Time taken for extract_generation_with_easyocr: {time_taken:.2f} seconds")

    return generations

def extract_card_generations(image):
    """Extracts generation numbers for each card by cropping the image into three sections."""
    card_width = CARD_DROP_WIDTH // 3
    card_generations = {}

    for i in range(3):
        left = i * card_width
        right = left + card_width
        card_image = image.crop((left, 0, right, CARD_DROP_HEIGHT))

        # Preprocess the image in memory
        processed_image = preprocess_image(card_image)
        extracted_numbers = extract_generation_with_easyocr(processed_image)

        if extracted_numbers:
            card_generations[i] = extracted_numbers
        else:
            logging.debug(f"❌ No valid generation numbers found for card {i}.")

    return card_generations

def find_buttons():
    """Detect and return claim button coordinates using image recognition."""
    screenshot = capture_discord_message()
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

    result = cv2.matchTemplate(screenshot_cv, CLAIM_BUTTON_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(result >= threshold)

    detected_buttons = [(pt[0] + CARD_DROP_X, pt[1] + CARD_DROP_Y) for pt in zip(*loc[::-1])]

    filtered_buttons = []
    min_distance = 50
    for button in detected_buttons:
        if all(np.linalg.norm(np.array(button) - np.array(existing_button)) > min_distance for existing_button in filtered_buttons):
            filtered_buttons.append(button)

    buttons = sorted(filtered_buttons, key=lambda x: x[0])[:3]

    if len(buttons) != 3:
        logging.warning(f"❌ Incorrect number of buttons detected ({len(buttons)} found). Adjust template or threshold.")
        buttons = [(CARD_DROP_X + offset, CARD_DROP_Y + 270) for offset in BUTTON_OFFSETS]
    else:
        logging.debug(f"✅ Detected {len(buttons)} claim buttons at positions: {buttons}")

    return buttons

def click_lowest_or_first_card():
    """Find and click the lowest generation card, or first if no gen is found."""
    logging.info("\n🔍 Starting claim process...")
    screenshot = capture_discord_message()
    card_gens = extract_card_generations(screenshot)
    buttons = find_buttons()

    if len(buttons) != 3:
        logging.warning(f"❌ Incorrect number of buttons detected ({len(buttons)} found). Adjust template or threshold.")
        buttons = [(CARD_DROP_X + offset, CARD_DROP_Y + 270) for offset in BUTTON_OFFSETS]

    cards_without_gen = [index for index in range(3) if index not in card_gens]
    if cards_without_gen:
        logging.info("⚠️ Cards without generation numbers detected. Prioritizing these cards.")
        button_index = cards_without_gen[0]
        lowest_gen_card = None
    else:
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

        text_area_x = monitor["left"] + 500
        text_area_y = monitor["top"] + 1000

        pyautogui.click(text_area_x, text_area_y)
        logging.info("✅ Focused on 'Send a message' text area.")
        time.sleep(1)
        return True
    return False

# Auto-roll every 8 minutes
while True:
    focus_discord()
    pyautogui.write("sd")  
    pyautogui.press("enter")  
    time.sleep(3)
    click_lowest_or_first_card()
    sleep_duration = random.randint(480, 500)  # Generate a random number between 480 and 500
    logging.info(f"Sleeping for {sleep_duration} seconds before the next action.")
    time.sleep(sleep_duration)  # Sleep for the random duration   