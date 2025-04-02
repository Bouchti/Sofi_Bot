import pyautogui 
import time
import easyocr
import re
import mss
import os
import logging
import pygetwindow as gw
import pyperclip

with mss.mss() as sct:
    monitor = sct.monitors[1]  # Select second monitor

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Constants
TOTAL_PAGES = 120
OUTPUT_FILE = "top_sofi_characters.txt"
message_position = (600, 600)  # Position of SWLB message

# Ensure output file is empty
open(OUTPUT_FILE, "w", encoding="utf-8").close()


# Coordinates (adjust with pyautogui.position())
selection_start = (392, 462)  # Start of leaderboard box
selection_end = (695, 860)    # End of leaderboard box
next_button_position = (570, 910)

def copy_leaderboard_text():
    # Click & drag to select the text
    pyautogui.moveTo(*selection_start)
    pyautogui.mouseDown()
    time.sleep(0.1)
    pyautogui.moveTo(*selection_end, duration=0.5)
    pyautogui.mouseUp()
    time.sleep(0.2)

    # Copy selected text
    pyautogui.hotkey("ctrl", "c")
    time.sleep(0.3)
    return pyperclip.paste()

def extract_character_names(text):
    logging.debug("🔍 Extracting character names from copied text...")
    logging.debug(f"📜 Raw Text:\n{text}")

    # Updated pattern:
    # - Match and skip: "❤️" + digits
    # - Then capture: name between the first and second separators
    pattern = r"(?:❤️?\d+\s+)?[•*·.]\s*(.*?)\s+[•*·.\-]"

    matches = re.findall(pattern, text)

    names = []
    for match in matches:
        name = match.strip()

        if name.startswith("❤️") or re.fullmatch(r"❤️?\d+", name):
            logging.debug(f"⛔️ Skipped heart count or invalid entry: {name}")
            continue

        if len(name) >= 3:
            logging.debug(f"✅ Extracted Character Name: {name}")
            names.append(name)
        else:
            logging.debug(f"⛔️ Skipped short or invalid name: {name}")

    logging.debug(f"📦 Total names extracted: {len(names)}")
    return names

def go_to_next_page():
    next_btn_x = monitor["left"] + 570
    next_btn_y = 910
    logging.debug(f"🖱️ Clicking next button at: ({next_btn_x}, {next_btn_y})")
    pyautogui.moveTo(next_btn_x, next_btn_y, duration=0.2)
    pyautogui.click()

def run_scraper():
    all_names = []
    seen_names = set()

    for page in range(1, TOTAL_PAGES + 1):
        print(f"📄 Processing page {page}/{TOTAL_PAGES}...")
        text = copy_leaderboard_text()
        names = extract_character_names(text)

        for name in names:
            if name not in seen_names:
                print(f"✅ {name}")
                all_names.append(name)
                seen_names.add(name)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for name in all_names:
                f.write(name + "\n")

        if page < TOTAL_PAGES:
            go_to_next_page()
            time.sleep(2)

    print("🎉 Done! All pages processed.")


def focus_discord():
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

if __name__ == "__main__":
    focus_discord()
    run_scraper()