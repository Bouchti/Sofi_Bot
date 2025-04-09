import pyautogui 
import time
import re
import mss
import os
import logging
import pygetwindow as gw
import pyperclip
import json

with mss.mss() as sct:
    monitor = sct.monitors[1]  # Select second monitor

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Constants
TOTAL_PAGES = 150
OUTPUT_FILE = "sofi_leaderboard.json"
message_position = (600, 600)  # Position of SWLB message

# Coordinates (adjust with pyautogui.position())
selection_start = (455, 451)  # Start of leaderboard box
selection_end = (785, 866)    # End of leaderboard box
next_button_position = (631, 920)

# Ensure output file is empty
open(OUTPUT_FILE, "w", encoding="utf-8").close()

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

def extract_structured_entries(text):
    logging.debug("🔍 Extracting structured data from copied text...")
    logging.debug(f"📜 Raw Text:\n{text}")
    
    pattern = re.compile(r"❤️?(\d{2,6})\s+[•*·.\-]\s+(.*?)\s+[•*·.\-]\s+(.*)")
    entries = []

    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            likes, character, series = match.groups()
            entry = {
                "likes": int(likes),
                "character": character.strip(),
                "series": series.strip()
            }
            logging.debug(f"✅ Extracted Entry: {entry}")
            entries.append(entry)
        else:
            logging.debug(f"⛔️ Skipped line (no match): {line.strip()}")

    logging.info(f"📦 Total entries extracted: {len(entries)}")
    return entries

def go_to_next_page():
    next_btn_x = next_button_position[0]
    next_btn_y = next_button_position[1]
    logging.debug(f"🖱️ Clicking next button at: ({next_btn_x}, {next_btn_y})")
    pyautogui.moveTo(next_btn_x, next_btn_y, duration=0.2)
    pyautogui.click()

def run_scraper():
    all_entries = []
    seen_keys = set()

    for page in range(1, TOTAL_PAGES + 1):
        print(f"📄 Processing page {page}/{TOTAL_PAGES}...")
        text = copy_leaderboard_text()
        entries = extract_structured_entries(text)

        for entry in entries:
            key = (entry["character"], entry["series"])
            if key not in seen_keys:
                print(f"✅ {entry['character']} ({entry['series']}) - {entry['likes']} likes")
                all_entries.append(entry)
                seen_keys.add(key)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)

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