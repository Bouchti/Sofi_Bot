import pyautogui
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import time

# Configuration: coordinates and region
# Define the region of the Discord window or embed to capture (x, y, width, height).
# These values should be set to capture the area containing the cards and claim buttons.
region_x = 100   # example top-left x of Discord content region
region_y = 100   # example top-left y of Discord content region
region_w = 800   # example width of the region (to be updated per screenshot)
region_h = 450   # example height of the region (to be updated per screenshot)

# Base reference dimensions from the latest screenshot (for dynamic scaling).
base_width = 800   # width of the region in the reference screenshot
base_height = 450  # height of the region in the reference screenshot

# Base relative positions of the "Claim" buttons (center of each button) in the reference screenshot.
# These coordinates are relative to the top-left of the captured region.
# Update these based on measured positions in the screenshot.
base_positions_rel = [
    (150, 420),  # Claim button 1 center (x, y) in reference
    (400, 420),  # Claim button 2 center (x, y) in reference
    (650, 420)   # Claim button 3 center (x, y) in reference
]

# Take a fresh screenshot of the Discord region containing the cards and buttons.
screenshot = pyautogui.screenshot(region=(region_x, region_y, region_w, region_h))

# (Optional) Save screenshot for debugging
# screenshot.save("debug_screenshot_new.png")

# Determine current dimensions (in case Discord window/region size changed)
cur_width, cur_height = screenshot.size

# Calculate scaling ratios compared to base reference
scale_x = cur_width / base_width
scale_y = cur_height / base_height

# Compute current positions of claim buttons by scaling base positions
positions_rel = [
    (int(x * scale_x), int(y * scale_y))
    for (x, y) in base_positions_rel
]

# Convert relative positions to absolute screen coordinates (add region origin offset)
positions_abs = [
    (region_x + rel_x, region_y + rel_y)
    for (rel_x, rel_y) in positions_rel
]

# Use OCR to read generation numbers on each card
gen_texts = []
# Define sub-region for generation number on each card (relative to the region)
# (These coordinates should be updated to accurately crop the gen number from each card.)
gen_regions = [
    (50, 300, 80, 30),   # (x, y, width, height) for card 1
    (300, 300, 80, 30),  # (x, y, width, height) for card 2
    (550, 300, 80, 30)   # (x, y, width, height) for card 3
]

for (gx, gy, gw, gh) in gen_regions:
    try:
        # Convert width, height into right, bottom coordinates
        gen_img = screenshot.crop((gx, gy, gx + gw, gy + gh))  # ✅ Fix applied
        
        # Preprocess for better OCR
        gen_img = gen_img.convert("L")  # Convert to grayscale
        gen_img = gen_img.resize((gw * 2, gh * 2), Image.ANTIALIAS)  # Enlarge image
        gen_img = gen_img.filter(ImageFilter.SHARPEN)  # Sharpen text

        # Use OCR to extract text
        text = pytesseract.image_to_string(gen_img, config="--psm 7").strip()
        text = ''.join(ch for ch in text if ch.isalnum())  # Remove unwanted chars

        gen_texts.append(text)

    except Exception as e:
        print(f"⚠️ Error extracting text for region ({gx}, {gy}, {gw}, {gh}): {e}")
        gen_texts.append("")

# Determine which card to claim:
target_index = None
for i, text in enumerate(gen_texts):
    if text == "" or text is None:
        # This card has no visible generation number – prioritize it
        target_index = i
        break

if target_index is None:
    # If all cards have a generation number visible, fall back to original selection logic.
    # (For example, choose the card with the lowest generation number or a default.)
    # Here we default to the first card (index 0) as an example.
    target_index = 0
    # Example: If you wanted the lowest generation number instead:
    # gen_numbers = [int(t) if t.isdigit() else float('inf') for t in gen_texts]
    # target_index = gen_numbers.index(min(gen_numbers))

# Simulate cursor movement to each claim button position for verification
for (x, y) in positions_abs:
    pyautogui.moveTo(x, y, duration=0.3)
    time.sleep(0.2)

# Move to the target button position (in case it's not the last one visited above)
target_x, target_y = positions_abs[target_index]
pyautogui.moveTo(target_x, target_y, duration=0.3)

# Brief pause, then click the target button
time.sleep(0.2)
pyautogui.click(target_x, target_y)

print(f"Claimed card {target_index+1} (Gen text='{gen_texts[target_index] or 'N/A'}').")