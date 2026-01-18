# Sofi_Bot - Automated Card Farming Bot for Sofi Discord Bot

Automate your Sofi experience and farm cards using OCR, image processing,
and smart decision-making. Includes a GUI for easier control.

---

## Overview

Sofi_Bot is an automation bot for the Sofi Discord Bot. It detects card drops,
analyzes images, and automatically claims the best cards using optimized logic.
GUI mode provides start/stop buttons, logs, and settings.

---

## Features

- Automated Farming - Sends the `sd` command every 8 minutes to drop cards
- Event Card Detection - Instantly claims event cards
- Smart Claiming Logic
  - Reads card generation numbers and character names
  - Claims cards with no generation first
  - Selects card with lowest generation or best-ranked match
- OCR Recognition - Uses EasyOCR for text extraction
- Template Matching - Uses OpenCV to detect buttons/icons
- Fast and Responsive - Multi-threaded image processing for low latency
- Self-Healing - Automatically restarts when Discord gateway is frozen
- Fully Configurable - Edit filters, logic, timers, and templates
- GUI Version - Start/stop bot, view logs, and manage settings
- Raid automation - Sends `sgr`, clicks Start Raid, selects a move, and clicks Apply
- Raid move rules - Avoids Refresh; prefers "Sofi Special" when Refresh is present; otherwise cycles moves

---

## How It Works

1. Listens to Sofi drops using discum
2. Processes attached card images with EasyOCR + OpenCV
3. Extracts generation and name from image
4. Matches name using fuzzy logic (RapidFuzz)
5. Sends claim interactions via the Discord API
6. Supports multi-server + auto-reconnect

---

## Technologies Used

- Python 3.9+
- discum - Discord API wrapper
- EasyOCR - OCR text extraction
- RapidFuzz - Fuzzy name matching
- OpenCV - Template matching
- Pillow - Image processing
- requests - API communication
- dotenv - Manage tokens and user data
- tkinter - GUI interface

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Bouchti/Sofi_Bot.git
cd Sofi_Bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create .env Configuration

```
DISCORD_TOKEN=your_discord_bot_token
GUILD_ID=your_main_server_id        # Optional
CHANNEL_ID=your_main_channel_id     # Only used for sending 'sd' commands
USER_ID=your_user_id                # Optional (for logs)
```

Make sure `.env` is listed in `.gitignore` to avoid pushing it to GitHub.

### 4. Run the Bot

GUI Mode (User-Friendly)

```bash
python sofi_bot_gui_V2.py
```

---

## Disclaimer

This bot interacts with Discord's internal API. Use responsibly.
You must not use this bot to spam, exploit, or violate Discord's Terms of Service.

Use at your own risk.
