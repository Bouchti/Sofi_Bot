# 🌾 Sofi_Bot – Automated Card Farming Bot for Sofi Discord Bot

🤖 Automate your **Sofi** experience and efficiently farm cards with powerful OCR and smart decision-making!

---

## 📌 Overview

**Sofi_Bot** is a powerful automation bot designed to interact with the [Sofi Discord Bot](https://discord.com/invite/sofi) — a popular card-collecting game on Discord. This tool simulates intelligent user behavior by processing images from drops, reading text, detecting special cards (like bouquets), and auto-claiming the best card.

---

## ⚙️ Features

- 🔁 **Automated Farming** – Periodically sends the `sd` command to drop cards every 8 minutes
- 🧠 **Smart Claiming Logic**:
  - Detects **bouquet-style** cards and claims them automatically
  - Analyzes **generation numbers** and **character names**
  - Uses a predefined list to prioritize **top characters**
  - Claims cards with no generation first, then lowest gen or best match
- 🧾 **OCR Text Recognition** – Uses `EasyOCR` to extract generation numbers and names
- 🧩 **Template Matching** – Matches icons like bouquet buttons with OpenCV
- ⚙️ **Fully Configurable** – Adjust filters, timers, templates, and recognition logic

---

## 🧠 How It Works

- Listens for new messages from Sofi using `discum`
- Parses attached images using EasyOCR
- Extracts character name + generation
- Uses logic to select the best card to claim
- Sends button click requests via Discord's API
- Supports auto-reconnect and self-healing

---

## 🖥️ Technologies Used

- `Python 3.9+`
- [`discum`](https://github.com/Merubokkusu/Discum) – lightweight Discord API wrapper
- `EasyOCR` – for optical character recognition
- `OpenCV` – for image preprocessing and template matching
- `requests` – for HTTP interactions with Discord API
- `dotenv` – for managing API keys and bot tokens
- `rapidfuzz` – for fuzzy character name matching
- `Pillow (PIL)` – for image manipulation

---

## 🚀 Getting Started

1. 📅 **Clone the Repository**

   ```bash
   git clone https://github.com/Bouchti/Sofi_Bot.git
   cd Sofi_Bot
   ```

2. 📦 **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. 🛠️ **Run the Bot**

   ```bash
   python farmV3.py
   ```
⚠️ Disclaimer
This bot interacts with Discord's API and should be used responsibly and ethically. Misuse may violate Discord's Terms of Service. Use at your own risk.

---

## 📸 Screenshots

*Coming soon...*

---

## 🧑‍💻 Author

Developed by [Bouchti](https://github.com/Bouchti)

