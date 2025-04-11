# 🌾 Sofi_Bot – Automated Card Farming Bot for Sofi Discord Bot

🤖 Automate your **Sofi** experience and efficiently farm cards using powerful OCR, image processing, and smart decision-making!

---

## 📌 Overview

**Sofi_Bot** is a smart automation bot designed to interact with the [Sofi Discord Bot](https://discord.com/invite/sofi), a popular card-collecting game.  
This tool detects dropped cards, analyzes image content, and automatically claims the best card using intelligent logic.

---

## ⚙️ Features

- 🔁 **Automated Farming** – Sends the `sd` command every 8 minutes to drop cards  
- 💐 **Bouquet Detection** – Instantly claims bouquet-style cards using template matching  
- 🧠 **Smart Claiming Logic**:
  - Reads card **generation numbers** and **character names**
  - Claims cards with **no generation first**
  - Selects card with **lowest generation** or **best-ranked match**
- 🧾 **OCR Recognition** – Uses EasyOCR for accurate text extraction  
- 🧩 **Template Matching** – Uses OpenCV to detect buttons/icons  
- 💥 **Fast & Responsive** – Multi-threaded image processing for low latency  
- 🔄 **Self-Healing** – Automatically restarts when Discord gateway is frozen  
- ⚙️ **Fully Configurable** – Edit filters, logic, timers, and templates easily

---

## 🧠 How It Works

1. Listens to Sofi drops using **discum**
2. Processes attached card images with **EasyOCR** + **OpenCV**
3. Extracts generation & name from image
4. Matches name using fuzzy logic (**RapidFuzz**)
5. Sends claim interactions via the Discord API
6. Supports multi-server + auto-reconnect

---

## 🖥️ Technologies Used

- 🐍 Python 3.9+
- ⚙️ `discum` – Discord API wrapper
- 🔍 `EasyOCR` – OCR text extraction
- 🧠 `RapidFuzz` – Fuzzy name matching
- 🧪 `OpenCV` – Template matching
- 🖼️ `Pillow` – Image processing
- 🌐 `requests` – API communication
- 🔐 `dotenv` – Manage tokens and user data

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Bouchti/Sofi_Bot.git
cd Sofi_Bot

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Create .env Configuration
DISCORD_TOKEN=your_discord_bot_token
GUILD_ID=your_main_server_id        # Optional
CHANNEL_ID=your_main_channel_id     # Only used for sending 'sd' commands
USER_ID=your_user_id                # Optional (for logs)

✅ Make sure .env is listed in .gitignore to avoid pushing it to GitHub.

### 4️⃣ Run the Bot

python farmV3.py

⚠️ Disclaimer
This bot interacts with Discord’s internal API. Use responsibly.
You must not use this bot to spam, exploit, or violate Discord’s Terms of Service.

Use at your own risk.

📸 Screenshots
🚧 Coming soon...

🧑‍💻 Author
Developed with ❤️ by Bouchti
