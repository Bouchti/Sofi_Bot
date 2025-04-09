🌾 Sofi_Bot – Automated Card Farming Bot for Sofi Discord Bot
🤖 Automate your Sofi experience and efficiently farm cards using powerful OCR, image processing, and intelligent decision-making!

📌 Overview
Sofi_Bot is a smart automation bot designed to interact with the Sofi Discord Bot — a card-collecting game on Discord. This tool simulates user behavior by detecting dropped cards, analyzing image content, and automatically claiming the best card using OCR and custom logic.

⚙️ Features
🔁 Automated Farming – Periodically sends sd command every 8 minutes to drop cards

🧠 Smart Claiming Logic

Detects and instantly claims bouquet-style cards

Extracts generation numbers and character names from card images

Prioritizes top-tier characters using a customizable list

Claims cards with no generation first, then the lowest gen, or best match

🧾 OCR Text Recognition – Uses EasyOCR to extract card details with high accuracy

🧩 Template Matching – Detects bouquet buttons using OpenCV

💥 Fast & Responsive – Uses multi-threading for parallel image processing

🔄 Self-Healing Gateway – Auto-reconnects on Discord gateway freezes

🔧 Fully Configurable – Easy to modify filters, logic, and timing from .env and source

🧠 How It Works
Listens for SOFI drops across one or multiple Discord servers/channels using discum

Parses attached images using EasyOCR and OpenCV

Extracts generation + character name

Compares names using fuzzy logic (RapidFuzz)

Sends button interactions directly to the Discord API to claim

Resilient to freezes or disconnections with auto-restart logic

🖥️ Technologies Used
🐍 Python 3.9+

⚙️ discum – Discord API wrapper

🔍 EasyOCR – Optical character recognition

🧠 RapidFuzz – For fuzzy matching card names

🧪 OpenCV – Template matching & image denoising

🖼️ Pillow – Image processing

🌐 requests – Interact with Discord's API

🔐 dotenv – Manage tokens/configs securely

🚀 Getting Started
1️⃣ Clone the Repo
bash
Copier
Modifier
git clone https://github.com/Bouchti/Sofi_Bot.git
cd Sofi_Bot
2️⃣ Install Dependencies
bash
Copier
Modifier
pip install -r requirements.txt
3️⃣ Configure .env
Create a .env file with:

env
Copier
Modifier
DISCORD_TOKEN=your_bot_token
GUILD_ID=optional_default_guild_id
CHANNEL_ID=optional_default_channel_id
USER_ID=your_user_id
Your bot must be in the server and have access to the drop channel.

4️⃣ Run the Bot
bash
Copier
Modifier
python farmV3.py
⚠️ Disclaimer
This bot interacts with Discord’s private API. Use responsibly and ethically. You must not use this bot to spam, exploit, or violate Discord’s Terms of Service.

📸 Screenshots
🚧 Coming soon...

🧑‍💻 Author
Developed with ❤️ by Bouchti
