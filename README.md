# 🌾 Sofi\_Bot – Automated Card Farming Bot for Sofi Discord Bot

\
🤖 Automate your **Sofi** experience and efficiently farm cards with ease!

---

## 📌 Overview

**Sofi\_Bot** is a lightweight and customizable automation bot designed to interact with the [Sofi Discord Bot](https://discord.com/invite/sofi) — a popular card-collecting bot on Discord. This tool simulates user interactions to help you farm cards, claim characters, and manage farming routines without manual input.

---

## ⚙️ Features

- 🔁 **Automated Farming** – Sends sd command at set intervals
- 👡 **Mouse Simulation** – Simulates real mouse clicks using `pyautogui`
- 🧠 **Smart Claiming** – Detects and claims cards quickly based on visual cues
- 🖼️ **Image Matching** – Uses template matching for detecting buttons and cards
- 🔧 **Customizable** – Modify timings, detection logic, and image templates to suit your needs

---

## 🖥️ Things Used

- `Python 3.9+`
- `OpenCV` – for image recognition and template matching
- `pyautogui` – for simulating mouse clicks
- `time`, `random`, and `os` – for task automation and randomness

---

## 📂 Repository Structure

```
├── assets/                 # Image templates used for detection
│   ├── bouquet_button.png
│   ├── cards.png
├── farm.py                # Main automation script
├── debug.py               # For testing image recognition
├── mouse.py               # Handles mouse movements and clicking
└── README.md              # Documentation and usage
```

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
   python farmV2.py
   ```

> ⚠️ This bot simulates user behavior. Use responsibly and at your own risk, in accordance with Discord's Terms of Service.

---

## 📸 Screenshots

*Coming soon...*

---

## 🧑‍💻 Author

Developed by [Bouchti](https://github.com/Bouchti)

