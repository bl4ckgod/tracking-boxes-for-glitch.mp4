# tracking boxes for glitch.mp4

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![UI](https://img.shields.io/badge/UI-CustomTkinter-orange)
![Engine](https://img.shields.io/badge/Engine-OpenCV%20%7C%20MoviePy-green)

A powerful, multi-threaded desktop utility designed to generate automated tracking overlay effects on videos. Perfect for creating data-moshing backgrounds, glitchcore/breakcore visual assets, and tech-hud overlays.

The application automatically detects optimal features in a video, locks bounding boxes onto them, links pairs of trackers with procedural lines, and renders the result in real-time.

---

## ✨ Features

* **Advanced Optical Flow (LK Algorithm):** Keeps track of objects smoothly, dynamically re-detecting new points when old ones disappear.
* **Procedural VFX Overlay:** Renders custom bounding boxes, random word tags (fully customizable via settings), and connecting networks between trackers.
* **Asynchronous Multi-threaded Processing:** Video compilation runs on a separate thread using `MoviePy` and `OpenCV`, keeping the GUI fully responsive.
* **Config Persistence:** All paths, language preferences, and rendering parameters are saved automatically into a `config.json` file.
* **Custom Animated Watermark:** Features a smooth, hardware-accelerated horizontal gradient animation inside the interface.
* **Localization:** Full support for 4 languages: English, Русский, Українська, and Español.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/tracking-boxes-for-glitch.mp4.git](https://github.com/YOUR_USERNAME/tracking-boxes-for-glitch.mp4.git)
   cd tracking-boxes-for-glitch.mp4
