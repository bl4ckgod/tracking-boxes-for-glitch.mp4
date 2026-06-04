import cv2
import numpy as np
import moviepy as mpe
import random
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import json
import os
import math

CONFIG_FILE = "config.json"

MAX_TRACKERS = 15
REDETECTION_INTERVAL = 30
WORDS = ["CAN", "YOU", "SEE", "ME", "?"]
BOX_LIFESPAN_MIN = 1.0
BOX_LIFESPAN_MAX = 3.0
BOX_SIZE = 50
LINE_COLOR = (255, 255, 255)
LINE_THICKNESS = 1
CURRENT_LANG = "en"

LOCALIZATION = {
    "en": {
        "title": "Video Tracker", "lbl_input": "Source Video:", "lbl_output": "Save Result To:",
        "btn_browse": "Browse", "btn_settings": "⚙ Settings", "btn_create": "Create",
        "status_running": "Processing started...", "status_done": "Done!", "status_error": "Error!",
        "msg_warn_paths": "Please specify file paths.", "msg_success": "Video successfully saved:\n",
        "msg_error_proc": "Processing failed:\n", "set_title": "Settings", "set_max": "Max Trackers:",
        "set_interval": "Detection Interval (frames):", "set_size": "Box Size (px):",
        "set_words": "Words (comma separated):", "set_lang": "Language:", "set_save": "Save",
        "set_error": "Check the validity of numeric values."
    },
    "ru": {
        "title": "Video Tracker", "lbl_input": "Исходное видео:", "lbl_output": "Куда сохранить результат:",
        "btn_browse": "Обзор", "btn_settings": "⚙ Настройки", "btn_create": "Создать",
        "status_running": "Обработка видео запущена...", "status_done": "Готово!", "status_error": "Ошибка!",
        "msg_warn_paths": "Укажите пути к файлам.", "msg_success": "Видео успешно сохранено:\n",
        "msg_error_proc": "Произошел сбой при обработке:\n", "set_title": "Настройки", "set_max": "Макс. трекеров:",
        "set_interval": "Интервал детекции (кадры):", "set_size": "Размер рамки (px):",
        "set_words": "Слова (через запятую):", "set_lang": "Язык:", "set_save": "Сохранить",
        "set_error": "Проверьте корректность числовых значений."
    },
    "uk": {
        "title": "Video Tracker", "lbl_input": "Вихідне відео:", "lbl_output": "Куди зберегти результат:",
        "btn_browse": "Огляд", "btn_settings": "⚙ Налаштування", "btn_create": "Створити",
        "status_running": "Обробка відео запущена...", "status_done": "Готово!", "status_error": "Помилка!",
        "msg_warn_paths": "Вкажіть шляхи до файлів.", "msg_success": "Відео успешно збережено:\n",
        "msg_error_proc": "Стався збій при обробці:\n", "set_title": "Налаштування", "set_max": "Макс. трекерів:",
        "set_interval": "Інтервал детекції (кадри):", "set_size": "Розмір рамки (px):",
        "set_words": "Слова (через кому):", "set_lang": "Мова:", "set_save": "Зберегти",
        "set_error": "Перевірте коректність числових значень."
    },
    "es": {
        "title": "Video Tracker", "lbl_input": "Video de Origen:", "lbl_output": "Guardar Resultado En:",
        "btn_browse": "Buscar", "btn_settings": "⚙ Ajustes", "btn_create": "Crear",
        "status_running": "Procesamiento iniciado...", "status_done": "¡Hecho!", "status_error": "¡Error!",
        "msg_warn_paths": "Por favor, especifique las rutas de los archivos.", "msg_success": "Video guardado exitosamente:\n",
        "msg_error_proc": "Error en el procesamiento:\n", "set_title": "Ajustes", "set_max": "Máx. Rastreadores:",
        "set_interval": "Intervalo de Detección (fotogramas):", "set_size": "Tamaño del Cuadro (px):",
        "set_words": "Palabras (separadas por comas):", "set_lang": "Idioma:", "set_save": "Guardar",
        "set_error": "Verifique la validez de los valores numéricos."
    }
}

feature_params = dict(maxCorners=MAX_TRACKERS, qualityLevel=0.3, minDistance=8, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

tracked_objects = []
prev_gray = None
frame_count = 0
last_time = -1

class TrackedObject:
    def __init__(self, point, creation_time):
        self.id = random.randint(1000, 9999)
        self.point = point
        self.text = random.choice(WORDS)
        self.creation_time = creation_time
        self.lifespan = random.uniform(BOX_LIFESPAN_MIN, BOX_LIFESPAN_MAX)

    def is_alive(self, current_time):
        return (current_time - self.creation_time) < self.lifespan

def process_frame_with_tracking(frame, t):
    global prev_gray, tracked_objects, frame_count, last_time
    if t < last_time:
        prev_gray = None
        tracked_objects = []
        frame_count = 0
    last_time = t

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_frame = frame.copy()
    tracked_objects = [obj for obj in tracked_objects if obj.is_alive(t)]

    if len(tracked_objects) > 0:
        old_points = np.float32([obj.point for obj in tracked_objects]).reshape(-1, 1, 2)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, old_points, None, **lk_params)
        good_new_points = new_points[status == 1]
        survived_objects = [obj for i, obj in enumerate(tracked_objects) if status[i] == 1]
        for i, obj in enumerate(survived_objects):
            obj.point = tuple(good_new_points[i].ravel())
        tracked_objects = survived_objects

    if len(tracked_objects) < MAX_TRACKERS // 2 or frame_count % REDETECTION_INTERVAL == 0:
        mask = np.ones_like(current_gray)
        for obj in tracked_objects:
            x, y = map(int, obj.point)
            cv2.circle(mask, (x, y), 15, 0, -1)
        new_features = cv2.goodFeaturesToTrack(current_gray, mask=mask, **feature_params)
        if new_features is not None:
            for point in new_features:
                if len(tracked_objects) < MAX_TRACKERS:
                    tracked_objects.append(TrackedObject(tuple(point.ravel()), t))

    if tracked_objects:
        for obj in tracked_objects:
            x, y = map(int, obj.point)
            half_size = BOX_SIZE // 2
            cv2.rectangle(output_frame, (x - half_size, y - half_size), (x + half_size, y + half_size), LINE_COLOR, LINE_THICKNESS)
            cv2.putText(output_frame, obj.text, (x - half_size, y - half_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOR, 1)
        if len(tracked_objects) > 1:
            num_lines = len(tracked_objects) // 2
            temp_list = random.sample(tracked_objects, len(tracked_objects))
            for i in range(num_lines):
                pt1 = tuple(map(int, temp_list[i*2].point))
                pt2 = tuple(map(int, temp_list[i*2 + 1].point))
                cv2.line(output_frame, pt1, pt2, LINE_COLOR, LINE_THICKNESS)

    prev_gray = current_gray.copy()
    frame_count += 1
    return output_frame

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        lang = LOCALIZATION[CURRENT_LANG]
        self.title(lang["set_title"])
        self.geometry("420x520")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        ctk.CTkLabel(self, text=lang["set_max"]).pack(pady=(15, 2))
        self.ent_max = ctk.CTkEntry(self, width=260)
        self.ent_max.insert(0, str(MAX_TRACKERS))
        self.ent_max.pack()

        ctk.CTkLabel(self, text=lang["set_interval"]).pack(pady=(10, 2))
        self.ent_int = ctk.CTkEntry(self, width=260)
        self.ent_int.insert(0, str(REDETECTION_INTERVAL))
        self.ent_int.pack()

        ctk.CTkLabel(self, text=lang["set_size"]).pack(pady=(10, 2))
        self.ent_size = ctk.CTkEntry(self, width=260)
        self.ent_size.insert(0, str(BOX_SIZE))
        self.ent_size.pack()

        ctk.CTkLabel(self, text=lang["set_words"]).pack(pady=(10, 2))
        self.ent_words = ctk.CTkEntry(self, width=260)
        self.ent_words.insert(0, ", ".join(WORDS))
        self.ent_words.pack()

        ctk.CTkLabel(self, text=lang["set_lang"]).pack(pady=(10, 2))
        self.combo_lang = ctk.CTkComboBox(self, values=["en", "ru", "uk", "es"], width=260)
        self.combo_lang.set(CURRENT_LANG)
        self.combo_lang.pack()

        ctk.CTkButton(self, text=lang["set_save"], fg_color="#007acc", hover_color="#005999", command=self.save_settings).pack(pady=(35, 20))

    def save_settings(self):
        global MAX_TRACKERS, REDETECTION_INTERVAL, BOX_SIZE, WORDS, CURRENT_LANG, feature_params
        lang = LOCALIZATION[CURRENT_LANG]
        try:
            MAX_TRACKERS = int(self.ent_max.get())
            REDETECTION_INTERVAL = int(self.ent_int.get())
            BOX_SIZE = int(self.ent_size.get())
            WORDS = [w.strip() for w in self.ent_words.get().split(",") if w.strip()]
            CURRENT_LANG = self.combo_lang.get()
            feature_params["maxCorners"] = MAX_TRACKERS
            self.parent.save_config()
            self.parent.update_localization()
            self.destroy()
        except ValueError:
            messagebox.showerror(lang["status_error"], lang["set_error"])

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Широкое, вытянутое окно по ТЗ (не квадратное гробище)
        self.geometry("620x450")
        self.resizable(False, False)

        self.lbl_input = ctk.CTkLabel(self)
        self.lbl_input.grid(row=0, column=0, padx=24, pady=(24, 2), sticky="w")
        self.ent_input = ctk.CTkEntry(self, width=430)
        self.ent_input.grid(row=1, column=0, padx=(24, 5), pady=0, sticky="w")
        self.btn_input = ctk.CTkButton(self, width=120, command=self.browse_input)
        self.btn_input.grid(row=1, column=1, padx=(0, 24), pady=0)

        self.lbl_output = ctk.CTkLabel(self)
        self.lbl_output.grid(row=2, column=0, padx=24, pady=(15, 2), sticky="w")
        self.ent_output = ctk.CTkEntry(self, width=430)
        self.ent_output.grid(row=3, column=0, padx=(24, 5), pady=0, sticky="w")
        self.btn_output = ctk.CTkButton(self, width=120, command=self.browse_output)
        self.btn_output.grid(row=3, column=1, padx=(0, 24), pady=0)

        self.btn_settings = ctk.CTkButton(self, fg_color="#2b2b2b", hover_color="#3a3a3a", border_width=1, border_color="#444444", command=self.open_settings)
        self.btn_settings.grid(row=4, column=0, padx=24, pady=(35, 0), sticky="w")
        self.btn_start = ctk.CTkButton(self, fg_color="#1a8cff", hover_color="#0073e6", font=("Helvetica", 14, "bold"), command=self.start_processing)
        self.btn_start.grid(row=4, column=1, padx=(0, 24), pady=(35, 0), sticky="e")

        self.progress_lbl = ctk.CTkLabel(self, text="", text_color="#1a8cff")
        self.progress_lbl.grid(row=5, column=0, columnspan=2, pady=(10, 0))

        # Адаптивный Canvas под правый нижний угол без обрезки
        self.canvas_wm = ctk.CTkCanvas(self, width=220, height=40, bg="#242424", highlightthickness=0)
        self.canvas_wm.grid(row=6, column=0, columnspan=2, padx=24, pady=(20, 20), sticky="se")

        self.anim_offset = 0.0
        self.load_config()
        self.update_localization()
        self.animate_watermark()

    def animate_watermark(self):
        self.canvas_wm.delete("all")
        
        # Смещение волны цвета
        self.anim_offset += 0.15
        
        text = "script by bl4ckgod"
        font_style = ("Helvetica", 14, "bold italic")
        
        current_x = 10
        for i, char in enumerate(text):
            temp_id = self.canvas_wm.create_text(0, -100, text=char, font=font_style, anchor="w")
            bbox = self.canvas_wm.bbox(temp_id)
            char_w = (bbox[2] - bbox[0]) if bbox else 11
            if char == " ":
                char_w = 7
            self.canvas_wm.delete(temp_id)

            # Формула СЛЕВА НАПРАВО (инвертировали знак у anim_offset)
            val = (math.sin((current_x * 0.04) - self.anim_offset) + 1) / 2
            
            # Перелив Белый -> Красный
            r = 255
            g = int(255 * (1 - val))
            b = int(255 * (1 - val))
            
            self.canvas_wm.create_text(
                current_x, 20, 
                text=char, 
                fill=f"#{r:02x}{g:02x}{b:02x}", 
                font=font_style, 
                anchor="w"
            )
            current_x += char_w

        self.after(30, self.animate_watermark)

    def update_localization(self):
        lang = LOCALIZATION[CURRENT_LANG]
        self.title(lang["title"])
        self.lbl_input.configure(text=lang["lbl_input"])
        self.lbl_output.configure(text=lang["lbl_output"])
        self.btn_input.configure(text=lang["btn_browse"])
        self.btn_output.configure(text=lang["btn_browse"])
        self.btn_settings.configure(text=lang["btn_settings"])
        self.btn_start.configure(text=lang["btn_create"])

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.ent_input.delete(0, ctk.END)
            self.ent_input.insert(0, path)
            self.save_config()

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if path:
            self.ent_output.delete(0, ctk.END)
            self.ent_output.insert(0, path)
            self.save_config()

    def open_settings(self):
        SettingsWindow(self)

    def save_config(self):
        data = {
            "input_path": self.ent_input.get(), "output_path": self.ent_output.get(),
            "max_trackers": MAX_TRACKERS, "redetection_interval": REDETECTION_INTERVAL,
            "box_size": BOX_SIZE, "words": WORDS, "current_lang": CURRENT_LANG
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_config(self):
        global MAX_TRACKERS, REDETECTION_INTERVAL, BOX_SIZE, WORDS, CURRENT_LANG, feature_params
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.ent_input.insert(0, data.get("input_path", ""))
                    self.ent_output.insert(0, data.get("output_path", ""))
                    MAX_TRACKERS = data.get("max_trackers", MAX_TRACKERS)
                    REDETECTION_INTERVAL = data.get("redetection_interval", REDETECTION_INTERVAL)
                    BOX_SIZE = data.get("box_size", BOX_SIZE)
                    WORDS = data.get("words", WORDS)
                    CURRENT_LANG = data.get("current_lang", CURRENT_LANG)
                    feature_params["maxCorners"] = MAX_TRACKERS
            except Exception:
                pass

    def start_processing(self):
        lang = LOCALIZATION[CURRENT_LANG]
        in_path = self.ent_input.get()
        out_path = self.ent_output.get()
        if not in_path or not out_path:
            messagebox.showwarning(lang["title"], lang["msg_warn_paths"])
            return
        self.save_config()
        self.btn_start.configure(state="disabled")
        self.btn_settings.configure(state="disabled")
        self.progress_lbl.configure(text=lang["status_running"])
        threading.Thread(target=self.process_video, args=(in_path, out_path), daemon=True).start()

    def process_video(self, in_path, out_path):
        global prev_gray, tracked_objects, frame_count, last_time
        lang = LOCALIZATION[CURRENT_LANG]
        try:
            prev_gray = None; tracked_objects = []; frame_count = 0; last_time = -1
            clip = mpe.VideoFileClip(in_path)
            final_clip = clip.transform(lambda gf, t: process_frame_with_tracking(gf(t), t))
            final_clip.write_videofile(out_path, codec='libx264', audio_codec='aac', logger=None)
            self.progress_lbl.configure(text=lang["status_done"])
            messagebox.showinfo(lang["title"], f"{lang['msg_success']}{out_path}")
        except Exception as e:
            self.progress_lbl.configure(text=lang["status_error"])
            messagebox.showerror(lang["status_error"], f"{lang['msg_error_proc']}{str(e)}")
        finally:
            self.btn_start.configure(state="normal")
            self.btn_settings.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()