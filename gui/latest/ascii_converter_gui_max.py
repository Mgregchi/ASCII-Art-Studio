import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Preset charsets
CHARSETS = {
    "Blocks (█▓▒░)": "█▓▒░ ",
    "ASCII Dense (@%#*:. )": "@%#*:. ",
    "ASCII Light ( .:-=+*#%@)": " .:-=+*#%@",
    "Binary (01)": "01",
    "Emoji (😎🔥💀✨)": "😎🔥💀✨ ",
    "Digits (1234567890)": "1234567890",
    # "Custom (set below)": ""  # user-defined
}


def image_to_ascii(img, cols=120, charset="█▓▒░ ", invert=False, spacing=(1, 1)):
    """Convert an image (BGR) to ASCII string."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Adjust grid with spacing
    cell_w = w / cols
    cell_h = 2 * cell_w * spacing[1]  # adjust row height for font aspect ratio
    rows = int(h / cell_h)
    if rows <= 0 or cols <= 0:
        return ""

    # Resize to grid
    small = cv.resize(gray, (cols, rows), interpolation=cv.INTER_AREA)

    # Invert if needed
    if invert:
        small = 255 - small

    # Map intensity → chars
    idx = (small / 256 * len(charset)).astype(int)
    idx = np.clip(idx, 0, len(charset) - 1)
    ascii_str = "\n".join(
        "".join(charset[i] + " " * (spacing[0] - 1) for i in row) for row in idx
    )
    return ascii_str


class AsciiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Art Studio (Tkinter)")
        self.cap = None
        self.update_job = None
        self.paused = False
        self.last_frame = None
        self.custom_color = "white"

        # ASCII output
        self.text = tk.Text(root, font=("Courier New", 8),
                            bg="black", fg="white")
        self.text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Controls frame
        controls = tk.Frame(root, padx=5, pady=5)
        controls.pack(fill=tk.Y, side=tk.RIGHT)

        # Columns slider
        tk.Label(controls, text="Columns").pack()
        self.cols_var = tk.IntVar(value=120)
        self.cols_slider = tk.Scale(controls, from_=40, to=300,
                                    orient=tk.HORIZONTAL, variable=self.cols_var)
        self.cols_slider.pack(fill=tk.X)

        # Charset dropdown
        tk.Label(controls, text="Charset").pack()
        self.charset_var = tk.StringVar(value="Blocks (█▓▒░)")
        charset_menu = ttk.Combobox(controls, textvariable=self.charset_var,
                                    values=list(CHARSETS.keys()), state="readonly")
        charset_menu.pack(fill=tk.X)

        # Invert toggle
        self.invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="Invert",
                       variable=self.invert_var).pack()

        # Spacing
        tk.Label(controls, text="Spacing X").pack()
        self.spacing_x = tk.IntVar(value=1)
        tk.Scale(controls, from_=1, to=5, orient=tk.HORIZONTAL,
                 variable=self.spacing_x).pack(fill=tk.X)

        tk.Label(controls, text="Spacing Y").pack()
        self.spacing_y = tk.IntVar(value=1)
        tk.Scale(controls, from_=1, to=5, orient=tk.HORIZONTAL,
                 variable=self.spacing_y).pack(fill=tk.X)

        # Color options
        self.colorize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="Enable Color",
                       variable=self.colorize_var).pack()
        tk.Button(controls, text="Pick Custom Color",
                  command=self.pick_color).pack(fill=tk.X, pady=2)

        # Buttons
        tk.Button(controls, text="Open Image",
                  command=self.open_image).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Open Camera",
                  command=self.open_camera).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Export TXT",
                  command=self.export_txt).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Export PNG",
                  command=self.export_png).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Export MP4",
                  command=self.export_mp4).pack(fill=tk.X, pady=2)

        # Pause/Play
        self.pause_btn = tk.Button(
            controls, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(fill=tk.X, pady=2)
        self.pause_btn.pack_forget()  # hidden by default

        tk.Button(controls, text="Quit", command=root.quit).pack(
            fill=tk.X, pady=2)

    def pick_color(self):
        color_code = colorchooser.askcolor(title="Choose text color")
        if color_code and color_code[1]:
            self.custom_color = color_code[1]
            self.text.config(fg=self.custom_color)

    def open_image(self):
        self.stop_camera()
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        img = cv.imread(path)
        self.render_ascii(img)

    def open_camera(self):
        self.stop_camera()
        self.cap = cv.VideoCapture(0)
        self.pause_btn.pack(fill=tk.X, pady=2)
        self.update_camera()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.update_job:
            self.root.after_cancel(self.update_job)
            self.update_job = None
        self.pause_btn.pack_forget()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Play" if self.paused else "Pause")

    def update_camera(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        if not self.paused:
            self.last_frame = frame
            self.render_ascii(frame)
        self.update_job = self.root.after(50, self.update_camera)

    def render_ascii(self, img):
        cols = self.cols_var.get()
        charset = CHARSETS[self.charset_var.get()]
        invert = self.invert_var.get()
        spacing = (self.spacing_x.get(), self.spacing_y.get())
        ascii_str = image_to_ascii(
            img, cols=cols, charset=charset, invert=invert, spacing=spacing)
        self.show_ascii(ascii_str)

    def show_ascii(self, ascii_str):
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, ascii_str)
        if self.colorize_var.get():
            self.text.config(fg=self.custom_color)
        else:
            self.text.config(fg="white")

    def export_txt(self):
        if not self.last_frame:
            return
        ascii_str = self.text.get("1.0", tk.END)
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(ascii_str)

    def export_png(self):
        if not self.last_frame:
            return
        ascii_str = self.text.get("1.0", tk.END)
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            font = ImageFont.truetype(
                "consola.ttf", 12) if os.name == "nt" else ImageFont.load_default()
            lines = ascii_str.split("\n")
            w = max([len(line) for line in lines]) * 10
            h = len(lines) * 14
            img = Image.new("RGB", (w, h), "black")
            draw = ImageDraw.Draw(img)
            for i, line in enumerate(lines):
                draw.text((0, i*14), line, font=font, fill=self.custom_color)
            img.save(path)

    def export_mp4(self):
        if not self.cap:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if not path:
            return
        # Export frames from camera as ASCII video
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(path, fourcc, 20.0, (640, 480))
        for _ in range(200):  # export ~10 seconds
            ret, frame = self.cap.read()
            if not ret:
                break
            ascii_str = image_to_ascii(frame, cols=self.cols_var.get(),
                                       charset=CHARSETS[self.charset_var.get()],
                                       invert=self.invert_var.get(),
                                       spacing=(self.spacing_x.get(), self.spacing_y.get()))
            img = self.ascii_to_image(ascii_str)
            frame_bgr = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
            frame_resized = cv.resize(frame_bgr, (640, 480))
            out.write(frame_resized)
        out.release()

    def ascii_to_image(self, ascii_str):
        font = ImageFont.truetype(
            "consola.ttf", 12) if os.name == "nt" else ImageFont.load_default()
        lines = ascii_str.split("\n")
        w = max([len(line) for line in lines]) * 10
        h = len(lines) * 14
        img = Image.new("RGB", (w, h), "black")
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            draw.text((0, i*14), line, font=font, fill=self.custom_color)
        return img


if __name__ == "__main__":
    root = tk.Tk()
    app = AsciiApp(root)
    root.mainloop()
