import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
import cv2 as cv
import numpy as np
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


def image_to_ascii(img, cols=120, charset="█▓▒░ ", invert=False,
                   hscale=2, vscale=1, colorize="gray", custom_color=(255, 255, 255)):
    """Convert an image (BGR) to ASCII string."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Grid cell size
    cell_w = w / cols
    cell_h = hscale * cell_w
    rows = int(h / cell_h)
    if rows <= 0 or cols <= 0:
        return ""

    # Resize
    small = cv.resize(gray, (cols, rows), interpolation=cv.INTER_AREA)
    small_bgr = cv.resize(img, (cols, rows), interpolation=cv.INTER_AREA)

    if invert:
        small = 255 - small

    # Map intensity to characters
    idx = (small / 256 * len(charset)).astype(int)
    idx = np.clip(idx, 0, len(charset) - 1)

    ascii_lines = []
    for i in range(rows):
        line = ""
        for j in range(cols):
            ch = charset[idx[i, j]]
            if colorize == "gray":
                line += ch
            elif colorize == "color":
                b, g, r = map(int, small_bgr[i, j])
                line += f"\x1b[38;2;{r};{g};{b}m{ch}\x1b[0m"
            elif colorize == "custom":
                line += ch
        ascii_lines.append(line * vscale)  # vertical scaling
    return "\n".join(ascii_lines)


class AsciiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Art Studio (Tkinter)")
        self.cap = None
        self.update_job = None
        self.paused = False
        self.video_writer = None
        self.frames = []

        # ASCII output
        self.text = tk.Text(root, font=("Courier New", 8),
                            bg="black", fg="white")
        self.text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Controls
        controls = tk.Frame(root, padx=5, pady=5)
        controls.pack(fill=tk.Y, side=tk.RIGHT)

        # Columns slider
        tk.Label(controls, text="Columns").pack()
        self.cols_var = tk.IntVar(value=120)
        tk.Scale(controls, from_=40, to=300, orient=tk.HORIZONTAL,
                 variable=self.cols_var).pack(fill=tk.X)

        # Charset dropdown
        tk.Label(controls, text="Charset").pack()
        self.charset_var = tk.StringVar(value="Blocks (█▓▒░)")
        ttk.Combobox(controls, textvariable=self.charset_var, values=list(
            CHARSETS.keys()), state="readonly").pack(fill=tk.X)

        # Invert toggle
        self.invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="Invert",
                       variable=self.invert_var).pack()

        # Spacing controls
        tk.Label(controls, text="Horizontal Scale").pack()
        self.hscale_var = tk.DoubleVar(value=2)
        tk.Scale(controls, from_=1, to=3, resolution=0.1,
                 orient=tk.HORIZONTAL, variable=self.hscale_var).pack(fill=tk.X)

        tk.Label(controls, text="Vertical Scale").pack()
        self.vscale_var = tk.IntVar(value=1)
        tk.Scale(controls, from_=1, to=3, orient=tk.HORIZONTAL,
                 variable=self.vscale_var).pack(fill=tk.X)
        # Flip / Mirror
        self.flip_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="Mirror Video",
                       variable=self.flip_var).pack()

        # Color options
        self.color_mode = tk.StringVar(value="gray")
        tk.Label(controls, text="Color Mode").pack()
        for mode in [("Gray", "gray"), ("Original", "color"), ("Custom", "custom")]:
            tk.Radiobutton(
                controls, text=mode[0], variable=self.color_mode, value=mode[1]).pack(anchor="w")
        tk.Button(controls, text="Pick Custom Color",
                  command=self.pick_color).pack(fill=tk.X)

        # Buttons
        tk.Button(controls, text="Open Image",
                  command=self.open_image).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Open Camera",
                  command=self.open_camera).pack(fill=tk.X, pady=2)

        self.pause_btn = tk.Button(
            controls, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(fill=tk.X, pady=2)
        self.pause_btn.pack_forget()  # hidden until camera starts

        tk.Button(controls, text="Export as TXT",
                  command=self.export_txt).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Export as PNG",
                  command=self.export_png).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Export as Video",
                  command=self.export_video).pack(fill=tk.X, pady=2)

        tk.Button(controls, text="Quit", command=root.quit).pack(
            fill=tk.X, pady=2)

        self.custom_color = (255, 255, 255)

    def pick_color(self):
        color_code = colorchooser.askcolor(title="Pick Custom Color")
        if color_code and color_code[0]:
            r, g, b = map(int, color_code[0])
            self.custom_color = (r, g, b)

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        img = cv.imread(path)
        self.render_ascii(img)

    def open_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv.VideoCapture(0)
        self.pause_btn.pack(fill=tk.X, pady=2)
        self.paused = False
        self.update_camera()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")

    def update_camera(self):
        if not self.cap or not self.cap.isOpened():
            return
        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                if self.flip_var.get():
                    frame = cv.flip(frame, 1)  # mirror horizontally

                self.frames.append(frame)
                self.render_ascii(frame)
        self.update_job = self.root.after(50, self.update_camera)

    def render_ascii(self, img):
        cols = self.cols_var.get()
        charset = CHARSETS[self.charset_var.get()]
        invert = self.invert_var.get()
        ascii_str = image_to_ascii(img, cols=cols, charset=charset, invert=invert,
                                   hscale=self.hscale_var.get(),
                                   vscale=self.vscale_var.get(),
                                   colorize=self.color_mode.get(),
                                   custom_color=self.custom_color)
        self.show_ascii(ascii_str)

    def show_ascii(self, ascii_str):
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, ascii_str)

    def export_txt(self):
        content = self.text.get("1.0", tk.END)
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    def export_png(self):
        content = self.text.get("1.0", tk.END)
        if not content.strip():
            return
        lines = content.split("\n")
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        line_height = 12
        width = max(len(line) for line in lines) * 8
        height = len(lines) * line_height
        img = np.zeros((height, width, 3), dtype=np.uint8)
        y = 15
        for line in lines:
            cv.putText(img, line, (5, y), font, font_scale,
                       (255, 255, 255), thickness, cv.LINE_AA)
            y += line_height
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv.imwrite(path, img)

    def export_video(self):
        if not self.frames:
            return
        h, w, _ = self.frames[0].shape
        path = filedialog.asksaveasfilename(defaultextension=".mp4")
        if not path:
            return
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(path, fourcc, 20.0, (w, h))
        for frame in self.frames:
            out.write(frame)
        out.release()
        print(f"[INFO] Video exported: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AsciiApp(root)
    root.mainloop()
