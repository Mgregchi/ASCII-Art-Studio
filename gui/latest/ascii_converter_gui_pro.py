import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

# Character set from dark → light
BLOCKS = "█▓▒░ "


def image_to_ascii(img, cols=120, invert=False):
    """Convert an image (BGR) to ASCII using block chars."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Scale down based on cols
    cell_w = w / cols
    cell_h = 2 * cell_w  # adjust row height for font aspect ratio
    rows = int(h / cell_h)

    # Resize to fit ASCII grid
    small = cv.resize(gray, (cols, rows), interpolation=cv.INTER_AREA)

    # Normalize indexes
    if invert:
        small = 255 - small
    idx = (small / 256 * len(BLOCKS)).astype(int)
    idx = np.clip(idx, 0, len(BLOCKS) - 1)

    # Build ASCII string
    ascii_str = "\n".join("".join(BLOCKS[i] for i in row) for row in idx)
    return ascii_str


class AsciiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Art Studio (Tkinter)")

        # Text widget for ASCII output
        self.text = tk.Text(root, font=("Courier New", 8),
                            bg="black", fg="white")
        self.text.pack(fill=tk.BOTH, expand=True)

        # Menu
        menu = tk.Menu(root)
        root.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Camera", command=self.open_camera)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)

        # Video/camera feed
        self.cap = None
        self.update_job = None

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        img = cv.imread(path)
        ascii_str = image_to_ascii(img, cols=120, invert=False)
        self.show_ascii(ascii_str)

    def open_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv.VideoCapture(0)
        self.update_camera()

    def update_camera(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        ascii_str = image_to_ascii(frame, cols=120, invert=False)
        self.show_ascii(ascii_str)
        self.update_job = self.root.after(50, self.update_camera)

    def show_ascii(self, ascii_str):
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, ascii_str)


if __name__ == "__main__":
    root = tk.Tk()
    app = AsciiApp(root)
    root.mainloop()
