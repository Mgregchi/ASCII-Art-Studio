import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import subprocess
import sys
import time
import threading
import re

# Predefined ASCII character ramps (dark to light)
ASCII_PRESETS = {
    "Standard": "@%#*+=-:. ",
    "Detailed": "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    "Simple": "@#*+=-. "
}


def map_pixel_to_color(pixel, chars):
    """Map RGB pixel to ANSI-colored ASCII character."""
    r, g, b = pixel
    intensity = (r + g + b) // 3
    char = chars[intensity // (256 // len(chars))]
    # Simple RGB to ANSI (8 colors)
    if r > 200 and g < 100 and b < 100:
        return f"\033[31m{char}\033[0m"  # Red
    elif g > 200 and r < 100 and b < 100:
        return f"\033[32m{char}\033[0m"  # Green
    elif b > 200 and r < 100 and g < 100:
        return f"\033[34m{char}\033[0m"  # Blue
    elif r > 200 and g > 200 and b < 100:
        return f"\033[33m{char}\033[0m"  # Yellow
    else:
        return char


def image_to_ascii(image, width=80, height=None, chars=ASCII_PRESETS["Standard"], use_color=False):
    """Convert image to ASCII art string."""
    if use_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if height is None:
        aspect_ratio = image.shape[0] / image.shape[1]
        height = int(width * aspect_ratio * 0.55)  # Adjust for char aspect
    image = cv2.resize(image, (width, height),
                       interpolation=cv2.INTER_AREA)  # Optimize resizing
    ascii_str = ""
    for i in range(height):
        for j in range(width):
            if use_color:
                ascii_str += map_pixel_to_color(image[i, j], chars)
            else:
                pixel = image[i, j]
                ascii_str += chars[pixel // (256 // len(chars))]
        ascii_str += "\n"
    return ascii_str.rstrip("\n")


def render_ascii_to_image(ascii_str, font_path=None, font_size=10):
    """Render ASCII string as PNG image (strip ANSI codes)."""
    ascii_str = re.sub(r'\033\[[0-9;]*m', '', ascii_str)
    font = ImageFont.load_default() if font_path is None else ImageFont.truetype(
        font_path, font_size)
    lines = ascii_str.split("\n")
    # Estimate width using max line length
    width = max(font.getbbox(line)[2] for line in lines if line.strip())
    height = len(lines) * font_size
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=(0, 0, 0))
        y += font_size
    return img, width, height


class ASCIIConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Art Converter")
        self.input_path = None
        self.is_running = False
        self.thread = None

        # GUI Layout
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Input Selection
        ttk.Label(self.frame, text="Input:").grid(row=0, column=0, sticky=tk.W)
        self.input_var = tk.StringVar(value="File")
        ttk.Radiobutton(self.frame, text="File", variable=self.input_var,
                        value="File", command=self.toggle_input).grid(row=0, column=1)
        ttk.Radiobutton(self.frame, text="Webcam", variable=self.input_var,
                        value="Webcam", command=self.toggle_input).grid(row=0, column=2)
        self.file_button = ttk.Button(
            self.frame, text="Select File", command=self.select_file)
        self.file_button.grid(row=0, column=3)
        self.file_label = ttk.Label(self.frame, text="No file selected")
        self.file_label.grid(row=0, column=4, columnspan=2)

        # Customization
        ttk.Label(self.frame, text="ASCII Characters:").grid(
            row=1, column=0, sticky=tk.W)
        self.char_var = tk.StringVar(value="Standard")
        self.char_menu = ttk.Combobox(self.frame, textvariable=self.char_var, values=list(
            ASCII_PRESETS.keys()) + ["Custom"])
        self.char_menu.grid(row=1, column=1, columnspan=2)
        self.char_var.trace("w", self.toggle_char_input)
        self.custom_chars = ttk.Entry(self.frame, state="disabled")
        self.custom_chars.grid(row=1, column=3, columnspan=2)

        ttk.Label(self.frame, text="Width (chars):").grid(
            row=2, column=0, sticky=tk.W)
        self.width_var = tk.IntVar(value=80)
        ttk.Scale(self.frame, from_=20, to=200, variable=self.width_var,
                  orient=tk.HORIZONTAL).grid(row=2, column=1, columnspan=2)
        self.width_label = ttk.Label(self.frame, textvariable=self.width_var)
        self.width_label.grid(row=2, column=3)

        ttk.Label(self.frame, text="Height (chars):").grid(
            row=3, column=0, sticky=tk.W)
        self.height_var = tk.IntVar(value=40)
        ttk.Scale(self.frame, from_=10, to=100, variable=self.height_var,
                  orient=tk.HORIZONTAL).grid(row=3, column=1, columnspan=2)
        self.height_label = ttk.Label(self.frame, textvariable=self.height_var)
        self.height_label.grid(row=3, column=3)

        ttk.Label(self.frame, text="Font Size (PNG/MP4):").grid(row=4,
                                                                column=0, sticky=tk.W)
        self.font_size_var = tk.IntVar(value=10)
        ttk.Scale(self.frame, from_=8, to=20, variable=self.font_size_var,
                  orient=tk.HORIZONTAL).grid(row=4, column=1, columnspan=2)
        self.font_size_label = ttk.Label(
            self.frame, textvariable=self.font_size_var)
        self.font_size_label.grid(row=4, column=3)

        self.color_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frame, text="Use Color (Terminal)",
                        variable=self.color_var).grid(row=5, column=0, columnspan=2)

        # Resolution Feedback
        self.resolution_label = ttk.Label(
            self.frame, text="Output Resolution: N/A")
        self.resolution_label.grid(row=6, column=0, columnspan=5)

        # Export Options
        ttk.Label(self.frame, text="Export:").grid(
            row=7, column=0, sticky=tk.W)
        self.txt_var = tk.BooleanVar(value=True)
        self.png_var = tk.BooleanVar(value=True)
        self.mp4_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.frame, text="TXT",
                        variable=self.txt_var).grid(row=7, column=1)
        ttk.Checkbutton(self.frame, text="PNG",
                        variable=self.png_var).grid(row=7, column=2)
        ttk.Checkbutton(self.frame, text="MP4 (Videos/Webcam)",
                        variable=self.mp4_var).grid(row=7, column=3)

        # Real-Time Toggle
        self.realtime_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.frame, text="Real-Time Terminal Display",
                        variable=self.realtime_var).grid(row=8, column=0, columnspan=2)

        # Preview Area
        self.preview = tk.Text(self.frame, height=20,
                               width=80, font=("Courier", 8))
        self.preview.grid(row=9, column=0, columnspan=5)

        # Buttons
        self.process_button = ttk.Button(
            self.frame, text="Process", command=self.start_processing)
        self.process_button.grid(row=10, column=0, columnspan=2)
        self.stop_button = ttk.Button(
            self.frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=10, column=2, columnspan=2)

        # Update resolution feedback
        self.width_var.trace("w", self.update_resolution_feedback)
        self.height_var.trace("w", self.update_resolution_feedback)
        self.font_size_var.trace("w", self.update_resolution_feedback)

    def toggle_input(self):
        """Enable/disable file button based on input type."""
        self.file_button["state"] = "normal" if self.input_var.get(
        ) == "File" else "disabled"
        self.file_label["text"] = "No file selected" if self.input_var.get(
        ) == "File" else "Webcam selected"
        self.mp4_var.set(True if self.input_var.get() ==
                         "Webcam" else self.mp4_var.get())
        self.update_resolution_feedback()

    def toggle_char_input(self, *args):
        """Enable custom char entry if Custom is selected."""
        self.custom_chars["state"] = "normal" if self.char_var.get(
        ) == "Custom" else "disabled"

    def select_file(self):
        """Open file dialog for image/video selection."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Images/Videos", "*.jpg *.jpeg *.png *.gif *.mp4 *.avi")])
        if file_path:
            self.input_path = file_path
            self.file_label["text"] = os.path.basename(file_path)
            self.update_resolution_feedback()

    def update_resolution_feedback(self, *args):
        """Update estimated output resolution for PNG/MP4."""
        width_chars = self.width_var.get()
        height_chars = self.height_var.get()
        font_size = self.font_size_var.get()
        # Estimate pixel dimensions (approximate, depends on font)
        pixel_width = width_chars * font_size * 0.6  # Approx char width
        pixel_height = height_chars * font_size
        self.resolution_label["text"] = f"Output Resolution: ~{int(pixel_width)}x{int(pixel_height)} pixels (PNG/MP4), {width_chars}x{height_chars} chars (Terminal/TXT)"

    def get_chars(self):
        """Get selected or custom ASCII characters."""
        if self.char_var.get() == "Custom":
            chars = self.custom_chars.get()
            return chars if chars else ASCII_PRESETS["Standard"]
        return ASCII_PRESETS.get(self.char_var.get(), ASCII_PRESETS["Standard"])

    def process_image(self):
        """Process a single image."""
        image = cv2.imread(self.input_path)
        if image is None:
            messagebox.showerror("Error", "Invalid image file.")
            return

        chars = self.get_chars()
        ascii_art = image_to_ascii(image, self.width_var.get(
        ), self.height_var.get(), chars, self.color_var.get())

        # Update GUI preview (strip ANSI codes)
        self.preview.delete(1.0, tk.END)
        self.preview.insert(tk.END, re.sub(r'\033\[[0-9;]*m', '', ascii_art))

        if self.realtime_var.get():
            print("\033[H\033[J")
            print(ascii_art)

        base_name = os.path.splitext(os.path.basename(self.input_path))[0]
        if self.txt_var.get():
            with open(f"{base_name}_ascii.txt", "w") as f:
                f.write(re.sub(r'\033\[[0-9;]*m', '', ascii_art))
            print(f"Saved TXT: {base_name}_ascii.txt")

        if self.png_var.get():
            img, _, _ = render_ascii_to_image(
                ascii_art, font_size=self.font_size_var.get())
            img.save(f"{base_name}_ascii.png")
            print(f"Saved PNG: {base_name}_ascii.png")

    def process_video(self, is_webcam):
        """Process video or webcam feed."""
        cap = cv2.VideoCapture(0 if is_webcam else self.input_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Invalid video source.")
            self.stop_processing()
            return

        chars = self.get_chars()
        ascii_frames = []
        frame_count = 0
        start_time = time.time()
        base_name = "webcam" if is_webcam else os.path.splitext(
            os.path.basename(self.input_path))[0]

        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ascii_art = image_to_ascii(frame, self.width_var.get(
            ), self.height_var.get(), chars, self.color_var.get())
            ascii_frames.append(ascii_art)

            # Update GUI preview (strip ANSI codes)
            self.preview.delete(1.0, tk.END)
            self.preview.insert(tk.END, re.sub(
                r'\033\[[0-9;]*m', '', ascii_art))

            if self.realtime_var.get():
                sys.stdout.write("\033[H\033[J")
                sys.stdout.write(ascii_art)
                sys.stdout.flush()

            frame_count += 1
            time.sleep(1 / 15)  # Cap at ~15 FPS for real-time

        cap.release()
        elapsed = time.time() - start_time
        print(
            f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count / elapsed:.2f} FPS)")

        if self.txt_var.get():
            for i, ascii_art in enumerate(ascii_frames):
                with open(f"{base_name}_frame_{i:04d}.txt", "w") as f:
                    f.write(re.sub(r'\033\[[0-9;]*m', '', ascii_art))
            print(f"Saved {frame_count} TXT frames.")

        if self.mp4_var.get():
            temp_dir = f"{base_name}_temp"
            os.makedirs(temp_dir, exist_ok=True)
            for i, ascii_art in enumerate(ascii_frames):
                img, _, _ = render_ascii_to_image(
                    ascii_art, font_size=self.font_size_var.get())
                img.save(f"{temp_dir}/frame_{i:04d}.png")

            subprocess.run([
                "ffmpeg", "-y", "-framerate", "15", "-i", f"{temp_dir}/frame_%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{base_name}_ascii.mp4"
            ])
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            print(f"Saved MP4: {base_name}_ascii.mp4")

        self.stop_processing()

    def start_processing(self):
        """Start processing based on input type."""
        if self.is_running:
            return

        if self.input_var.get() == "File" and not self.input_path:
            messagebox.showerror("Error", "Please select a file.")
            return

        self.is_running = True
        self.process_button["state"] = "disabled"
        self.stop_button["state"] = "normal"

        is_webcam = self.input_var.get() == "Webcam"
        file_ext = os.path.splitext(self.input_path)[
            1].lower() if self.input_path else ""
        is_image = file_ext in [".jpg", ".jpeg",
                                ".png", ".gif"] and not is_webcam

        self.thread = threading.Thread(
            target=self.process_image if is_image else self.process_video, args=(is_webcam,))
        self.thread.daemon = True
        self.thread.start()

    def stop_processing(self):
        """Stop real-time processing."""
        self.is_running = False
        self.process_button["state"] = "normal"
        self.stop_button["state"] = "disabled"


def main():
    root = tk.Tk()
    app = ASCIIConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
