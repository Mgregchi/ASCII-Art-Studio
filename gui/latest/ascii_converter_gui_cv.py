import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
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
    if not chars:
        chars = ASCII_PRESETS["Standard"]
    index = min(max(intensity // (256 // len(chars)), 0), len(chars) - 1)
    char = chars[index]
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
    try:
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image.")

        if not chars:
            chars = ASCII_PRESETS["Standard"]

        if use_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Normalize pixel values to 0-255
            image = np.clip(image, 0, 255).astype(np.uint8)

        if height is None:
            aspect_ratio = image.shape[0] / image.shape[1]
            height = int(width * aspect_ratio * 0.55)  # Adjust for char aspect
        height = max(1, height)  # Prevent zero height
        image = cv2.resize(image, (width, height),
                           interpolation=cv2.INTER_AREA)

        ascii_str = ""
        for i in range(height):
            for j in range(width):
                if use_color:
                    ascii_str += map_pixel_to_color(image[i, j], chars)
                else:
                    pixel = image[i, j]
                    index = min(
                        max(pixel // (256 // len(chars)), 0), len(chars) - 1)
                    ascii_str += chars[index]
            ascii_str += "\n"
        return ascii_str.rstrip("\n")
    except Exception as e:
        print(f"Error in image_to_ascii: {str(e)}")
        raise


def render_ascii_to_image(ascii_str, font_path=None, font_size=10):
    """Render ASCII string as Pillow image with uniform spacing (strip ANSI codes)."""
    ascii_str = re.sub(r'\033\[[0-9;]*m', '', ascii_str)
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path and os.path.exists(
            font_path) else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    lines = ascii_str.split("\n")
    if not lines:
        return Image.new("RGB", (100, 100), (255, 255, 255)), 100, 100

    # Calculate uniform spacing (force monospaced behavior)
    char_bbox = font.getbbox('W')  # Use 'W' for widest char
    char_width = char_bbox[2] - char_bbox[0]
    line_height = char_bbox[3] - char_bbox[1] + \
        2  # Add padding to avoid overlap

    max_line_len = max(len(line) for line in lines)
    img_width = max(1, max_line_len * char_width)
    img_height = max(1, len(lines) * line_height)

    img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = 0
    for line in lines:
        x = 0
        for char in line:
            draw.text((x, y), char, font=font, fill=(0, 0, 0))
            x += char_width
        y += line_height

    return img, img_width, img_height


class ASCIIConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Art Converter")
        self.input_path = None
        self.font_path = None
        self.is_running = False
        self.thread = None

        # GUI Layout (same as before)
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

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

        ttk.Label(self.frame, text="Font File:").grid(
            row=5, column=0, sticky=tk.W)
        self.font_button = ttk.Button(
            self.frame, text="Select Font", command=self.select_font)
        self.font_button.grid(row=5, column=1)
        self.font_label = ttk.Label(self.frame, text="Default font")
        self.font_label.grid(row=5, column=2, columnspan=3)

        ttk.Label(self.frame, text="FPS (Real-Time/MP4):").grid(row=6,
                                                                column=0, sticky=tk.W)
        self.fps_var = tk.IntVar(value=15)
        ttk.Scale(self.frame, from_=1, to=30, variable=self.fps_var,
                  orient=tk.HORIZONTAL).grid(row=6, column=1, columnspan=2)
        self.fps_label = ttk.Label(self.frame, textvariable=self.fps_var)
        self.fps_label.grid(row=6, column=3)

        self.color_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frame, text="Use Color (Terminal)",
                        variable=self.color_var).grid(row=7, column=0, columnspan=2)

        self.resolution_label = ttk.Label(
            self.frame, text="Output Resolution: N/A")
        self.resolution_label.grid(row=8, column=0, columnspan=5)

        ttk.Label(self.frame, text="Export:").grid(
            row=9, column=0, sticky=tk.W)
        self.txt_var = tk.BooleanVar(value=True)
        self.png_var = tk.BooleanVar(value=True)
        self.mp4_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.frame, text="TXT",
                        variable=self.txt_var).grid(row=9, column=1)
        ttk.Checkbutton(self.frame, text="PNG",
                        variable=self.png_var).grid(row=9, column=2)
        ttk.Checkbutton(self.frame, text="MP4 (Videos/Webcam)",
                        variable=self.mp4_var).grid(row=9, column=3)

        self.realtime_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.frame, text="Real-Time Terminal Display",
                        variable=self.realtime_var).grid(row=10, column=0, columnspan=2)

        self.preview = tk.Text(self.frame, height=20,
                               width=80, font=("Courier", 8))
        self.preview.grid(row=11, column=0, columnspan=5)

        self.process_button = ttk.Button(
            self.frame, text="Process", command=self.start_processing)
        self.process_button.grid(row=12, column=0, columnspan=2)
        self.stop_button = ttk.Button(
            self.frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=12, column=2, columnspan=2)

        self.width_var.trace("w", self.update_resolution_feedback)
        self.height_var.trace("w", self.update_resolution_feedback)
        self.font_size_var.trace("w", self.update_resolution_feedback)
        self.fps_var.trace("w", self.update_resolution_feedback)

    def toggle_input(self):
        self.file_button["state"] = "normal" if self.input_var.get(
        ) == "File" else "disabled"
        self.file_label["text"] = "No file selected" if self.input_var.get(
        ) == "File" else "Webcam selected"
        self.mp4_var.set(True if self.input_var.get() ==
                         "Webcam" else self.mp4_var.get())
        self.update_resolution_feedback()

    def toggle_char_input(self, *args):
        self.custom_chars["state"] = "normal" if self.char_var.get(
        ) == "Custom" else "disabled"

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images/Videos", "*.jpg *.jpeg *.png *.gif *.mp4 *.avi")])
        if file_path:
            self.input_path = file_path
            self.file_label["text"] = os.path.basename(file_path)
            self.update_resolution_feedback()

    def select_font(self):
        font_path = filedialog.askopenfilename(
            filetypes=[("Font Files", "*.ttf *.otf")])
        if font_path:
            try:
                ImageFont.truetype(font_path, 10)  # Test loading
                self.font_path = font_path
                self.font_label["text"] = os.path.basename(font_path)
            except Exception:
                messagebox.showerror(
                    "Error", "Invalid font file. Using default font.")
                self.font_path = None
                self.font_label["text"] = "Default font"
        else:
            self.font_path = None
            self.font_label["text"] = "Default font"

    def update_resolution_feedback(self, *args):
        width_chars = self.width_var.get()
        height_chars = self.height_var.get()
        font_size = self.font_size_var.get()
        fps = self.fps_var.get()
        pixel_width = width_chars * font_size * 0.6  # Approx
        pixel_height = height_chars * font_size
        self.resolution_label["text"] = f"Output: ~{int(pixel_width)}x{int(pixel_height)} pixels (PNG/MP4, {fps} FPS), {width_chars}x{height_chars} chars (Terminal/TXT)"

    def get_chars(self):
        if self.char_var.get() == "Custom":
            chars = self.custom_chars.get()
            if not chars:
                messagebox.showwarning(
                    "Warning", "Custom character set is empty. Using Standard preset.")
                return ASCII_PRESETS["Standard"]
            return chars
        return ASCII_PRESETS.get(self.char_var.get(), ASCII_PRESETS["Standard"])

    def process_image(self):
        try:
            image = cv2.imread(self.input_path)
            if image is None:
                raise ValueError("Invalid image file.")

            chars = self.get_chars()
            ascii_art = image_to_ascii(image, self.width_var.get(
            ), self.height_var.get(), chars, self.color_var.get())

            self.preview.delete(1.0, tk.END)
            self.preview.insert(tk.END, re.sub(
                r'\033\[[0-9;]*m', '', ascii_art))

            if self.realtime_var.get():
                print("\033[H\033[J")
                print(ascii_art)

            base_name = os.path.splitext(os.path.basename(self.input_path))[0]
            if self.txt_var.get():
                with open(f"{base_name}_ascii.txt", "w") as f:
                    f.write(re.sub(r'\033\[[0-9;]*m', '', ascii_art))
                print(f"Saved TXT: {base_name}_ascii.txt")

            if self.png_var.get():
                pil_img, _, _ = render_ascii_to_image(
                    ascii_art, self.font_path, self.font_size_var.get())
                pil_img.save(f"{base_name}_ascii.png")
                print(f"Saved PNG: {base_name}_ascii.png")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            print(f"Error in process_image: {str(e)}")
            self.stop_processing()

    def process_video(self, is_webcam):
        try:
            cap = cv2.VideoCapture(0 if is_webcam else self.input_path)
            if not cap.isOpened():
                raise ValueError("Invalid video source.")

            chars = self.get_chars()
            ascii_frames = []
            frame_count = 0
            start_time = time.time()
            base_name = "webcam" if is_webcam else os.path.splitext(
                os.path.basename(self.input_path))[0]
            fps = max(1, self.fps_var.get())

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                ascii_art = image_to_ascii(frame, self.width_var.get(
                ), self.height_var.get(), chars, self.color_var.get())
                ascii_frames.append(ascii_art)

                self.preview.delete(1.0, tk.END)
                self.preview.insert(tk.END, re.sub(
                    r'\033\[[0-9;]*m', '', ascii_art))

                if self.realtime_var.get():
                    sys.stdout.write("\033[H\033[J")
                    sys.stdout.write(ascii_art)
                    sys.stdout.flush()

                frame_count += 1
                time.sleep(1 / fps)

            cap.release()
            elapsed = time.time() - start_time
            print(
                f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count / elapsed:.2f} FPS)")

            if self.txt_var.get():
                for i, ascii_art in enumerate(ascii_frames):
                    with open(f"{base_name}_frame_{i:04d}.txt", "w") as f:
                        f.write(re.sub(r'\033\[[0-9;]*m', '', ascii_art))
                print(f"Saved {frame_count} TXT frames.")

            if self.mp4_var.get() and ascii_frames:
                # Render first frame to get dimensions
                pil_img, w, h = render_ascii_to_image(
                    ascii_frames[0], self.font_path, self.font_size_var.get())
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(
                    f"{base_name}_ascii.mp4", fourcc, fps, (w, h))

                for ascii_art in ascii_frames:
                    pil_img, _, _ = render_ascii_to_image(
                        ascii_art, self.font_path, self.font_size_var.get())
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    video.write(cv_img)

                video.release()
                print(f"Saved MP4: {base_name}_ascii.mp4")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process video: {str(e)}")
            print(f"Error in process_video: {str(e)}")

        self.stop_processing()

    def start_processing(self):
        if self.is_running:
            return

        if self.input_var.get() == "File" and not self.input_path:
            messagebox.showerror("Error", "Please select a file.")
            return

        if self.fps_var.get() < 1:
            messagebox.showerror("Error", "FPS must be at least 1.")
            return

        if self.width_var.get() < 1 or self.height_var.get() < 1:
            messagebox.showerror(
                "Error", "Width and height must be at least 1.")
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
            target=self.process_image if is_image else self.process_video,
            args=() if is_image else (is_webcam,)
        )
        self.thread.daemon = True
        self.thread.start()

    def stop_processing(self):
        self.is_running = False
        self.process_button["state"] = "normal"
        self.stop_button["state"] = "disabled"


def main():
    root = tk.Tk()
    app = ASCIIConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
