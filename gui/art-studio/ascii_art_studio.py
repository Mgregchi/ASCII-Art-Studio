import os
import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk


# ------------------------
# Renderer Classes
# ------------------------


class TextRenderer:
    def render(self, frame, cols=120, charset="█▓▒░ ", invert=False,
               hscale=1.0, vscale=2.0):
        """Convert frame (BGR) into ASCII text with spacing control."""

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Compute grid cell size with scaling factors
        cell_w = (w / cols) * hscale
        cell_h = cell_w * vscale
        rows = int(h / cell_h)

        if rows <= 0 or cols <= 0:
            return ""

        # Resize to fit ASCII grid
        small = cv.resize(gray, (cols, rows), interpolation=cv.INTER_AREA)

        # Invert if requested
        if invert:
            small = 255 - small

        # Map intensity → charset
        charset = charset or "█▓▒░ "  # fallback if empty
        chars = list(charset)
        idx = (small / 256 * len(chars)).astype(int)
        idx = np.clip(idx, 0, len(chars) - 1)

        # Build ASCII string
        ascii_str = "\n".join(
            "".join(chars[i] for i in row) for row in idx
        )
        return ascii_str


class TileRenderer:
    def __init__(self, tile_folder=None, tile_size=16):
        self.tiles = []
        self.tile_size = tile_size
        if tile_folder:
            self.load_tiles(tile_folder)

    def load_tiles(self, folder):
        """Load PNG tiles from folder and resize to tile_size."""
        self.tiles = []
        for f in sorted(os.listdir(folder)):
            path = os.path.join(folder, f)
            img = cv.imread(path, cv.IMREAD_UNCHANGED)  # keep alpha if present
            if img is not None:
                img = cv.resize(img, (self.tile_size, self.tile_size))
                if img.shape[2] == 4:  # if RGBA, convert to BGR
                    alpha = img[:, :, 3] / 255.0
                    bg = np.ones(
                        (self.tile_size, self.tile_size, 3), dtype=np.uint8) * 0
                    for c in range(3):
                        bg[:, :, c] = (1.0 - alpha) * \
                            bg[:, :, c] + alpha * img[:, :, c]
                    img = bg
                self.tiles.append(img)
        if not self.tiles:
            print("⚠️ No tiles loaded from", folder)

    def render(self, frame, cols=120):
        """Convert frame into a mosaic using tiles."""
        if not self.tiles:
            return "⚠️ No tiles loaded! Please set a tile folder."

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Compute grid
        cell_w = w / cols
        cell_h = cell_w
        rows = int(h / cell_h)
        if rows <= 0 or cols <= 0:
            return "⚠️ Invalid grid size"

        # Downsample to grid
        small = cv.resize(gray, (cols, rows), interpolation=cv.INTER_AREA)

        # Create blank canvas
        out = np.zeros((rows*self.tile_size, cols *
                       self.tile_size, 3), dtype=np.uint8)

        # Fill canvas with tiles
        for i in range(rows):
            for j in range(cols):
                idx = int(small[i, j] / 256 * len(self.tiles))
                idx = min(idx, len(self.tiles)-1)
                tile = self.tiles[idx]
                out[i*self.tile_size:(i+1)*self.tile_size,
                    j*self.tile_size:(j+1)*self.tile_size] = tile

        return out


class EdgeRenderer:
    def __init__(self, mode="normal", lower_green=(35, 80, 80), upper_green=(85, 255, 255)):
        """
        mode: "normal" or "greenscreen"
        lower_green, upper_green: HSV bounds for chroma keying
        """
        self.mode = mode
        self.lower_green = np.array(lower_green, dtype=np.uint8)
        self.upper_green = np.array(upper_green, dtype=np.uint8)
        self.line_color = (255, 255, 255)

    def set_mode(self, mode):
        if mode in ["normal", "greenscreen"]:
            self.mode = mode

    def set_color(self, bgr_tuple):
        self.line_color = bgr_tuple

    def render(self, frame, thickness=1):
        """
        Render stickman-like edges.
        Returns: grayscale edge image (BGR for display).
        """
        h, w, _ = frame.shape
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if self.mode == "normal":
            edges = cv.Canny(gray, 100, 200)

        elif self.mode == "greenscreen":
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, self.lower_green, self.upper_green)
            mask_inv = cv.bitwise_not(mask)
            fg = cv.bitwise_and(gray, gray, mask=mask_inv)
            edges = cv.Canny(fg, 100, 200)

        else:
            raise ValueError(f"Unknown EdgeRenderer mode: {self.mode}")

        # Thicken lines
        if thickness > 1:
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv.dilate(edges, kernel, iterations=1)

        # Convert to BGR for display
        # return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        # Colorize edges
        edge_bgr = np.zeros_like(frame)
        edge_bgr[edges > 0] = self.line_color
        return edge_bgr

# ------------------------
# Main App
# ------------------------


class AsciiStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Art Studio (Skeleton)")
        self.cap = None
        self.update_job = None

        # Colors
        self.fg_color = "white"
        self.bg_color = "black"

        # Output area
        # self.text = tk.Text(root, font=("Courier New", 8),
        #                     bg="black", fg="white")
        # self.text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        # Output area (both text + image, we toggle visibility)
        self.output_frame = tk.Frame(root, bg="black")
        self.output_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.text = tk.Text(self.output_frame, font=("Courier New", 8),
                            bg="black", fg="white")
        self.text.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.output_frame, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Start in text mode
        self.image_label.pack_forget()

        # Controls
        controls = tk.Frame(root, padx=5, pady=5)
        controls.pack(fill=tk.Y, side=tk.RIGHT)

        # Mode selection
        tk.Label(controls, text="Mode").pack()
        self.mode_var = tk.StringVar(value="Text")
        mode_menu = ttk.Combobox(controls, textvariable=self.mode_var,
                                 values=["Text", "Image Tiles", "Stickman"], state="readonly")
        mode_menu.pack(fill=tk.X)

        edge_frame = tk.LabelFrame(controls, text="Edge Renderer")
        edge_frame.pack(fill=tk.X, pady=5)

        tk.Button(edge_frame, text="Normal Edges",
                  command=lambda: self.edge_renderer.set_mode("normal")).pack(fill=tk.X)
        tk.Button(edge_frame, text="Green Screen Edges",
                  command=lambda: self.edge_renderer.set_mode("greenscreen")).pack(fill=tk.X)

        # Columns slider
        tk.Label(controls, text="Columns").pack()
        self.cols_var = tk.IntVar(value=120)
        tk.Scale(controls, from_=40, to=300, orient=tk.HORIZONTAL,
                 variable=self.cols_var).pack(fill=tk.X)

        # Charset entry (only for text mode)
        tk.Label(controls, text="Custom Charset").pack()
        self.charset_entry = tk.Entry(controls)
        self.charset_entry.pack(fill=tk.X)

        # Invert checkbox
        self.invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="Invert",
                       variable=self.invert_var).pack()

        # Thickness slider
        tk.Label(controls, text="Edge Thickness").pack()
        self.thickness_var = tk.IntVar(value=2)
        tk.Scale(controls, from_=1, to=10, orient=tk.HORIZONTAL,
                 variable=self.thickness_var).pack(fill=tk.X)

        # Line color selector
        tk.Button(controls, text="Pick Edge Color",
                  command=self.pick_edge_color).pack(fill=tk.X, pady=2)

        # Text colors
        tk.Button(controls, text="Pick Text FG",
                  command=self.pick_fg_color).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Pick Text BG",
                  command=self.pick_bg_color).pack(fill=tk.X, pady=2)

        # Flip / Mirror
        self.flip_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="Mirror Video",
                       variable=self.flip_var).pack()

        # Buttons
        tk.Button(controls, text="Open Image",
                  command=self.open_image).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Open Camera",
                  command=self.open_camera).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Stop Camera",
                  command=self.stop_camera).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Load Tile Folder",
                  command=self.load_tiles).pack(fill=tk.X, pady=2)
        tk.Button(controls, text="Export", command=self.export).pack(
            fill=tk.X, pady=2)
        tk.Button(controls, text="Quit", command=root.quit).pack(
            fill=tk.X, pady=2)
        # Renderers
        self.text_renderer = TextRenderer()
        self.tile_renderer = TileRenderer()
        self.edge_renderer = EdgeRenderer()

    def pick_edge_color(self):
        color = colorchooser.askcolor(title="Choose Edge Color")
        if color[0]:
            b, g, r = [int(c) for c in color[0]]
            self.edge_renderer.set_color((b, g, r))

    def pick_fg_color(self):
        color = colorchooser.askcolor(title="Choose Text Foreground")
        if color[1]:
            self.fg_color = color[1]
            self.text.config(fg=self.fg_color)

    def pick_bg_color(self):
        color = colorchooser.askcolor(title="Choose Text Background")
        if color[1]:
            self.bg_color = color[1]
            self.text.config(bg=self.bg_color)

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        img = cv.imread(path)
        self.render_ascii(img)

    def load_tiles(self):
        folder = filedialog.askdirectory()
        if folder:
            self.tile_renderer.load_tiles(folder)
            print(
                f"Loaded {len(self.tile_renderer.tiles)} tiles from {folder}")

    def open_camera(self):
        self.stop_camera()
        self.cap = cv.VideoCapture(0)
        self.update_camera()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.update_job:
            self.root.after_cancel(self.update_job)
            self.update_job = None

    def update_camera(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.flip_var.get():
            frame = cv.flip(frame, 1)  # mirror horizontally

        self.render_ascii(frame)
        self.update_job = self.root.after(50, self.update_camera)

    def render_ascii_(self, frame):
        mode = self.mode_var.get()
        if mode == "Text":
            charset = self.charset_entry.get() or "█▓▒░ "
            ascii_str = self.text_renderer.render(frame, cols=self.cols_var.get(),
                                                  charset=charset, invert=self.invert_var.get(),
                                                  hscale=1.0,   # we can later connect sliders here
                                                  vscale=2.0
                                                  )
        elif mode == "Image Tiles":
            if isinstance(self.tile_renderer.render(frame, cols=self.cols_var.get()), str):
                ascii_str = self.tile_renderer.render(
                    frame, cols=self.cols_var.get())
            else:
                # Instead of showing in text widget, show popup window with OpenCV
                img_out = self.tile_renderer.render(
                    frame, cols=self.cols_var.get())
                cv.imshow("Tile Mode Output", img_out)
                cv.waitKey(1)
                ascii_str = "Tile mosaic rendering active (see OpenCV window)"

        # elif mode == "Stickman":
        #     ascii_str = self.edge_renderer.render(frame)
        elif mode == "Stickman":
            edge_img = self.edge_renderer.render(
                frame, thickness=self.thickness_var.get()
            )
            self.show_output(edge_img, is_text=False)

        elif mode == "Stickman (Edges)":
            edge_img = self.edge_renderer.render(
                frame, thickness=self.thickness_var.get())
            cv.imshow("Edge Mode Output", edge_img)
            cv.waitKey(1)
            ascii_str = "Edge rendering active (see OpenCV window)"

        else:
            ascii_str = "Unknown mode"

        self.show_output(ascii_str)

    def render_ascii(self, frame):
        mode = self.mode_var.get()
        if mode == "Text":
            charset = self.charset_entry.get() or "█▓▒░ "
            ascii_str = self.text_renderer.render(
                frame,
                cols=self.cols_var.get(),
                charset=charset,
                invert=self.invert_var.get(),
                hscale=1.0,
                vscale=2.0
            )
            self.show_output(ascii_str, is_text=True)

        elif mode == "Image Tiles":
            img_out = self.tile_renderer.render(
                frame, cols=self.cols_var.get())
            if isinstance(img_out, str):  # error message
                self.show_output(img_out, is_text=True)
            else:
                self.show_output(img_out, is_text=False)

        elif mode == "Stickman":
            edge_img = self.edge_renderer.render(frame, thickness=2)
            self.show_output(edge_img, is_text=False)

        else:
            self.show_output("Unknown mode", is_text=True)

    # def show_output(self, output):
    #     self.text.delete("1.0", tk.END)
    #     self.text.insert(tk.END, str(output))

    def show_output(self, output, is_text=True):
        if is_text:
            # Show text widget, hide image
            self.image_label.pack_forget()
            self.text.pack(fill=tk.BOTH, expand=True)
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, str(output))
        else:
            # Show image, hide text
            self.text.pack_forget()
            self.image_label.pack(fill=tk.BOTH, expand=True)
            tk_img = self.cv_to_tk(output)
            self.image_label.configure(image=tk_img)
            self.image_label.image = tk_img  # prevent garbage collection

    def export(self):
        mode = self.mode_var.get()
        print(f"Export triggered in {mode} mode (to be implemented)")

    def cv_to_tk(self, cv_img):
        cv_rgb = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(cv_rgb)
        return ImageTk.PhotoImage(img_pil)


# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AsciiStudioApp(root)
    root.mainloop()
