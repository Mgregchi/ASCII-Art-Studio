#!/usr/bin/env python3
"""
ASCII Art Studio Plus (OpenCV + Streamlit)
-----------------------------------------
Features:
- Image, Video, and Camera input modes (CLI).
- Batch image folder conversion.
- Transparent background export option (PNG with alpha).
- Custom TTF fonts via cv2.freetype (if installed, e.g. pip install opencv-contrib-python).
- Streamlit Web UI (--web) with sliders, file uploader, and live preview.

# Example usage:
    # Image
    python ascii_art_studio_plus.py --image photo.jpg

    # Video
    python ascii_art_studio_plus.py --video movie.mp4

    # Camera
    python ascii_art_studio_plus.py --camera 0

    # Batch folder
    python ascii_art_studio_plus.py --batch ./images --out ./ascii_out

    # Transparent ASCII PNG
    python ascii_art_studio_plus.py --image photo.jpg --transparent --out ascii.png

    # Web UI
    python ascii_art_studio_plus.py --web


Requirements:
    pip install opencv-python numpy streamlit

Optional:
    pip install opencv-contrib-python  (for cv2.freetype)
"""

import argparse
import os
import sys
import cv2 as cv
import numpy as np
from datetime import datetime

# Try to enable FreeType (optional)
ft2 = None

try:
    ft2 = cv.freetype.createFreeType2()
    FREETYPE_AVAILABLE = True
    # load a monospaced font (Courier, Consolas, etc.)
    font_path = "C:/Windows/Fonts/consola.ttf"  # update if needed
    if os.path.exists(font_path):
        ft2.loadFontData(fontFileName=font_path, id=0)
    else:
        print("[WARN] Font not found, FreeType fallback disabled")
        ft2 = None
except Exception as e:
    print("[WARN] FreeType not available:", e)
    ft2 = None
    FREETYPE_AVAILABLE = False

# Character sets
CHARSETS = [
    "@%#*+=-:. ",
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    "MWN@#&Q$%9876543210?!abc;:+=-,._ ",
    "█▓▒░ .",
    "#XOxo-.' ",
]


def compute_cell_size(font_scale, thickness, font_face=cv.FONT_HERSHEY_SIMPLEX, font_path=None):
    if font_path and FREETYPE_AVAILABLE:
        (w, h), baseline = ft2.getTextSize(
            "M", font_height=int(font_scale*20), thickness=thickness)
    else:
        (w, h), baseline = cv.getTextSize("M", font_face, font_scale, thickness)
    return w+2, h+2+baseline, h


def ascii_map(gray, charset, invert):
    n = len(charset)
    norm = gray.astype(np.float32)/255.0
    if invert:
        norm = 1-norm
    idx = np.clip((norm*(n-1)).round().astype(np.int32), 0, n-1)
    return idx


def render_ascii_frame_(frame_bgr, cols, font_scale, thickness, charset, invert, colorize,
                        transparent=False, font_path=None, show_original=False):
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell_w, cell_h, glyph_h = compute_cell_size(
        font_scale, thickness, font_path=font_path)
    cols = max(2, min(cols, w//cell_w))
    rows = max(1, int((h/w)*cols*(cell_w/cell_h)))
    small = cv.resize(gray, (cols, rows), interpolation=cv.INTER_AREA)
    if colorize:
        small_bgr = cv.resize(frame_bgr, (cols, rows),
                              interpolation=cv.INTER_AREA)
    else:
        small_bgr = None
    idx = ascii_map(small, charset, invert)
    out_h, out_w = rows*cell_h, cols*cell_w
    if transparent:
        canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    else:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    glyphs = np.array(list(charset))
    for i in range(rows):
        y = i*cell_h+glyph_h
        for j in range(cols):
            ch = glyphs[idx[i, j]]
            x = j*cell_w
            if colorize and small_bgr is not None:
                b, g, r = map(int, small_bgr[i, j])
                color = (b, g, r)
            else:
                val = 255 if invert else 230
                color = (val, val, val)
            if transparent:
                # Draw on RGBA
                if font_path and FREETYPE_AVAILABLE:
                    ft2.loadFontData(fontFileName=font_path, id=0)
                    ft2.putText(canvas, ch, (x, y), fontHeight=int(font_scale*20), color=(
                        color[0], color[1], color[2], 255), thickness=thickness, line_type=cv.LINE_AA, bottomLeftOrigin=False)
                else:
                    cv.putText(canvas, ch, (x, y), cv.FONT_HERSHEY_SIMPLEX,
                               font_scale, color+(255,), thickness, lineType=cv.LINE_AA)
            else:
                if font_path and FREETYPE_AVAILABLE:
                    ft2.loadFontData(fontFileName=font_path, id=0)
                    ft2.putText(canvas, ch, (x, y), fontHeight=int(font_scale*20), color=color,
                                thickness=thickness, line_type=cv.LINE_AA, bottomLeftOrigin=False)
                else:
                    cv.putText(canvas, ch, (x, y), cv.FONT_HERSHEY_SIMPLEX,
                               font_scale, color, thickness, lineType=cv.LINE_AA)
    if show_original:
        scale = out_h/h
        orig_w = int(w*scale)
        orig = cv.resize(frame_bgr, (orig_w, out_h))
        return np.hstack([orig, canvas])
    return canvas


def render_ascii_frame(frame_bgr, cols, font_scale, thickness, charset, invert, colorize,
                       transparent=False, font_path=None, font_height=None, show_original=False):
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Compute grid
    cell_w, cell_h, glyph_h = compute_cell_size(
        font_scale, thickness, font_path=font_path)
    cols = max(2, min(cols, w // cell_w))
    rows = max(1, int((h / w) * cols * (cell_w / cell_h)))

    # Downscale image for mapping
    small = cv.resize(gray, (cols, rows), interpolation=cv.INTER_AREA)
    small_bgr = cv.resize(frame_bgr, (cols, rows),
                          interpolation=cv.INTER_AREA) if colorize else None
    idx = ascii_map(small, charset, invert)

    # Output canvas
    out_h, out_w = rows * cell_h, cols * cell_w
    if transparent:
        canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    else:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    glyphs = np.array(list(charset))

    # If FreeType is available and a font path is given, load once
    ft2_inst = None
    if font_path and FREETYPE_AVAILABLE:
        try:
            ft2_inst = cv.freetype.createFreeType2()
            ft2_inst.loadFontData(fontFileName=font_path, id=0)
        except Exception as e:
            print(f"[WARN] Could not load FreeType font: {e}")
            ft2_inst = None

    # Default FreeType size if none provided
    if font_height is None:
        font_height = int(font_scale * 20)

    # Draw ASCII
    for i in range(rows):
        y = i * cell_h + glyph_h
        for j in range(cols):
            ch = glyphs[idx[i, j]]
            x = j * cell_w

            # Color
            if colorize and small_bgr is not None:
                b, g, r = map(int, small_bgr[i, j])
                color = (b, g, r)
            else:
                val = 255 if invert else 230
                color = (val, val, val)

            if transparent:
                if ft2_inst is not None:
                    ft2_inst.putText(canvas, ch, (x, y),
                                     fontHeight=font_height,
                                     color=(color[0], color[1], color[2], 255),
                                     thickness=thickness,
                                     line_type=cv.LINE_AA,
                                     bottomLeftOrigin=False)
                else:
                    cv.putText(canvas, ch, (x, y),
                               cv.FONT_HERSHEY_SIMPLEX,
                               font_scale, color + (255,),
                               thickness, lineType=cv.LINE_AA)
            else:
                if ft2_inst is not None:
                    ft2_inst.putText(canvas, ch, (x, y),
                                     fontHeight=font_height,
                                     color=color,
                                     thickness=thickness,
                                     line_type=cv.LINE_AA,
                                     bottomLeftOrigin=False)
                else:
                    cv.putText(canvas, ch, (x, y),
                               cv.FONT_HERSHEY_SIMPLEX,
                               font_scale, color,
                               thickness, lineType=cv.LINE_AA)

    # Show original next to ASCII
    if show_original:
        scale = out_h / h
        orig_w = int(w * scale)
        orig = cv.resize(frame_bgr, (orig_w, out_h))
        return np.hstack([orig, canvas])

    return canvas


def save_frame(image, prefix="ascii_frame"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.png"
    cv.imwrite(fname, image)
    return os.path.abspath(fname)


def batch_convert(input_dir, output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
            img = cv.imread(os.path.join(input_dir, file))
            if img is None:
                continue
            ascii_img = render_ascii_frame(img, **kwargs)
            out = os.path.join(output_dir, f"ascii_{file}.png")
            cv.imwrite(out, ascii_img)
            print("Saved", out)


def run_streamlit():
    import streamlit as st
    st.title("ASCII Art Studio (Web UI)")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    cols = st.slider("Columns", 40, 400, 160)
    font_scale = st.slider("Font scale", 1, 30, 10)/10.0
    thickness = st.slider("Thickness", 1, 5, 1)
    charset_idx = st.selectbox("Charset", list(range(len(CHARSETS))), index=0)
    invert = st.checkbox("Invert")
    colorize = st.checkbox("Colorize")
    transparent = st.checkbox("Transparent background")
    font_path = st.text_input("Custom TTF font path (optional)", value="")
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        ascii_img = render_ascii_frame(
            img, cols, font_scale, thickness, CHARSETS[charset_idx], invert, colorize, transparent, font_path if font_path else None)
        st.image(ascii_img, channels="BGR")
        if st.button("Save"):
            out = save_frame(ascii_img)
            st.success(f"Saved {out}")


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image")
    src.add_argument("--video")
    src.add_argument("--camera", type=int, nargs="?", const=0)
    src.add_argument("--batch")
    p.add_argument("--out")
    p.add_argument("--cols", type=int, default=160)
    p.add_argument("--font-scale", type=float, default=0.5)
    p.add_argument("--thickness", type=int, default=1)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--colorize", action="store_true")
    p.add_argument("--charset", type=int, default=0)
    p.add_argument("--transparent", action="store_true")
    p.add_argument("--font-path", type=str, default=None)
    p.add_argument("--show-original", action="store_true")
    p.add_argument("--web", action="store_true")
    args = p.parse_args()

    if args.web:
        run_streamlit()
        return
    kwargs = dict(cols=args.cols, font_scale=args.font_scale, thickness=args.thickness,
                  charset=CHARSETS[args.charset], invert=args.invert, colorize=args.colorize,
                  transparent=args.transparent, font_path=args.font_path, show_original=args.show_original)
    if args.image:
        img = cv.imread(args.image)
        out = render_ascii_frame(img, **kwargs)
        cv.imshow("ASCII", out)
        cv.waitKey(0)
        cv.destroyAllWindows()
    elif args.batch:
        batch_convert(args.batch, args.out or "ascii_batch", **kwargs)
    elif args.video:
        cap = cv.VideoCapture(args.video)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = render_ascii_frame(frame, **kwargs)
            cv.imshow("ASCII", out)
            if cv.waitKey(1) & 0xFF in (27, ord('q')):
                break
        cap.release()
        cv.destroyAllWindows()
    elif args.camera is not None:
        cap = cv.VideoCapture(args.camera)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = render_ascii_frame(frame, **kwargs)
            cv.imshow("ASCII", out)
            if cv.waitKey(1) & 0xFF in (27, ord('q')):
                break
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
