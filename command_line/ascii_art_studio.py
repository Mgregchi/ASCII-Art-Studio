#!/usr/bin/env python3
"""
ASCII Art Studio with OpenCV
----------------------------
- Supports: image file, video file, or live camera feed.
- Renders ASCII onto an OpenCV canvas using cv2.putText for sharp control.
- Interactive controls via trackbars (live preview):
    * Columns (character resolution)
    * Font scale
    * Thickness
    * Invert (light/dark mapping)
    * Colorize (grayscale vs color)
    * Charset selection (cycle with 'c' key)
- Save output video with --out path when processing a video file.
- Save single ASCII frame with 's' when viewing any mode (writes PNG).

Usage:
    python ascii_art_studio.py --image path/to/image.jpg
    python ascii_art_studio.py --video path/to/video.mp4
    python ascii_art_studio.py --camera 0
    python ascii_art_studio.py --video input.mp4 --out ascii_output.mp4

Keys:
    q or ESC  - quit
    s         - save current ASCII frame (PNG)
    c         - cycle charsets
    space     - pause/play (for video/camera)
"""

import argparse
import os
import sys
from typing import Tuple, List
import cv2 as cv
import numpy as np
from datetime import datetime

# Default character sets (from dense to sparse characters)
CHARSETS: List[str] = [
    "@%#*+=-:. ",  # Classic gradient
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",  # Extended
    "MWN@#&Q$%9876543210?!abc;:+=-,._ ",  # Medium
    "█▓▒░ .",  # Blocks
    "#XOxo-.' ",  # Simple shapes
]


def get_args():
    p = argparse.ArgumentParser(description="ASCII Art Studio with OpenCV")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Path to an image file")
    src.add_argument("--video", type=str, help="Path to a video file")
    src.add_argument("--camera", type=int, nargs='?',
                     const=0, help="Camera index (default 0)")
    p.add_argument("--out", type=str, default=None,
                   help="Output video path (for --video)")
    p.add_argument("--cols", type=int, default=160,
                   help="Target number of character columns (default: 160)")
    p.add_argument("--font-scale", type=float, default=0.5,
                   help="cv2.putText fontScale (default: 0.5)")
    p.add_argument("--thickness", type=int, default=1,
                   help="cv2.putText thickness (default: 1)")
    p.add_argument("--invert", action="store_true",
                   help="Invert luminance mapping")
    p.add_argument("--colorize", action="store_true",
                   help="Use per-cell average color instead of grayscale white")
    p.add_argument("--charset", type=int, default=0,
                   help=f"Charset index 0..{len(CHARSETS)-1} (default: 0)")
    p.add_argument("--no-ui", action="store_true",
                   help="Disable interactive UI and trackbars")
    p.add_argument("--max-height", type=int, default=1080,
                   help="Max preview window height (default: 1080)")
    p.add_argument("--show-original", action="store_true",
                   help="Show side-by-side original + ASCII")
    return p.parse_args()


def compute_cell_size(font_scale: float, thickness: int, font_face: int = cv.FONT_HERSHEY_SIMPLEX) -> Tuple[int, int, int]:
    # Use 'M' as a tall/wide reference glyph to size cells
    (w, h), baseline = cv.getTextSize("M", font_face, font_scale, thickness)
    # Pad a bit for spacing to avoid overlap
    pad_w = max(1, int(round(w * 0.15)))
    pad_h = max(1, int(round(h * 0.10)))
    cell_w = w + pad_w
    cell_h = h + pad_h + baseline
    return cell_w, cell_h, h  # return glyph height (h) for baseline calc


def ascii_map(values: np.ndarray, charset: str, invert: bool) -> np.ndarray:
    # values: 2D array of grayscale [0..255]; map to indices of charset
    ch_len = len(charset)
    if ch_len < 2:
        raise ValueError("Charset must have at least 2 characters")
    # Normalize to [0, 1]
    norm = values.astype(np.float32) / 255.0
    if invert:
        norm = 1.0 - norm
    # Map to [0..ch_len-1]
    idx = np.clip(
        (norm * (ch_len - 1)).round().astype(np.int32), 0, ch_len - 1)
    return idx


def render_ascii_frame(frame_bgr: np.ndarray,
                       cols: int,
                       font_scale: float,
                       thickness: int,
                       charset: str,
                       invert: bool,
                       colorize: bool,
                       show_original: bool) -> np.ndarray:
    # Convert to grayscale for luminance
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

    # Determine cell size
    cell_w, cell_h, glyph_h = compute_cell_size(font_scale, thickness)

    # Compute rows from desired columns and aspect
    h, w = gray.shape
    # clamp so we have at least 2 columns and fit horizontally
    cols = max(2, min(cols, w // max(1, cell_w)))
    # preserve aspect using cell aspect ratio
    rows = max(1, int(round((h / w) * cols * (cell_w / cell_h))))

    # Resize image to cells grid size
    # Downsample to grid resolution: (cols, rows)
    small = cv.resize(gray, (cols, rows), interpolation=cv.INTER_AREA)

    # For colorize, compute average color per cell by resizing each channel
    if colorize:
        small_bgr = cv.resize(frame_bgr, (cols, rows),
                              interpolation=cv.INTER_AREA)
    else:
        small_bgr = None

    # Map to charset
    idx = ascii_map(small, charset, invert)

    # Prepare canvas
    out_h = rows * cell_h
    out_w = cols * cell_w
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Precompute per-glyph string for faster putText
    glyphs = np.array(list(charset), dtype="<U1")

    # Render each cell
    for i in range(rows):
        y = i * cell_h + glyph_h  # baseline y
        for j in range(cols):
            ch = glyphs[idx[i, j]]
            x = j * cell_w
            if colorize and small_bgr is not None:
                b, g, r = map(int, small_bgr[i, j])
                color = (int(b), int(g), int(r))
            else:
                # grayscale white-ish for visibility
                val = int(255 if invert else 230)
                color = (val, val, val)
            # Draw character
            cv.putText(canvas, ch, (x, y), cv.FONT_HERSHEY_SIMPLEX,
                       font_scale, color, thickness, lineType=cv.LINE_AA)

    if show_original:
        # Create side-by-side view
        # Resize original to match height
        scale = out_h / h
        orig_w = int(round(w * scale))
        orig_resized = cv.resize(
            frame_bgr, (orig_w, out_h), interpolation=cv.INTER_AREA)
        combo = np.hstack([orig_resized, canvas])
        return combo

    return canvas


def ensure_window_(name: str):
    try:
        cv.getWindowProperty(name, cv.WND_PROP_VISIBLE)
    except Exception:
        cv.namedWindow(name, cv.WINDOW_NORMAL)


def ensure_window(name: str):
    # Always create the window if not existing
    try:
        cv.namedWindow(name, cv.WINDOW_NORMAL)
    except:
        pass


def build_ui(win: str, init_cols: int, init_font_scale: float, init_thickness: int, invert: bool, colorize: bool):
    ensure_window(win)
    cv.resizeWindow(win, 1280, 720)
    # Trackbars
    cv.createTrackbar("Cols", win, int(init_cols), 400, lambda v: None)
    cv.createTrackbar("FontScale x10", win, int(
        init_font_scale * 10), 100, lambda v: None)
    cv.createTrackbar("Thickness", win, int(init_thickness), 5, lambda v: None)
    cv.createTrackbar("Invert (0/1)", win,
                      int(1 if invert else 0), 1, lambda v: None)
    cv.createTrackbar("Colorize (0/1)", win,
                      int(1 if colorize else 0), 1, lambda v: None)


def read_ui_(win: str, fallback_cols: int, fallback_font_scale: float, fallback_thickness: int, fallback_invert: bool, fallback_colorize: bool):
    if cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 0:
        return fallback_cols, fallback_font_scale, fallback_thickness, fallback_invert, fallback_colorize
    cols = cv.getTrackbarPos("Cols", win) or fallback_cols
    fs10 = cv.getTrackbarPos("FontScale x10", win) or int(
        fallback_font_scale * 10)
    thickness = cv.getTrackbarPos("Thickness", win) or fallback_thickness
    invert = bool(cv.getTrackbarPos("Invert (0/1)", win))
    colorize = bool(cv.getTrackbarPos("Colorize (0/1)", win))
    return cols, max(1, fs10) / 10.0, max(1, thickness), invert, colorize


def read_ui(win: str, fallback_cols, fallback_font_scale, fallback_thickness, fallback_invert, fallback_colorize):
    # If window not visible, just return fallback values
    if cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
        return fallback_cols, fallback_font_scale, fallback_thickness, fallback_invert, fallback_colorize

    try:
        cols = cv.getTrackbarPos("Cols", win)
        fs10 = cv.getTrackbarPos("FontScale x10", win)
        thickness = cv.getTrackbarPos("Thickness", win)
        invert = bool(cv.getTrackbarPos("Invert (0/1)", win))
        colorize = bool(cv.getTrackbarPos("Colorize (0/1)", win))
    except cv.error:
        # Trackbars not ready yet → fall back
        return fallback_cols, fallback_font_scale, fallback_thickness, fallback_invert, fallback_colorize

    # Replace zeros with fallback (avoids "or" bug if trackbar is set to 0)
    if cols <= 0:
        cols = fallback_cols
    if fs10 <= 0:
        fs10 = int(fallback_font_scale * 10)
    if thickness <= 0:
        thickness = fallback_thickness

    return cols, fs10 / 10.0, thickness, invert, colorize


def save_frame_png(image: np.ndarray, prefix: str = "ascii_frame") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.png"
    cv.imwrite(fname, image)
    return os.path.abspath(fname)


def process_image(args):
    img = cv.imread(args.image)
    if img is None:
        print(f"Failed to read image: {args.image}")
        return
    win = "ASCII Art Studio"
    if not args.no_ui:
        build_ui(win, args.cols, args.font_scale,
                 args.thickness, args.invert, args.colorize)

    charset_idx = max(0, min(args.charset, len(CHARSETS) - 1))
    paused = False

    while True:
        cols, font_scale, thickness, invert, colorize = (
            args.cols, args.font_scale, args.thickness, args.invert, args.colorize)
        if not args.no_ui:
            cols, font_scale, thickness, invert, colorize = read_ui(
                win, cols, font_scale, thickness, invert, colorize)
        ascii_img = render_ascii_frame(
            img, cols, font_scale, thickness, CHARSETS[charset_idx], invert, colorize, args.show_original)

        # Limit display height if needed
        if ascii_img.shape[0] > args.max_height:
            scale = args.max_height / ascii_img.shape[0]
            disp = cv.resize(ascii_img, (int(
                ascii_img.shape[1] * scale), args.max_height), interpolation=cv.INTER_AREA)
        else:
            disp = ascii_img

        cv.imshow(win, disp)
        key = cv.waitKey(30) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('s'):
            path = save_frame_png(ascii_img)
            print(f"Saved: {path}")
        elif key == ord('c'):
            charset_idx = (charset_idx + 1) % len(CHARSETS)
        # space toggles pause (useful if we add future animations)
        elif key == 32:
            paused = not paused
        if paused:
            cv.waitKey(0)

    cv.destroyAllWindows()


def process_stream(capture: cv.VideoCapture, args, is_video: bool):
    if not capture.isOpened():
        print("Failed to open capture source")
        return

    # Output writer if saving video
    writer = None
    if is_video and args.out:
        # We will initialize writer once we have the first ASCII frame to know size
        fourcc = cv.VideoWriter_fourcc(
            *"mp4v") if args.out.lower().endswith(".mp4") else cv.VideoWriter_fourcc(*"XVID")

    win = "ASCII Art Studio"
    if not args.no_ui:
        build_ui(win, args.cols, args.font_scale,
                 args.thickness, args.invert, args.colorize)

    charset_idx = max(0, min(args.charset, len(CHARSETS) - 1))
    paused = False

    while True:
        if not paused:
            ret, frame = capture.read()
            if not ret:
                break
        else:
            # If paused, just wait for key events but keep displaying the same frame
            pass

        if frame is None:
            break

        cols, font_scale, thickness, invert, colorize = (
            args.cols, args.font_scale, args.thickness, args.invert, args.colorize)
        if not args.no_ui:
            cols, font_scale, thickness, invert, colorize = read_ui(
                win, cols, font_scale, thickness, invert, colorize)

        ascii_img = render_ascii_frame(
            frame, cols, font_scale, thickness, CHARSETS[charset_idx], invert, colorize, args.show_original)

        # Initialize writer if requested
        if is_video and args.out:
            if writer is None:
                fps = capture.get(cv.CAP_PROP_FPS)
                if not fps or fps <= 0 or fps > 120:
                    fps = 25.0  # fallback
                h, w = ascii_img.shape[:2]
                writer = cv.VideoWriter(args.out, fourcc, fps, (w, h))
                if not writer.isOpened():
                    print(
                        "Warning: failed to open VideoWriter; proceeding without saving.")
                    writer = None
            if writer is not None:
                writer.write(ascii_img)

        # Display
        disp = ascii_img
        if ascii_img.shape[0] > args.max_height:
            scale = args.max_height / ascii_img.shape[0]
            disp = cv.resize(ascii_img, (int(
                ascii_img.shape[1] * scale), args.max_height), interpolation=cv.INTER_AREA)
        cv.imshow(win, disp)

        key = cv.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('s'):
            path = save_frame_png(ascii_img)
            print(f"Saved: {path}")
        elif key == ord('c'):
            charset_idx = (charset_idx + 1) % len(CHARSETS)
        elif key == 32:  # space
            paused = not paused

    if writer is not None:
        writer.release()
    capture.release()
    cv.destroyAllWindows()


def main():
    args = get_args()

    # Validate inputs
    if args.image:
        if not os.path.isfile(args.image):
            print(f"Image not found: {args.image}")
            sys.exit(1)
        process_image(args)
        return

    if args.video:
        if not os.path.isfile(args.video):
            print(f"Video not found: {args.video}")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        process_stream(cap, args, is_video=True)
        return

    if args.camera is not None:
        cap = cv.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Failed to open camera index {args.camera}")
            sys.exit(1)
        process_stream(cap, args, is_video=False)
        return


if __name__ == "__main__":
    main()
