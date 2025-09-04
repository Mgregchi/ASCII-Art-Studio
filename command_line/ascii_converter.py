import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import os
import subprocess
import sys
import time
import io

# ASCII character ramp (dark to light)
ASCII_CHARS = '@%#*+=-:. '


def image_to_ascii(image, width=80, height=None):
    """Convert image to ASCII art string."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if height is None:
        aspect_ratio = image.shape[0] / image.shape[1]
        height = int(width * aspect_ratio * 0.55)  # Adjust for char aspect
    image = cv2.resize(image, (width, height))
    ascii_str = ''
    for row in image:
        for pixel in row:
            ascii_str += ASCII_CHARS[pixel // (256 // len(ASCII_CHARS))]
        ascii_str += '\n'
    return ascii_str.rstrip('\n')


def render_ascii_to_image(ascii_str, font_path=None, font_size=10):
    """Render ASCII string as PNG image."""
    font = ImageFont.load_default() if font_path is None else ImageFont.truetype(
        font_path, font_size)
    lines = ascii_str.split('\n')
    width = max(font.getbbox(line)[2] for line in lines)
    height = len(lines) * font_size
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=(0, 0, 0))
        y += font_size
    return img


def process_image(input_path, realtime, export):
    """Handle single image conversion."""
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Invalid image file.")
    ascii_art = image_to_ascii(image)

    if realtime == 'yes':
        print(ascii_art)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    exports = export.split(',') if export != 'all' else ['txt', 'png']

    if 'txt' in exports or export == 'all':
        with open(f"{base_name}_ascii.txt", 'w') as f:
            f.write(ascii_art)
        print(f"Saved TXT: {base_name}_ascii.txt")

    if 'png' in exports or export == 'all':
        img = render_ascii_to_image(ascii_art)
        img.save(f"{base_name}_ascii.png")
        print(f"Saved PNG: {base_name}_ascii.png")


def process_video(input_path, realtime, export, fps=10):
    """Handle video conversion."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Invalid video file.")

    frame_count = 0
    ascii_frames = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ascii_art = image_to_ascii(frame)
        ascii_frames.append(ascii_art)

        if realtime == 'yes':
            sys.stdout.write("\033[H\033[J")  # Clear terminal
            sys.stdout.write(ascii_art)
            sys.stdout.flush()
            time.sleep(1 / fps)  # Throttle FPS

        frame_count += 1

    cap.release()
    elapsed = time.time() - start_time
    print(
        f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count / elapsed:.2f} FPS)")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    exports = export.split(',') if export != 'all' else ['txt', 'mp4']

    if 'txt' in exports or export == 'all':
        for i, ascii_art in enumerate(ascii_frames):
            with open(f"{base_name}_frame_{i:04d}.txt", 'w') as f:
                f.write(ascii_art)
        print(f"Saved {frame_count} TXT frames.")

    if 'mp4' in exports or export == 'all':
        # Render frames as images and stitch with FFmpeg
        temp_dir = f"{base_name}_temp"
        os.makedirs(temp_dir, exist_ok=True)
        for i, ascii_art in enumerate(ascii_frames):
            img = render_ascii_to_image(ascii_art)
            img.save(f"{temp_dir}/frame_{i:04d}.png")

        # Use FFmpeg to create MP4 (assumes FFmpeg installed)
        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(
                fps), '-i', f"{temp_dir}/frame_%04d.png",
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f"{base_name}_ascii.mp4"
        ])
        # Clean up temp files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        print(f"Saved MP4: {base_name}_ascii.mp4")


def main():
    parser = argparse.ArgumentParser(
        description="ASCII Art Converter for Images/Videos")
    parser.add_argument('--input', required=True, help="Path to input file")
    parser.add_argument(
        '--type', choices=['image', 'video'], help="Input type (auto-detect if omitted)")
    parser.add_argument(
        '--realtime', choices=['yes', 'no'], default='yes', help="Real-time terminal display")
    parser.add_argument('--export', default='all',
                        help="Export formats: txt,png,mp4,all")
    args = parser.parse_args()

    file_ext = os.path.splitext(args.input)[1].lower()
    input_type = args.type or ('image' if file_ext in [
                               '.jpg', '.jpeg', '.png', '.gif'] else 'video')

    if input_type == 'image':
        process_image(args.input, args.realtime, args.export)
    elif input_type == 'video':
        process_video(args.input, args.realtime, args.export)
    else:
        raise ValueError("Unsupported input type.")


if __name__ == "__main__":
    main()
