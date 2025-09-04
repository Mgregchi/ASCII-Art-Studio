### Features
- **GUI**: Tkinter-based interface with:
  - File picker or webcam input selection.
  - Customization for ASCII character set (presets or custom), width/height sliders, and color toggle.
  - Export options (TXT, PNG, MP4) via checkboxes.
  - Preview area for ASCII output (static for images, single frame for videos/webcam).
  - Process/Stop buttons for starting/stopping real-time processing.
- **Real-Time**: Displays ASCII art in the terminal for videos/webcam (with GUI preview of the current frame).
- **Exports**: Saves to TXT (raw ASCII), PNG (rendered text image), or MP4 (video via FFmpeg).
- **Customizations**:
  - Character set: Choose from presets or enter a custom string.
  - Width/Height: Sliders for output resolution (20-200 chars wide, 10-100 chars tall).
  - Color: Checkbox for ANSI-colored output in terminal (RGB-mapped).
- **Webcam**: Supports live webcam feed as input.
- **No Audio**: Excluded as requested.

### Dependencies
- Python 3.x (Tkinter included).
- `opencv-python`: `pip install opencv-python`.
- `Pillow`: `pip install pillow`.
- FFmpeg: Install separately for MP4 export (`apt-get install ffmpeg`, `brew install ffmpeg`, or Windows binary).
- Optional: Monospaced font (e.g., 'Courier New') for PNG export; defaults to Pillow’s font.

### Usage
1. Save as `ascii_converter_gui.py`.
2. Run: `python ascii_converter_gui.py`.
3. In the GUI:
   - Select input (file or webcam).
   - Adjust settings (chars, width, height, color).
   - Choose exports (TXT, PNG, MP4).
   - Click "Process" to start (real-time in terminal, preview in GUI).
   - Click "Stop" for webcam/videos to halt processing.
   - Exports save in the input file’s directory (or current directory for webcam).
