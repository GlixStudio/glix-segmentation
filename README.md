# MediaPipe Live Segmentation

A real-time video segmentation application using MediaPipe that segments people into different categories (background, hair, body-skin, face-skin, clothes, others) with texture support and face landmark detection.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- 🎭 Real-time video segmentation with 6 categories
- 🎨 Texture support for each category (5000+ textures included)
- 👤 Face landmark detection and visualization
- ⚡ GPU acceleration support (automatic fallback to CPU)
- 🎮 Interactive keyboard controls
- 💾 Memory-optimized for long-running sessions

## Prerequisites

- **Python 3.11** (Python 3.13+ not supported - MediaPipe compatibility)
- **Webcam/camera**
- **macOS, Linux, or Windows**

## Quick Start

### Option 1: Using the Run Script (Easiest)

```bash
# Make script executable (first time only)
chmod +x run.sh

# Run the application
./run.sh
```

The script will automatically:
- Extract textures if needed
- Create virtual environment if it doesn't exist
- Install dependencies
- Run the application

### Option 2: Manual Setup

1. **Install Python 3.11** (if not already installed)
   ```bash
   # macOS with Homebrew
   brew install python@3.11
   
   # Or download from python.org
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Extract textures** (if textures.zip exists)
   ```bash
   unzip -q textures.zip  # Creates textures/ directory
   ```

5. **Run the application**
   ```bash
   python mediapipe_live_segmentation.py
   ```

## First Run

On the first run, the application will:
- ✅ Automatically download MediaPipe models (~5-10MB) to `models/` directory
- ✅ Detect your webcam/camera
- ✅ Load random textures for all categories
- ✅ Start processing video in real-time

**Note:** Make sure your webcam is connected and not being used by another application.

## Keyboard Controls

While the application is running:

| Key | Action |
|-----|--------|
| **q** | Quit the application |
| **r** | Randomize textures for all categories |
| **t** | Toggle text display (FPS, mode, etc.) |
| **p** | Cycle processing size (256-640px) - affects quality/performance |
| **x** | Toggle textures on/off |
| **f** | Toggle fullscreen mode |
| **s** | Save current frame to `captures/` directory |

## Project Structure

```
glix-segmentation/
├── mediapipe_live_segmentation.py  # Main application
├── requirements.txt                 # Python dependencies
├── run.sh                          # Quick start script
├── textures.zip                    # Texture archive (extracted automatically)
├── textures/                        # Texture images (extracted from zip)
│   ├── GANSTILLFINAL/              # 1000+ textures
│   └── tiles2/                     # 3000+ textures
├── models/                         # MediaPipe models (auto-downloaded)
└── captures/                       # Saved frames (created automatically)
```

## Performance Tips

- **Higher processing size** = Better quality, lower FPS
- **Lower processing size** = Lower quality, higher FPS
- **Disable textures** (press 'x') for maximum FPS
- **GPU mode** provides 2-3x better performance than CPU

## Troubleshooting

### Camera not detected
- Make sure your webcam is connected
- Close other applications using the camera (Zoom, Teams, etc.)
- On macOS: Grant camera permissions in System Settings → Privacy & Security

### Low FPS / Performance issues
- Press **'p'** to reduce processing size
- Press **'x'** to disable textures temporarily
- The app automatically uses CPU if GPU is unavailable

### Import errors
- Make sure you've activated the virtual environment: `source venv/bin/activate`
- Ensure Python 3.11 is installed (not 3.13+)
- Reinstall dependencies: `pip install -r requirements.txt`

### Memory issues
- The application includes automatic memory management
- Segmenter is recreated periodically to prevent memory leaks
- If memory grows, reduce processing size or disable textures

### Models not downloading
- Check your internet connection
- Models download automatically on first run (~5-10MB)
- Models are saved to `models/` directory

## System Requirements

- **Python:** 3.11 (required - MediaPipe doesn't support 3.13+ yet)
- **OS:** macOS, Linux, or Windows
- **Camera:** Webcam or external camera
- **GPU:** Optional but recommended (Metal on macOS, CUDA on Linux/Windows)
- **RAM:** 4GB+ recommended
- **Disk:** ~100MB for models and dependencies

## Expected Performance

- **With GPU:** 30-60 FPS (depending on processing size)
- **With CPU:** 15-30 FPS (depending on processing size)
- **Memory:** ~200-500 MB during operation (optimized)

## Technical Details

- Uses MediaPipe's Image Segmenter for real-time segmentation
- Supports 6 categories: background, hair, body-skin, face-skin, clothes, others
- Includes 5000+ texture images for visual effects
- Memory-optimized with automatic segmenter recreation
- Thread-safe async processing with queue management

## License

MIT License - feel free to use this project for your own purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [MediaPipe](https://mediapipe.dev/)
- Uses [OpenCV](https://opencv.org/) for video processing
- Textures included for visual effects
