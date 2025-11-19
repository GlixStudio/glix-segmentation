# Quick Start Guide

## Option 1: Using Conda (Recommended)

### Step 1: Create and activate the conda environment
```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate yolo-detection
```

### Step 2: Run the application
```bash
python mediapipe_live_segmentation.py
```

---

## Option 2: Using pip (Simpler, if you already have Python)

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the application
```bash
python mediapipe_live_segmentation.py
```

---

## First Run

On the first run, the application will:
1. Automatically download MediaPipe models to a `models/` directory
2. Detect your webcam/camera
3. Start processing video in real-time

**Note:** Make sure your webcam is connected and not being used by another application.

---

## Keyboard Controls

While the application is running:

- **'q'** - Quit the application
- **'r'** - Randomize textures for all categories
- **'t'** - Toggle text display (FPS, mode, etc.)
- **'p'** - Cycle processing size (256-640px) - affects quality/performance
- **'x'** - Toggle textures on/off
- **'f'** - Toggle fullscreen mode
- **'s'** - Save current frame to `captures/` directory

---

## Troubleshooting

### Camera not detected
- Make sure your webcam is connected
- Close other applications using the camera (Zoom, Teams, etc.)
- On macOS, grant camera permissions in System Settings

### Low FPS / Performance issues
- Press **'p'** to reduce processing size (lower = faster)
- Press **'x'** to disable textures temporarily
- The app will automatically use CPU if GPU is unavailable

### Import errors
- Make sure you've activated the conda environment: `conda activate yolo-detection`
- Or install dependencies: `pip install -r requirements.txt`

### Models not downloading
- Check your internet connection
- The models will be downloaded to `models/` directory on first run
- Models are ~5-10MB total

---

## System Requirements

- **Python:** 3.11+ (recommended)
- **OS:** macOS, Linux, or Windows
- **Camera:** Webcam or external camera
- **GPU:** Optional but recommended for better performance
- **RAM:** 4GB+ recommended
- **Disk:** ~100MB for models and dependencies

---

## Expected Performance

- **With GPU:** 30-60 FPS (depending on processing size)
- **With CPU:** 15-30 FPS (depending on processing size)
- **Memory:** ~200-500 MB during operation (optimized version)

---

## Project Structure

```
glix-segmentation/
├── mediapipe_live_segmentation.py  # Main application
├── environment.yml                  # Conda environment file
├── requirements.txt                 # Pip requirements
├── textures/                        # Texture images (should exist)
│   ├── GANSTILLFINAL/
│   └── tiles2/
├── models/                          # Auto-downloaded (created on first run)
└── captures/                        # Saved frames (created automatically)
```

---

## Next Steps

1. Run the application
2. Press **'r'** to randomize textures
3. Press **'f'** for fullscreen mode
4. Experiment with different processing sizes using **'p'**

Enjoy! 🎉

