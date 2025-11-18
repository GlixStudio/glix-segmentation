# MediaPipe Live Segmentation

A real-time video segmentation application using MediaPipe that segments people into different categories (background, hair, body-skin, face-skin, clothes, others) with texture support and face landmark detection.

## Features

- Real-time video segmentation with 6 categories
- Texture support for each category
- Face landmark detection and visualization
- GPU acceleration support
- Interactive controls via keyboard

## Prerequisites

- Conda (Anaconda or Miniconda)
- Webcam/camera
- GPU support (optional, but recommended for better performance)

## Installation

1. Create conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate yolo-detection
```

Note: The environment.yml uses conda for most packages and pip (within conda) for packages not available in conda-forge (mediapipe and opencv-contrib-python).

## Running the Application

Simply run the main script:

```bash
python mediapipe_live_segmentation.py
```

On first run, the application will automatically download the required MediaPipe models to a `models/` directory.

## Controls

- **'q'** - Quit the application
- **'r'** - Randomize textures for all categories
- **'t'** - Toggle text display (FPS, mode, etc.)
- **'p'** - Cycle processing size (256-640px)
- **'x'** - Toggle textures on/off
- **'f'** - Toggle fullscreen mode
- **'s'** - Save current frame to `captures/` directory

## Project Structure

```
glix-segmentation/
├── mediapipe_live_segmentation.py  # Main application
├── environment.yml                  # Conda environment file
├── textures/                        # Texture images for categories
│   ├── GANSTILLFINAL/
│   └── tiles2/
├── models/                          # Auto-downloaded MediaPipe models
└── captures/                        # Saved frames (created automatically)
```

## Notes

- The application will automatically detect and use GPU if available, otherwise falls back to CPU
- Textures are loaded from PNG files in the `textures/` folders
- Processing size affects both quality and performance (higher = better quality, lower FPS)
- Face landmarks are enabled by default and can be toggled via the control panel

## Troubleshooting

- **Camera not working**: Make sure your webcam is connected and not being used by another application
- **Low FPS**: Try reducing the processing size with 'p' key, or disable textures with 'x'
- **GPU errors**: The app will automatically fall back to CPU mode if GPU is unavailable

