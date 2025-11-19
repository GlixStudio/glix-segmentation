"""Live video feed with MediaPipe Image Segmentation - Optimized"""

import warnings
import os
# Suppress MediaPipe and system warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import threading
import time
import urllib.request
import random
import glob
import platform
from collections import OrderedDict
import gc  # For explicit garbage collection
import tracemalloc  # For memory leak detection

def get_screen_resolution():
    """Get the primary screen resolution.
    
    Note: platform.system() == 'Darwin' detects macOS because macOS is built on Darwin,
    the underlying Unix-like operating system kernel. Darwin is the open-source component
    of macOS, so checking for 'Darwin' is the standard way to detect macOS in Python.
    """
    try:
        if platform.system() == 'Darwin':  # macOS (Darwin is the underlying OS kernel)
            try:
                from AppKit import NSScreen
                screen = NSScreen.mainScreen()
                frame = screen.frame()
                return int(frame.size.width), int(frame.size.height)
            except ImportError:
                # Fallback to tkinter if AppKit not available
                import tkinter as tk
                root = tk.Tk()
                width = root.winfo_screenwidth()
                height = root.winfo_screenheight()
                root.destroy()
                return width, height
        elif platform.system() == 'Windows':
            import tkinter as tk
            root = tk.Tk()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        else:  # Linux
            import tkinter as tk
            root = tk.Tk()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
    except:
        # Fallback: return a common resolution
        return 1920, 1080

def resize_to_fill_screen(frame, target_width, target_height):
    """Resize frame to fill screen while maintaining aspect ratio (cover mode).
    This ensures no gray areas appear - the frame will be cropped if needed."""
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate aspect ratios
    frame_aspect = frame_width / frame_height
    screen_aspect = target_width / target_height
    
    # Scale to cover the entire screen (fill mode)
    if frame_aspect > screen_aspect:
        # Frame is wider - scale based on height, crop width
        scale = target_height / frame_height
        new_width = int(frame_width * scale)
        new_height = target_height
    else:
        # Frame is taller - scale based on width, crop height
        scale = target_width / frame_width
        new_width = target_width
        new_height = int(frame_height * scale)
    
    # Resize the frame
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Crop to exact screen size if needed (center crop)
    if new_width != target_width or new_height != target_height:
        start_x = (new_width - target_width) // 2
        start_y = (new_height - target_height) // 2
        # Create a view/copy for the cropped region
        cropped = resized[start_y:start_y + target_height, start_x:start_x + target_width].copy()
        del resized  # Explicitly delete intermediate array
        return cropped
    
    return resized

# Pre-computed constants for performance
# Categories: 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others
CATEGORY_COLORS = np.array([
    [100, 150, 200],  # Background - blue
    [255, 50, 255],   # Hair - bright magenta (more visible)
    [100, 255, 100],  # Body-skin - green
    [255, 150, 255],  # Face-skin - pink
    [255, 100, 150],  # Clothes - hot pink
    [200, 200, 100]   # Others - yellow
], dtype=np.uint8)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


class LiveSegmentation:
    def __init__(self, model_path=None, blur_background=True, background_color=(192, 192, 192), use_gpu=None):
        self.blur_background = blur_background
        self.background_color = np.array(background_color, dtype=np.uint8)
        self.latest_mask = None
        self.latest_face_landmarks = None
        self.lock = threading.Lock()
        self.face_lock = threading.Lock()
        self.use_gpu = use_gpu if use_gpu is not None else True
        self.use_srgba = self.use_gpu
        self.enable_face_landmarks = True
        
        # Error tracking for automatic GPU fallback
        self.gpu_error_count = 0
        self.max_gpu_errors = 10  # Switch to CPU after this many consecutive errors
        self.error_lock = threading.Lock()
        
        # Configurable parameters
        self.blend_factor = 0.3
        self.blur_kernel_size = 35
        self.use_textures = True  # Start with textures ON
        self.keep_background_original = True  # Keep background as original (no effects) - toggle with 'b'
        self.textures = {}  # Store textures for each category: {0: texture_img, 1: texture_img, ...}
        self.texture_scales = {}  # Random scale for each category
        self.texture_folders = ["textures/GANSTILLFINAL", "textures/tiles2"]  # Folders with texture PNGs (inside textures folder)
        self.texture_files = []  # List of available texture files
        self._tiled_cache = OrderedDict()  # Cache tiled textures: {(cat_id, scale): tiled_texture} - FIFO eviction
        self._last_frame_size = None  # Track frame size changes
        
        # Pre-allocated buffers to reduce memory allocations
        self._output_buffer = None  # Reusable output buffer
        self._alpha_channel = None  # Reusable alpha channel for RGBA
        self._cached_mask = None  # Cached resized mask
        self._cached_mask_size = None  # Size of cached mask
        self._landmark_points = np.zeros((478, 2), dtype=np.int32)  # Pre-allocated landmark points
        
        # Frame skipping to prevent queue buildup
        self._last_processed_timestamp = 0
        self._processing_queue_size = 0  # Track approximate queue size
        self._max_queue_size = 2  # Skip frames if queue gets too large (reduced from 3)
        self._frames_skipped = 0  # Track skipped frames for monitoring
        self._last_gc_time = time.time()  # Track last GC time
        
        # CRITICAL: Track pending async operations to prevent unbounded queue
        # MediaPipe's async queue is in C++ and Python GC can't free it
        self._pending_segments = 0  # Count of pending segment operations
        self._max_pending = 2  # Maximum pending operations before blocking
        self._pending_lock = threading.Lock()  # Lock for pending counter
        
        # CRITICAL: Track segmenter recreation to force memory release
        # MediaPipe's C++ code holds memory that Python GC can't free
        # Recreating the segmenter forces MediaPipe to release all internal buffers
        self._segmenter_frame_count = 0  # Frames processed by current segmenter
        self._max_segmenter_frames = 500  # Recreate segmenter every 500 frames (~50 seconds at 10 FPS) to force memory release
        self._model_path = None  # Store model path for recreation
        
        # Face landmark color schemes (BGR format for OpenCV)
        # Each scheme has: landmark_color, connection_color, key_point_color
        self.landmark_color_schemes = [
            # Bright cyan/yellow/white - high contrast
            ((255, 255, 0), (0, 255, 255), (255, 255, 255)),
            # Bright green/magenta/white
            ((0, 255, 0), (255, 0, 255), (255, 255, 255)),
            # Bright red/cyan/white
            ((0, 0, 255), (255, 255, 0), (255, 255, 255)),
            # Bright yellow/blue/white
            ((0, 255, 255), (255, 0, 0), (255, 255, 255)),
            # Bright magenta/green/white
            ((255, 0, 255), (0, 255, 0), (255, 255, 255)),
            # Bright orange/cyan/white
            ((0, 165, 255), (255, 255, 0), (255, 255, 255)),
            # Bright lime/pink/white
            ((0, 255, 127), (203, 192, 255), (255, 255, 255)),
            # Bright purple/yellow/white
            ((128, 0, 128), (0, 255, 255), (255, 255, 255)),
        ]
        self.current_landmark_colors = self.landmark_color_schemes[0]  # Default to first scheme
        
        if model_path is None:
            model_path = self._download_model()
        
        self._model_path = model_path  # Store for recreation
        self.segmenter = self._create_segmenter(model_path)
        
        # Initialize face landmarker
        if self.enable_face_landmarks:
            self.face_landmarker = self._create_face_landmarker()
            self._face_landmarker_frame_count = 0  # Track frames for face landmarker too
        else:
            self.face_landmarker = None
            self._face_landmarker_frame_count = 0
        
        # Load texture file list
        self._load_texture_file_list()
    
    def _load_texture_file_list(self):
        """Load list of available texture files from all texture folders."""
        self.texture_files = []
        for folder in self.texture_folders:
            if not os.path.exists(folder):
                print(f"⚠️ Texture folder not found: {folder}")
                continue
            
            # Search for PNG files
            texture_path = os.path.join(folder, "*.png")
            files = glob.glob(texture_path)
            if not files:
                # Try case-insensitive
                texture_path = os.path.join(folder, "*.PNG")
                files = glob.glob(texture_path)
            
            self.texture_files.extend(files)
            print(f"📁 Found {len(files)} texture files in {folder}")
        
        print(f"✅ Total: {len(self.texture_files)} texture files available")
    
    def randomize_textures(self):
        """Randomly load textures for all categories from texture folders."""
        if not self.texture_files:
            print("❌ No texture files found in texture folders")
            return False
        
        # Explicitly free old textures to prevent memory accumulation
        for texture in self.textures.values():
            del texture
        self.textures.clear()
        self.texture_scales.clear()
        self._tiled_cache.clear()  # Clear cache when textures change
        
        # Randomize face landmark colors too
        color_names = ["Cyan/Yellow", "Green/Magenta", "Red/Cyan", "Yellow/Blue", 
                      "Magenta/Green", "Orange/Cyan", "Lime/Pink", "Purple/Yellow"]
        scheme_idx = random.randint(0, len(self.landmark_color_schemes) - 1)
        self.current_landmark_colors = self.landmark_color_schemes[scheme_idx]
        print(f"🎨 Face landmark colors: {color_names[scheme_idx]}")
        
        # Random scale range: 0.5x to 2.0x (for 32x32, that's 16x16 to 64x64)
        min_scale = 0.5
        max_scale = 1.25
        
        for cat_id in range(6):  # 0-5 categories
            # Pick random texture file
            texture_path = random.choice(self.texture_files)
            
            # Random scale for this category
            random_scale = random.uniform(min_scale, max_scale)
            self.texture_scales[cat_id] = random_scale
            
            # Load texture
            if self.load_texture(cat_id, texture_path):
                print(f"  Category {cat_id}: {os.path.basename(texture_path)} (scale: {random_scale:.2f}x)")
        
        print(f"✅ Randomized {len(self.textures)} textures")
        return True
    
    def load_texture(self, category_id, texture_path):
        """Load a texture image for a specific category."""
        if not os.path.exists(texture_path):
            print(f"❌ Texture file not found: {texture_path}")
            return False
        
        try:
            # Load texture with alpha channel if available
            texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
            if texture is None:
                print(f"❌ Failed to load texture: {texture_path}")
                return False
            
            # Debug: check original format
            original_channels = texture.shape[2] if len(texture.shape) == 3 else 1
            print(f"📊 Texture loaded: {texture.shape}, channels: {original_channels}")
            
            # Convert BGRA to RGBA if has alpha, otherwise BGR to RGB
            if texture.shape[2] == 4:
                texture = cv2.cvtColor(texture, cv2.COLOR_BGRA2RGBA)
                print(f"✅ Texture has alpha channel (RGBA)")
            elif texture.shape[2] == 3:
                texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                print(f"⚠️ Texture has no alpha channel (RGB)")
            else:
                print(f"⚠️ Unexpected channel count: {texture.shape[2]}")
                return False
            
            self.textures[category_id] = texture
            print(f"✅ Loaded texture for category {category_id}: {texture_path} ({texture.shape[1]}x{texture.shape[0]})")
            return True
        except Exception as e:
            print(f"❌ Error loading texture: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def remove_texture(self, category_id):
        """Remove texture for a category."""
        if category_id in self.textures:
            del self.textures[category_id]
            print(f"✅ Removed texture for category {category_id}")
    
    def _apply_texture_to_mask(self, texture, mask_region, target_shape, category_id=None):
        """Apply texture to a mask region - HIGHLY OPTIMIZED: only processes masked pixels."""
        h, w = target_shape[:2]
        tex_h, tex_w = texture.shape[:2]
        has_alpha = texture.shape[2] == 4
        
        # Get scale for this category
        if category_id is not None and category_id in self.texture_scales:
            scale = self.texture_scales[category_id]
        else:
            scale = 1.0
        
        # Scale texture once (cache the scaled texture, not the full tiled version)
        scaled_w = max(1, int(tex_w * scale))
        scaled_h = max(1, int(tex_h * scale))
        
        cache_key = (category_id, scale)
        if cache_key in self._tiled_cache:
            texture_scaled = self._tiled_cache[cache_key]
        else:
            if scaled_w != tex_w or scaled_h != tex_h:
                texture_scaled = cv2.resize(texture, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            else:
                texture_scaled = texture.copy()
            
            # Cache only the scaled texture (much smaller than full tiled)
            self._tiled_cache[cache_key] = texture_scaled
            # Limit cache size with proper FIFO eviction
            if len(self._tiled_cache) > 12:
                self._tiled_cache.popitem(last=False)  # Remove oldest entry (FIFO)
        
        tex_h_scaled, tex_w_scaled = texture_scaled.shape[:2]
        
        # CRITICAL OPTIMIZATION: Only process pixels where mask is True
        # Get coordinates of all masked pixels
        y_coords, x_coords = np.where(mask_region)
        
        if len(y_coords) == 0:
            # No masked pixels, return empty
            result = np.zeros((h, w, 3), dtype=np.uint8)
            return result, None if not has_alpha else np.zeros((h, w, 1), dtype=np.float32)
        
        # Tile texture coordinates using modulo (no full-screen tiling needed!)
        tex_y = y_coords % tex_h_scaled
        tex_x = x_coords % tex_w_scaled
        
        # Extract texture values only for masked pixels
        if has_alpha:
            rgb = texture_scaled[tex_y, tex_x, :3]  # Shape: (N, 3)
            alpha = texture_scaled[tex_y, tex_x, 3:4]  # Shape: (N, 1)
            alpha_float = alpha.astype(np.float32) / 255.0
        else:
            rgb = texture_scaled[tex_y, tex_x, :3]
            alpha_float = None
        
        # Create result arrays
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[y_coords, x_coords] = rgb
        
        if has_alpha:
            alpha_result = np.zeros((h, w, 1), dtype=np.float32)
            alpha_result[y_coords, x_coords] = alpha_float
            return result, alpha_result
        else:
            return result, None
        
    def _download_model(self):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "selfie_multiclass_256x256.tflite")
        
        if not os.path.exists(model_path):
            print(f"Downloading model to {model_path}...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
        
        return model_path
    
    def _create_segmenter(self, model_path):
        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        def result_callback(result, output_image, timestamp_ms):
            try:
                # CRITICAL: Decrement pending counter when callback executes
                # This means MediaPipe has finished processing and released the frame
                with self._pending_lock:
                    if self._pending_segments > 0:
                        self._pending_segments -= 1
                
                if hasattr(result, 'category_mask') and result.category_mask is not None:
                    # CRITICAL: Copy the numpy view to break MediaPipe's reference
                    # This prevents MediaPipe from holding onto old mask data
                    mask_view = result.category_mask.numpy_view()
                    mask = mask_view.copy()  # Create independent copy
                    del mask_view  # Release view reference immediately
                    
                    with self.lock:
                        # Clear old mask reference before assigning new one
                        old_mask = self.latest_mask
                        self.latest_mask = mask
                        del old_mask  # Explicitly delete old reference
            except Exception as e:
                # Silently handle callback errors - MediaPipe may send invalid data
                # Don't crash the application on callback errors
                # Still decrement counter even on error
                with self._pending_lock:
                    if self._pending_segments > 0:
                        self._pending_segments -= 1
                pass
        
        base_options = BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.GPU if self.use_gpu else BaseOptions.Delegate.CPU
        )
        
        options = ImageSegmenterOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.LIVE_STREAM,
            output_category_mask=True,
            result_callback=result_callback
        )
        
        return ImageSegmenter.create_from_options(options)
    
    def _create_face_landmarker(self):
        """Create face landmarker for face detection."""
        try:
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            BaseOptions = mp.tasks.BaseOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            # Download face landmarker model
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            face_model_path = os.path.join(model_dir, "face_landmarker.task")
            
            if not os.path.exists(face_model_path):
                print("Downloading Face Landmarker model...")
                urllib.request.urlretrieve(FACE_LANDMARKER_URL, face_model_path)
                print(f"✅ Face model downloaded")
            
            def face_result_callback(result, output_image, timestamp_ms):
                try:
                    if result and result.face_landmarks:
                        with self.face_lock:
                            # Clear old landmark reference before assigning new one
                            old_landmarks = self.latest_face_landmarks
                            self.latest_face_landmarks = result
                            del old_landmarks  # Explicitly delete old reference
                except Exception as e:
                    # Silently handle face landmark callback errors
                    # Don't crash the application on callback errors
                    pass
            
            base_options = BaseOptions(
                model_asset_path=face_model_path,
                delegate=BaseOptions.Delegate.GPU if self.use_gpu else BaseOptions.Delegate.CPU
            )
            
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_faces=5,  # Support up to 5 faces
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                result_callback=face_result_callback
            )
            
            return FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"⚠️ Face landmarker initialization failed: {e}")
            return None
    
    def process_frame(self, frame, timestamp_ms, processing_size=512):
        """Process frame at higher resolution for better hair detection."""
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > processing_size:
            scale = processing_size / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame
        
        # Skip contrast enhancement for speed (can be re-enabled if needed)
        # if processing_size >= 512:
        #     lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
        #     l, a, b = cv2.split(lab)
        #     l = cv2.convertScaleAbs(l, alpha=1.1, beta=5)
        #     frame_resized = cv2.merge([l, a, b])
        #     frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_LAB2BGR)
        
        # CRITICAL: Periodically recreate segmenter to force MediaPipe to release C++ memory
        # MediaPipe's C++ code holds memory that Python GC can't free
        # Recreating the segmenter forces MediaPipe to release all internal buffers
        self._segmenter_frame_count += 1
        if self._segmenter_frame_count >= self._max_segmenter_frames:
            print(f"🔄 Recreating segmenter to force memory release (processed {self._segmenter_frame_count} frames)...")
            
            # Wait for pending operations to complete
            max_wait = 50  # Wait up to 50 iterations
            wait_count = 0
            while wait_count < max_wait:
                with self._pending_lock:
                    pending = self._pending_segments
                if pending == 0:
                    break
                time.sleep(0.01)  # Wait 10ms
                wait_count += 1
            
            # Close old segmenter
            try:
                if self.segmenter:
                    self.segmenter.close()
            except:
                pass
            
            # Force GC to collect Python-side objects
            gc.collect()
            
            # Recreate segmenter - this forces MediaPipe to release C++ memory
            self.segmenter = self._create_segmenter(self._model_path)
            self._segmenter_frame_count = 0
            print("✅ Segmenter recreated")
        
        # CRITICAL: Check MediaPipe's async queue size before processing
        # MediaPipe queues frames in C++ memory - Python GC can't free them!
        with self._pending_lock:
            pending = self._pending_segments
        
        # If too many frames are pending, skip this frame to prevent queue buildup
        if pending >= self._max_pending:
            self._frames_skipped += 1
            # If we've skipped many frames, wait a bit for queue to drain
            if self._frames_skipped > 5:
                time.sleep(0.01)  # Small sleep to let queue drain
            return
        
        # AGGRESSIVE Frame skipping: Don't process if queue is too full (prevents memory buildup)
        time_since_last = timestamp_ms - self._last_processed_timestamp
        
        # More aggressive skipping - only process every 33ms (~30 FPS max) instead of 16ms
        # This prevents MediaPipe's internal queue from growing unbounded
        if time_since_last < 33:  # ~30 FPS max processing rate (was 16ms/60 FPS)
            self._frames_skipped += 1
            return
        
        # If we've skipped many frames, force GC to clean up
        if self._frames_skipped > 10:
            current_time = time.time()
            if current_time - self._last_gc_time > 1.0:  # GC at most once per second
                gc.collect()
                self._last_gc_time = current_time
            self._frames_skipped = 0
        
        # CRITICAL: Copy frame data to break any references MediaPipe might hold
        # MediaPipe's async queue holds references to the numpy arrays
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).copy()  # Explicit copy
        
        # Reuse alpha channel buffer to reduce allocations
        if self.use_srgba:
            h, w = rgb_frame.shape[:2]
            if self._alpha_channel is None or self._alpha_channel.shape != (h, w):
                self._alpha_channel = np.full((h, w), 255, dtype=np.uint8)
            # Create RGBA frame - copy to ensure MediaPipe doesn't hold reference to our buffers
            rgba_frame = np.dstack([rgb_frame, self._alpha_channel]).copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)
        else:
            # Already copied above, but ensure we're passing a copy
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            # CRITICAL: Increment pending counter BEFORE async call
            # This tracks how many frames MediaPipe is holding in its C++ queue
            with self._pending_lock:
                self._pending_segments += 1
            
            self.segmenter.segment_async(mp_image, timestamp_ms)
            self._last_processed_timestamp = timestamp_ms
            
            # Process face landmarks in parallel (same frame)
            if self.face_landmarker:
                # Also recreate face landmarker periodically
                self._face_landmarker_frame_count += 1
                if self._face_landmarker_frame_count >= self._max_segmenter_frames:
                    print(f"🔄 Recreating face landmarker to force memory release...")
                    try:
                        if self.face_landmarker:
                            self.face_landmarker.close()
                    except:
                        pass
                    gc.collect()
                    self.face_landmarker = self._create_face_landmarker()
                    self._face_landmarker_frame_count = 0
                    print("✅ Face landmarker recreated")
                
                try:
                    self.face_landmarker.detect_async(mp_image, timestamp_ms)
                except Exception as e:
                    # Silently handle face landmark errors (non-critical)
                    pass
            
            # Explicitly release MediaPipe image references to help GC
            # CRITICAL: This helps break references so GC can collect them
            del mp_image
            if self.use_srgba:
                del rgba_frame
            del rgb_frame
            del frame_resized
            
            # Reset error count on successful processing
            with self.error_lock:
                if self.gpu_error_count > 0:
                    self.gpu_error_count = 0
                    
        except Exception as e:
            # Handle MediaPipe errors gracefully
            with self.error_lock:
                self.gpu_error_count += 1
                
                # If too many GPU errors, suggest CPU fallback
                if self.use_gpu and self.gpu_error_count >= self.max_gpu_errors:
                    print(f"\n⚠️ Multiple GPU errors detected ({self.gpu_error_count}). Consider restarting with CPU mode.")
                    print("   The application will continue running but may have reduced performance.")
                    self.gpu_error_count = 0  # Reset to avoid spam
            # Continue running - don't crash on errors
    
    # Removed get_segmented_frame - only using mask-only mode for speed
    
    def get_mask_only(self, original_frame, show_categories=True):
        """Get only the segmentation mask with textures - highly optimized."""
        # Fast lock check
        with self.lock:
            if self.latest_mask is None:
                return np.zeros_like(original_frame)
            # No copy needed - lock protects access and we're not modifying mask
            mask = self.latest_mask
        
        h, w = original_frame.shape[:2]
        
        # Cache resized mask to avoid repeated resizing
        if mask.shape[:2] != (h, w):
            # Check if we can reuse cached mask
            if self._cached_mask is None or self._cached_mask_size != (h, w) or self._cached_mask.shape[:2] != mask.shape[:2]:
                # Resize and cache
                self._cached_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                self._cached_mask_size = (h, w)
            mask = self._cached_mask
        
        # Clear cache if frame size changed
        if self._last_frame_size != (h, w):
            self._tiled_cache.clear()  # Clear cache when frame size changes
            self._cached_mask = None  # Invalidate cached mask
            self._cached_mask_size = None
            self._last_frame_size = (h, w)
        
        if show_categories and self.use_textures and self.textures:
            # ULTRA-OPTIMIZED: Use direct mask indexing instead of np.where
            if self.keep_background_original:
                # Start with original frame (background will remain unchanged)
                output = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB).copy()
                # Only initialize with category colors for categories without textures
                # Exclude category 0 (background) - keep original frame for background
                categories_with_textures = set(self.textures.keys())
                categories_without_textures = set(range(1, 6)) - categories_with_textures  # Exclude category 0
            else:
                # Start with zeros (apply effects to all categories including background)
                if self._output_buffer is None or self._output_buffer.shape != (h, w, 3):
                    self._output_buffer = np.zeros((h, w, 3), dtype=np.uint8)
                output = self._output_buffer
                # Only initialize with category colors for categories without textures
                categories_with_textures = set(self.textures.keys())
                categories_without_textures = set(range(6)) - categories_with_textures  # Include category 0
                
            if categories_without_textures:
                # Only set colors for categories that don't have textures
                for cat_id in categories_without_textures:
                    cat_mask = (mask == cat_id)
                    if np.any(cat_mask):
                        output[cat_mask] = CATEGORY_COLORS[cat_id]
            
            # Process categories that have textures loaded
            for cat_id in self.textures:
                # Skip category 0 (background) if keep_background_original is True
                if self.keep_background_original and cat_id == 0:
                    continue
                cat_mask = (mask == cat_id)
                if not np.any(cat_mask):
                    continue
                
                # Get texture for this category
                texture = self.textures[cat_id]
                
                # Get scale
                scale = self.texture_scales.get(cat_id, 1.0)
                
                # Get scaled texture from cache
                cache_key = (cat_id, scale)
                if cache_key in self._tiled_cache:
                    texture_scaled = self._tiled_cache[cache_key]
                else:
                    tex_h, tex_w = texture.shape[:2]
                    scaled_w = max(1, int(tex_w * scale))
                    scaled_h = max(1, int(tex_h * scale))
                    if scaled_w != tex_w or scaled_h != tex_h:
                        texture_scaled = cv2.resize(texture, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        texture_scaled = texture.copy()
                    self._tiled_cache[cache_key] = texture_scaled
                    if len(self._tiled_cache) > 12:
                        self._tiled_cache.popitem(last=False)  # Remove oldest entry (FIFO)
                
                tex_h_scaled, tex_w_scaled = texture_scaled.shape[:2]
                
                # Get mask coordinates ONCE (faster than np.where in loop)
                y_coords, x_coords = np.where(cat_mask)
                num_pixels = len(y_coords)
                
                if num_pixels == 0:
                    continue
                
                # Tile coordinates
                tex_y = y_coords % tex_h_scaled
                tex_x = x_coords % tex_w_scaled
                
                # Extract texture values directly - NO BLENDING (textures are opaque)
                # Direct assignment (fastest path - no alpha checking or blending)
                output[y_coords, x_coords] = texture_scaled[tex_y, tex_x, :3]
        elif show_categories:
            if self.keep_background_original:
                # Start with original frame (background will remain unchanged)
                output = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB).copy()
                # Apply colors only to human categories (1-5), skip background (0)
                human_mask = (mask > 0) & (mask < 6)
                if np.any(human_mask):
                    output[human_mask] = CATEGORY_COLORS[mask[human_mask]]
            else:
                # Apply colors to all categories including background
                output = CATEGORY_COLORS[mask]
        else:
            # Binary mask
            if self.keep_background_original:
                # Keep background as original frame
                output = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB).copy()
                person_mask = mask > 0
                if np.any(person_mask):
                    output[person_mask] = [255, 255, 255]
            else:
                # Binary mask with black background
                person_mask = mask > 0
                output = np.where(person_mask[:, :, np.newaxis], [255, 255, 255], [0, 0, 0]).astype(np.uint8)
        
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    def draw_face_landmarks(self, frame):
        """Draw face landmarks as a mesh with connections for all detected faces."""
        # Fast lock check
        try:
            with self.face_lock:
                if self.latest_face_landmarks is None or not self.latest_face_landmarks.face_landmarks:
                    return frame
                all_face_landmarks = self.latest_face_landmarks.face_landmarks
        except:
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw landmarks for each detected face
        for face_idx, face_landmarks in enumerate(all_face_landmarks):
            if len(face_landmarks) < 478:
                continue  # Skip invalid landmark data
            
            # Pre-compute all landmark points (reuse pre-allocated array)
            num_landmarks = len(face_landmarks)
            if num_landmarks > len(self._landmark_points):
                # Resize if needed (shouldn't happen, but safe)
                self._landmark_points = np.zeros((num_landmarks, 2), dtype=np.int32)
            
            # Update in-place to avoid allocation
            for i, lm in enumerate(face_landmarks):
                self._landmark_points[i, 0] = int(lm.x * w)
                self._landmark_points[i, 1] = int(lm.y * h)
            points = self._landmark_points[:num_landmarks]
            
            # Use different color for each face to distinguish them
            # Cycle through color schemes based on face index
            color_scheme_idx = face_idx % len(self.landmark_color_schemes)
            LANDMARK_COLOR, CONNECTION_COLOR, KEY_POINT_COLOR = self.landmark_color_schemes[color_scheme_idx]
            
            # Only draw eyes, nose, and mouth mesh
            
            # Nose bridge (27-30)
            for i in range(27, 30):
                pt1, pt2 = tuple(points[i]), tuple(points[i + 1])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
            
            # Nose tip (30-35)
            for i in range(30, 35):
                pt1, pt2 = tuple(points[i]), tuple(points[i + 1])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
            
            # Left eye - connect in order
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            left_eye_valid = [i for i in left_eye_indices if i < len(points)]
            for i in range(len(left_eye_valid) - 1):
                pt1, pt2 = tuple(points[left_eye_valid[i]]), tuple(points[left_eye_valid[i + 1]])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
            if len(left_eye_valid) > 2:
                # Close the eye loop
                pt1, pt2 = tuple(points[left_eye_valid[-1]]), tuple(points[left_eye_valid[0]])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
            
            # Right eye - connect in order
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            right_eye_valid = [i for i in right_eye_indices if i < len(points)]
            for i in range(len(right_eye_valid) - 1):
                pt1, pt2 = tuple(points[right_eye_valid[i]]), tuple(points[right_eye_valid[i + 1]])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
            if len(right_eye_valid) > 2:
                # Close the eye loop
                pt1, pt2 = tuple(points[right_eye_valid[-1]]), tuple(points[right_eye_valid[0]])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
            
            # Mouth outer
            mouth_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
            mouth_outer_valid = [i for i in mouth_outer_indices if i < len(points)]
            for i in range(len(mouth_outer_valid) - 1):
                pt1, pt2 = tuple(points[mouth_outer_valid[i]]), tuple(points[mouth_outer_valid[i + 1]])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
            if len(mouth_outer_valid) > 2:
                # Close the mouth loop
                pt1, pt2 = tuple(points[mouth_outer_valid[-1]]), tuple(points[mouth_outer_valid[0]])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
            
            # Mouth inner
            mouth_inner_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
            mouth_inner_valid = [i for i in mouth_inner_indices if i < len(points)]
            for i in range(len(mouth_inner_valid) - 1):
                pt1, pt2 = tuple(points[mouth_inner_valid[i]]), tuple(points[mouth_inner_valid[i + 1]])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
            if len(mouth_inner_valid) > 2:
                # Close the inner mouth loop
                pt1, pt2 = tuple(points[mouth_inner_valid[-1]]), tuple(points[mouth_inner_valid[0]])
                cv2.line(frame, pt1, pt2, CONNECTION_COLOR, 1)
        
        return frame
    
    def overlay_logo(self, frame, logo_path="glixLogo.jpeg", max_height=80, opacity=1, bg_opacity=1):
        """Overlay logo at the top right of the frame with semi-transparent background."""
        try:
            if not os.path.exists(logo_path):
                return frame
            
            # Load logo
            logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo is None:
                return frame
            
            h, w = frame.shape[:2]
            logo_h, logo_w = logo.shape[:2]
            
            # Calculate scale to fit max_height while maintaining aspect ratio
            scale = min(max_height / logo_h, (w * 0.3) / logo_w)  # Max 30% of frame width
            new_logo_w = int(logo_w * scale)
            new_logo_h = int(logo_h * scale)
            
            # Resize logo
            if logo.shape[2] == 4:  # Has alpha channel
                logo_resized = cv2.resize(logo, (new_logo_w, new_logo_h), interpolation=cv2.INTER_AREA)
            else:
                logo_resized = cv2.resize(logo, (new_logo_w, new_logo_h), interpolation=cv2.INTER_AREA)
                # Add alpha channel if missing
                logo_resized = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2BGRA)
                logo_resized[:, :, 3] = 255  # Full opacity
            
            # Calculate position (top right)
            x_offset = w - new_logo_w - 20  # 20 pixels from right edge
            y_offset = 20  # 20 pixels from top
            
            # Get the region where logo will be placed
            y1, y2 = y_offset, y_offset + new_logo_h
            x1, x2 = x_offset, x_offset + new_logo_w
            
            # Make sure we don't go out of bounds
            if y2 > h or x2 > w or x1 < 0 or y1 < 0:
                return frame
            
            # Add semi-transparent background behind logo
            padding = 4  # Padding around logo (reduced for thinner borders)
            bg_y1 = max(0, y1 - padding)
            bg_y2 = min(h, y2 + padding)
            bg_x1 = max(0, x1 - padding)
            bg_x2 = min(w, x2 + padding)
            
            # Create semi-transparent black background
            bg_alpha = bg_opacity
            frame_bg_region = frame[bg_y1:bg_y2, bg_x1:bg_x2].copy()
            bg_overlay = np.zeros_like(frame_bg_region)
            bg_blended = (bg_alpha * bg_overlay + (1 - bg_alpha) * frame_bg_region).astype(np.uint8)
            frame[bg_y1:bg_y2, bg_x1:bg_x2] = bg_blended
            
            # Extract alpha channel and normalize for logo
            alpha = logo_resized[:, :, 3].astype(np.float32) / 255.0 * opacity
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            
            # Blend logo with frame (over the background)
            logo_bgr = logo_resized[:, :, :3]
            frame_region = frame[y1:y2, x1:x2]
            blended = (alpha_3d * logo_bgr + (1 - alpha_3d) * frame_region).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended
            
            return frame
        except Exception as e:
            # Silently fail if logo overlay fails
            return frame
    
    def close(self):
        if self.segmenter:
            self.segmenter.close()
        if self.face_landmarker:
            self.face_landmarker.close()


class ControlPanel:
    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.window_name = "Control Panel"
        self.width = 350
        self.height = 600
        self.panel_img = None
        
        # Button positions and states (adjusted for 350px width)
        # Initial state: textures ON, blur OFF
        self.buttons = {
            'use_textures': {'x': 20, 'y': 300, 'w': 150, 'h': 35, 'state': True, 'label': 'Textures'},
            'face_landmarks': {'x': 180, 'y': 300, 'w': 150, 'h': 35, 'state': True, 'label': 'Face Landmarks'},
        }
        
        # Apply initial states
        segmenter.use_textures = True
        segmenter.enable_face_landmarks = True
        
        # Randomize textures button
        self.randomize_button = {'x': 20, 'y': 390, 'w': 310, 'h': 40, 'label': 'Randomize Textures'}
        
        # Trackbar values (stored for display)
        self.processing_size = 512
        self.cam_width = 720
        self.cam_height = 480
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        
        # Trackbar callbacks
        def on_processing_size(val):
            self.processing_size = max(256, val)
            self.update_display()
        
        def on_cam_width(val):
            # Camera resolution will be updated in main loop
            pass
        
        def on_cam_height(val):
            # Camera resolution will be updated in main loop
            pass
        
        # Create trackbars with short names
        cv2.createTrackbar("Proc Size", self.window_name, 512, 640, on_processing_size)
        cv2.createTrackbar("Cam Width", self.window_name, 720, 1280, on_cam_width)
        cv2.createTrackbar("Cam Height", self.window_name, 480, 720, on_cam_height)
        
        # Mouse callback for buttons
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self.update_display()
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check randomize button first
            btn = self.randomize_button
            if (btn['x'] <= x <= btn['x'] + btn['w'] and 
                btn['y'] <= y <= btn['y'] + btn['h']):
                print("\n🎲 Randomizing textures...")
                self.segmenter.randomize_textures()
                self.update_display()
                return
            
            # Check regular buttons
            for key, btn in self.buttons.items():
                if (btn['x'] <= x <= btn['x'] + btn['w'] and 
                    btn['y'] <= y <= btn['y'] + btn['h']):
                    btn['state'] = not btn['state']
                    self._apply_button_state(key, btn['state'])
                    self.update_display()
                    break
    
    def _apply_button_state(self, key, state):
        if key == 'use_textures':
            self.segmenter.use_textures = state
        elif key == 'face_landmarks':
            self.segmenter.enable_face_landmarks = state
    
    def update_display(self):
        """Update the control panel display."""
        self.panel_img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 40
        
        # Title
        cv2.putText(self.panel_img, "Control Panel", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Trackbar labels
        y_pos = 70
        cv2.putText(self.panel_img, "Processing Size:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(self.panel_img, f"{self.processing_size}px", (200, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        y_pos += 40
        cv2.putText(self.panel_img, "Cam Width:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(self.panel_img, f"{self.cam_width}px", (200, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        y_pos += 40
        cv2.putText(self.panel_img, "Cam Height:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(self.panel_img, f"{self.cam_height}px", (200, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Buttons
        y_pos = 280
        cv2.putText(self.panel_img, "Toggle Buttons:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for key, btn in self.buttons.items():
            color = (50, 200, 50) if btn['state'] else (50, 50, 200)
            text_color = (255, 255, 255) if btn['state'] else (200, 200, 200)
            
            # Draw button
            cv2.rectangle(self.panel_img,
                         (btn['x'], btn['y']),
                         (btn['x'] + btn['w'], btn['y'] + btn['h']),
                         color, -1)
            cv2.rectangle(self.panel_img,
                         (btn['x'], btn['y']),
                         (btn['x'] + btn['w'], btn['y'] + btn['h']),
                         (255, 255, 255), 2)
            
            # Button text
            text_size = cv2.getTextSize(btn['label'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = btn['x'] + (btn['w'] - text_size[0]) // 2
            text_y = btn['y'] + (btn['h'] + text_size[1]) // 2
            cv2.putText(self.panel_img, btn['label'], (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Randomize textures button
        btn = self.randomize_button
        has_textures = len(self.segmenter.textures) > 0
        color = (50, 200, 50) if has_textures else (80, 80, 200)
        text_color = (255, 255, 255)
        
        # Draw button
        cv2.rectangle(self.panel_img,
                     (btn['x'], btn['y']),
                     (btn['x'] + btn['w'], btn['y'] + btn['h']),
                     color, -1)
        cv2.rectangle(self.panel_img,
                     (btn['x'], btn['y']),
                     (btn['x'] + btn['w'], btn['y'] + btn['h']),
                     (255, 255, 255), 2)
        
        # Button text
        text_size = cv2.getTextSize(btn['label'], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = btn['x'] + (btn['w'] - text_size[0]) // 2
        text_y = btn['y'] + (btn['h'] + text_size[1]) // 2
        cv2.putText(self.panel_img, btn['label'], (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Status indicator
        y_pos = 450
        cv2.putText(self.panel_img, "Status:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        gpu_text = "GPU" if self.segmenter.use_gpu else "CPU"
        cv2.putText(self.panel_img, f"Mode: {gpu_text}", (10, y_pos + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Texture status
        texture_count = len(self.segmenter.textures)
        if texture_count > 0:
            cv2.putText(self.panel_img, f"Textures: {texture_count}/6 loaded", (10, y_pos + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            # Show scale info
            if self.segmenter.texture_scales:
                scales_str = ", ".join([f"{s:.1f}x" for s in list(self.segmenter.texture_scales.values())[:3]])
                cv2.putText(self.panel_img, f"Scales: {scales_str}...", (10, y_pos + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 255), 1)
        
        # Instructions
        y_pos = 520
        cv2.putText(self.panel_img, "Click 'Randomize Textures'", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(self.panel_img, "to load random textures", (10, y_pos + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        cv2.imshow(self.window_name, self.panel_img)
    
    def get_processing_size(self):
        return max(256, cv2.getTrackbarPos("Proc Size", self.window_name))
    
    
    def get_cam_width(self):
        return cv2.getTrackbarPos("Cam Width", self.window_name)
    
    def get_cam_height(self):
        return cv2.getTrackbarPos("Cam Height", self.window_name)
    
    def set_cam_resolution(self, width, height):
        """Update camera resolution display values."""
        self.cam_width = width
        self.cam_height = height
        cv2.setTrackbarPos("Cam Width", self.window_name, width)
        cv2.setTrackbarPos("Cam Height", self.window_name, height)
        self.update_display()


def main():
    print("MediaPipe Live Segmentation")
    print("=" * 50)
    print("Controls:")
    print("  'q' - Quit")
    print("  'r' - Randomize textures")
    print("  't' - Toggle text display")
    print("  'p' - Cycle processing size (256-640px)")
    print("  'x' - Toggle textures on/off")
    print("  'b' - Toggle background effects (keep original background)")
    print("  'f' - Toggle fullscreen")
    print("  's' - Save current frame")
    print("=" * 50 + "\n")
    
    try:
        segmenter = LiveSegmentation(blur_background=False, use_gpu=True)
        print("✅ GPU enabled")
    except:
        segmenter = LiveSegmentation(blur_background=False, use_gpu=False)
        segmenter.use_srgba = False
        print("✅ CPU mode")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera error")
        segmenter.close()
        return
    
    # CRITICAL: Reduce OpenCV buffer size to prevent frame accumulation
    # This prevents OpenCV from buffering too many frames in memory
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer - only keep latest frame
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera: {width}x{height}\n")
    
    # Control panel disabled - using hotkeys only
    # control_panel = ControlPanel(segmenter)
    # control_panel.set_cam_resolution(width, height)
    
    # Processing size settings - cycle through with hotkey
    processing_sizes = [256, 320, 384, 448, 512, 576, 640]
    processing_size_idx = 6  # Start at 384 (index 2)
    processing_size = processing_sizes[processing_size_idx]
    
    frame_count = 0
    start_time = time.time()
    process_every_n = 1 if segmenter.use_gpu else 2
    
    # Automatic texture randomization and capture timers
    randomize_interval = 20.0  # Randomize every 20 seconds
    capture_delay = 5.0  # First capture 5 seconds after randomize
    capture_interval = 10.0  # Then capture every 10 seconds
    last_capture_time = 0  # Track last capture time
    textures_need_toggle_back = False  # Track if textures need to be toggled back on
    
    # Auto-randomize textures on startup (non-blocking)
    print("\n🎲 Auto-loading random textures...")
    segmenter.randomize_textures()
    last_randomize_time = time.time()  # Set after initial randomize
    last_capture_time = last_randomize_time  # Initialize capture time
    
    # Warmup: skip first few frames to let camera stabilize
    warmup_frames = 5
    
    # Garbage collection settings - MORE AGGRESSIVE
    gc_frequency = 30  # Run GC every 30 frames (was 60) - more frequent
    gc_threshold = (500, 5, 5)  # Even more aggressive GC thresholds (was 700, 10, 10)
    gc.set_threshold(*gc_threshold)
    
    # CRITICAL: Enable tracemalloc to track actual memory allocations
    # This will show us what Python objects are actually leaking
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    
    # Memory monitoring (optional - graceful fallback if psutil not available)
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_monitoring = True
    except ImportError:
        print("⚠️ psutil not installed - memory monitoring disabled")
        memory_monitoring = False
        initial_memory = 0
        process = None
    last_memory_log = time.time()
    last_tracemalloc_snapshot = time.time()
    
    # Display settings
    show_text = True
    is_fullscreen = False
    segmenter.enable_face_landmarks = True  # Default: face landmarks ON
    screen_width, screen_height = get_screen_resolution()
    print(f"Screen resolution: {screen_width}x{screen_height}")
    
    # Create main window - resizable and fullscreen-capable
    cv2.namedWindow("Live Segmentation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Segmentation", width, height)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror the camera view horizontally (like a mirror)
            frame = cv2.flip(frame, 1)
            
            # Skip warmup frames - don't render camera feed, just process
            if frame_count < warmup_frames:
                # Process frame but don't display (camera feed is hidden)
                if frame_count % process_every_n == 0:
                    try:
                        timestamp_ms = int((time.time() - start_time) * 1000)
                        segmenter.process_frame(frame, timestamp_ms, processing_size)
                    except Exception as e:
                        # Silently handle processing errors - continue with next frame
                        pass
                frame_count += 1
                continue
            
            # Process frame
            if frame_count % process_every_n == 0:
                try:
                    timestamp_ms = int((time.time() - start_time) * 1000)
                    segmenter.process_frame(frame, timestamp_ms, processing_size)
                except Exception as e:
                    # Silently handle processing errors - continue with next frame
                    pass
            
            # Fast non-blocking mask check
            try:
                with segmenter.lock:
                    has_mask = segmenter.latest_mask is not None
            except:
                has_mask = False
            
            # Always use mask-only mode (textured segmentations only)
            # Always show categories (button removed)
            if has_mask:
                try:
                    display_frame = segmenter.get_mask_only(frame, show_categories=True)
                except Exception as e:
                    # Fallback to black frame on error
                    display_frame = np.zeros_like(frame)
            else:
                display_frame = np.zeros_like(frame)
            
            # Draw face landmarks if enabled
            if segmenter.enable_face_landmarks and segmenter.face_landmarker:
                try:
                    display_frame = segmenter.draw_face_landmarks(display_frame)
                except Exception as e:
                    # Silently handle face landmark drawing errors
                    pass
            
            # Save frame for capture (before logo overlay) - captures should not include logo
            capture_frame = display_frame.copy()
            
            # Overlay Glix logo at top center (only for display, not for captures)
            try:
                display_frame = segmenter.overlay_logo(display_frame)
            except Exception as e:
                # Silently handle logo overlay errors
                pass
            
            # Automatic texture randomization every 20 seconds
            current_time = time.time()
            time_since_randomize = current_time - last_randomize_time
            
            if time_since_randomize >= randomize_interval:
                # If textures were toggled off previously, toggle them back on first
                if textures_need_toggle_back:
                    segmenter.use_textures = True
                    textures_need_toggle_back = False
                    print(f"\n🎲 Toggling textures back ON...")
                
                print(f"\n🎲 Auto-randomizing textures (every {randomize_interval}s)...")
                segmenter.randomize_textures()
                
                # Occasionally toggle textures off (30% chance)
                if random.random() < 0.3:
                    segmenter.use_textures = False
                    textures_need_toggle_back = True
                    print(f"   ⚠️ Textures toggled OFF (will toggle back ON next cycle)")
                
                last_randomize_time = current_time
                last_capture_time = last_randomize_time  # Reset capture time for first capture after randomize
            
            # Automatic capture: first capture 5 seconds after randomize, then every 10 seconds
            time_since_capture = current_time - last_capture_time
            time_since_randomize = current_time - last_randomize_time
            
            # Check if it's time to capture:
            # 1. First capture: 5 seconds after randomize (when last_capture_time <= last_randomize_time)
            # 2. Subsequent captures: every 10 seconds after the last capture
            should_capture = False
            if last_capture_time <= last_randomize_time and time_since_randomize >= capture_delay:
                # First capture after randomize (5 seconds)
                should_capture = True
            elif last_capture_time > last_randomize_time and time_since_capture >= capture_interval:
                # Subsequent captures (every 10 seconds)
                should_capture = True
            
            if should_capture:
                try:
                    os.makedirs("captures", exist_ok=True)
                    capture_filename = f"./captures/segmented_frame_{int(current_time)}.jpg"
                    cv2.imwrite(capture_filename, capture_frame)  # Use capture_frame without logo
                    print(f"📸 Auto-captured: {capture_filename}")
                    last_capture_time = current_time
                except Exception as e:
                    print(f"⚠️ Failed to auto-capture frame: {e}")
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Only show text if toggle is on
            if show_text:
                mode = "Textures" if (has_mask and segmenter.use_textures) else "Mask"
                # Ensure GPU status is always a valid string
                gpu_status = "GPU" if (hasattr(segmenter, 'use_gpu') and segmenter.use_gpu) else "CPU"
                status = "✓" if has_mask else "⏳"
                bg_status = "BG:Orig" if segmenter.keep_background_original else "BG:FX"
                
                # Create text string once and reuse
                text_overlay = f"{mode} | {gpu_status} {status} | {bg_status} | FPS: {fps:.1f} | Proc: {processing_size}px"
                cv2.putText(display_frame, text_overlay,
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                del text_overlay  # Clear string reference immediately
            
            # Resize frame to fill screen when in fullscreen mode (maintain aspect ratio)
            if is_fullscreen:
                try:
                    # Store old reference before creating new one
                    old_display_frame = display_frame
                    display_frame = resize_to_fill_screen(display_frame, screen_width, screen_height)
                    # Explicitly delete old frame to free memory immediately
                    del old_display_frame
                except Exception as e:
                    # If resize fails, use original frame
                    pass
            
            try:
                cv2.imshow("Live Segmentation", display_frame)
                # Note: cv2.imshow may buffer frames internally, but we can't control that
                # The frame will be collected by GC after we delete the reference
            except Exception as e:
                # Handle display errors gracefully
                print(f"⚠️ Display error (non-fatal): {e}")
                # Continue running
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Randomize textures
                print("\n🎲 Randomizing textures...")
                segmenter.randomize_textures()
            elif key == ord('t'):
                # Toggle text display
                show_text = not show_text
                print(f"Text display: {'ON' if show_text else 'OFF'}")
            elif key == ord('p'):
                # Cycle processing size
                processing_size_idx = (processing_size_idx + 1) % len(processing_sizes)
                processing_size = processing_sizes[processing_size_idx]
                print(f"Processing size: {processing_size}px")
            elif key == ord('x'):
                # Toggle textures on/off
                segmenter.use_textures = not segmenter.use_textures
                print(f"Textures: {'ON' if segmenter.use_textures else 'OFF'}")
            elif key == ord('b'):
                # Toggle background effects
                segmenter.keep_background_original = not segmenter.keep_background_original
                print(f"Background effects: {'OFF (original)' if segmenter.keep_background_original else 'ON (effects applied)'}")
            elif key == ord('f'):
                # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    # Get current screen resolution (in case it changed)
                    screen_width, screen_height = get_screen_resolution()
                    cv2.setWindowProperty("Live Segmentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print(f"Fullscreen: ON ({screen_width}x{screen_height})")
                else:
                    cv2.setWindowProperty("Live Segmentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    # Restore window size
                    cv2.resizeWindow("Live Segmentation", width, height)
                    print("Fullscreen: OFF")
            elif key == ord('s'):
                try:
                    os.makedirs("captures", exist_ok=True)
                    # Save frame without logo - use capture_frame which is defined before logo overlay
                    cv2.imwrite(f"./captures/segmented_frame_{frame_count}.jpg", capture_frame)
                    print(f"Saved: ./captures/segmented_frame_{frame_count}.jpg")
                except Exception as e:
                    print(f"⚠️ Failed to save frame: {e}")
            
            frame_count += 1
            
            # Periodic garbage collection to prevent memory accumulation
            if frame_count % gc_frequency == 0:
                # Force garbage collection every N frames
                collected = gc.collect()
                
                # Memory monitoring - log if memory is growing
                if memory_monitoring:
                    current_time = time.time()
                    if current_time - last_memory_log > 5.0:  # Check every 5 seconds
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_delta = current_memory - initial_memory
                        
                        # Check what Python objects are actually leaking
                        if current_time - last_tracemalloc_snapshot > 10.0:  # Every 10 seconds
                            snapshot_after = tracemalloc.take_snapshot()
                            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                            
                            print(f"\n🔍 Top 5 memory allocations:")
                            for index, stat in enumerate(top_stats[:5], 1):
                                print(f"  {index}. {stat}")
                            
                            last_tracemalloc_snapshot = current_time
                        
                        # Show pending MediaPipe operations
                        with segmenter._pending_lock:
                            pending = segmenter._pending_segments
                        
                        if collected > 0:
                            print(f"🧹 GC collected {collected} objects | Memory: {current_memory:.1f} MB (+{memory_delta:.1f} MB) | Pending: {pending}")
                        last_memory_log = current_time
            
            # Print FPS stats before clearing variables
            if frame_count % 60 == 0:
                processed = frame_count // process_every_n
                proc_fps = processed / elapsed if elapsed > 0 else 0
                
                # Get current memory before print (if monitoring enabled)
                if memory_monitoring:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = current_memory - initial_memory
                    
                    # Get pending MediaPipe operations
                    with segmenter._pending_lock:
                        pending = segmenter._pending_segments
                    
                    # Create print string and immediately print, then clear variables
                    log_msg = f"FPS: {fps:.1f} | Processed: {proc_fps:.1f} | Frames: {frame_count} | Mem: {current_memory:.1f} MB (+{memory_delta:.1f} MB) | Pending: {pending}"
                    
                    # If memory is growing too much, be even more aggressive
                    if memory_delta > 100:  # If memory increased by more than 100 MB
                        log_msg += " ⚠️ HIGH MEM!"
                        print(f"⚠️ High memory usage detected! Pending operations: {pending}")
                        print(f"   This suggests MediaPipe's C++ queue is holding frames")
                        # Can't force MediaPipe to release - it's in C++ code
                        # But we can skip more frames
                        segmenter._max_pending = max(1, segmenter._max_pending - 1)
                        print(f"   Reduced max pending to {segmenter._max_pending}")
                        for _ in range(3):  # Multiple GC passes for Python objects
                            gc.collect()
                else:
                    # Still show pending even without psutil
                    with segmenter._pending_lock:
                        pending = segmenter._pending_segments
                    log_msg = f"FPS: {fps:.1f} | Processed: {proc_fps:.1f} | Frames: {frame_count} | Pending: {pending}"
                
                print(log_msg)
                del log_msg, processed, proc_fps  # Clear temporary variables
                
                # Force GC after EVERY print to prevent accumulation
                gc.collect()
            
            # Clear frame references to help GC (after all uses)
            del display_frame
            # Note: 'frame' is read at start of loop, so deleting here is fine
            # It will be reassigned on next iteration
            del frame
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        # Catch any unexpected errors and log them, but try to continue
        print(f"\n⚠️ Unexpected error in main loop: {e}")
        print("   Attempting to continue...")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cap.release()
        except:
            pass
        try:
            segmenter.close()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    main()
