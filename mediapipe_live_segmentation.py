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
        
        # Configurable parameters
        self.blend_factor = 0.3
        self.blur_kernel_size = 35
        self.use_textures = True  # Start with textures ON
        self.textures = {}  # Store textures for each category: {0: texture_img, 1: texture_img, ...}
        self.texture_scales = {}  # Random scale for each category
        self.texture_folders = ["textures/GANSTILLFINAL", "textures/tiles2"]  # Folders with texture PNGs (inside textures folder)
        self.texture_files = []  # List of available texture files
        self._tiled_cache = {}  # Cache tiled textures: {(cat_id, scale, w, h): tiled_texture}
        self._last_frame_size = None  # Track frame size changes
        
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
        
        self.segmenter = self._create_segmenter(model_path)
        
        # Initialize face landmarker
        if self.enable_face_landmarks:
            self.face_landmarker = self._create_face_landmarker()
        else:
            self.face_landmarker = None
        
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
        
        self.textures = {}
        self.texture_scales = {}
        self._tiled_cache = {}  # Clear cache when textures change
        
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
            # Limit cache size
            if len(self._tiled_cache) > 12:
                oldest_key = next(iter(self._tiled_cache))
                del self._tiled_cache[oldest_key]
        
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
                if hasattr(result, 'category_mask') and result.category_mask is not None:
                    mask = result.category_mask.numpy_view()
                    with self.lock:
                        self.latest_mask = mask
            except:
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
                            self.latest_face_landmarks = result
                except:
                    pass
            
            base_options = BaseOptions(
                model_asset_path=face_model_path,
                delegate=BaseOptions.Delegate.GPU if self.use_gpu else BaseOptions.Delegate.CPU
            )
            
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_faces=1,
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
        
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        if self.use_srgba:
            rgba_frame = np.dstack([rgb_frame, np.full((rgb_frame.shape[0], rgb_frame.shape[1]), 255, dtype=np.uint8)])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            self.segmenter.segment_async(mp_image, timestamp_ms)
            
            # Process face landmarks in parallel (same frame)
            if self.face_landmarker:
                try:
                    self.face_landmarker.detect_async(mp_image, timestamp_ms)
                except:
                    pass
        except:
            pass
    
    # Removed get_segmented_frame - only using mask-only mode for speed
    
    def get_mask_only(self, original_frame, show_categories=True):
        """Get only the segmentation mask with textures - highly optimized."""
        # Fast lock check
        with self.lock:
            if self.latest_mask is None:
                return np.zeros_like(original_frame)
            mask = self.latest_mask.copy()  # Copy needed for thread safety
        
        h, w = original_frame.shape[:2]
        
        # Fast resize if needed
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Clear cache if frame size changed
        if self._last_frame_size != (h, w):
            self._tiled_cache = {}
            self._last_frame_size = (h, w)
        
        if show_categories and self.use_textures and self.textures:
            # ULTRA-OPTIMIZED: Use direct mask indexing instead of np.where
            # Pre-allocate output with category colors (fastest base)
            output = CATEGORY_COLORS[mask].copy()
            
            # Only process categories that have textures loaded
            for cat_id in self.textures:
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
                        oldest_key = next(iter(self._tiled_cache))
                        del self._tiled_cache[oldest_key]
                
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
            # Fastest path: direct color lookup
            output = CATEGORY_COLORS[mask]
        else:
            # Binary mask
            person_mask = mask > 0
            output = np.where(person_mask[:, :, np.newaxis], [255, 255, 255], [0, 0, 0]).astype(np.uint8)
        
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    def draw_face_landmarks(self, frame):
        """Draw face landmarks as a mesh with connections."""
        # Fast lock check
        try:
            with self.face_lock:
                if self.latest_face_landmarks is None or not self.latest_face_landmarks.face_landmarks:
                    return frame
                face_landmarks = self.latest_face_landmarks.face_landmarks[0]
        except:
            return frame
        
        h, w = frame.shape[:2]
        
        if len(face_landmarks) < 478:
            return frame  # Invalid landmark data
        
        # Pre-compute all landmark points (vectorized for performance)
        points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks], dtype=np.int32)
        
        # Use randomized color scheme that changes with textures
        LANDMARK_COLOR, CONNECTION_COLOR, KEY_POINT_COLOR = self.current_landmark_colors
        
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
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera: {width}x{height}\n")
    
    # Control panel disabled - using hotkeys only
    # control_panel = ControlPanel(segmenter)
    # control_panel.set_cam_resolution(width, height)
    
    # Auto-randomize textures on startup (non-blocking)
    print("\n🎲 Auto-loading random textures...")
    segmenter.randomize_textures()
    
    # Processing size settings - cycle through with hotkey
    processing_sizes = [256, 320, 384, 448, 512, 576, 640]
    processing_size_idx = 6  # Start at 384 (index 2)
    processing_size = processing_sizes[processing_size_idx]
    
    frame_count = 0
    start_time = time.time()
    process_every_n = 1 if segmenter.use_gpu else 2
    
    # Warmup: skip first few frames to let camera stabilize
    warmup_frames = 5
    
    # Display settings
    show_text = True
    is_fullscreen = False
    segmenter.enable_face_landmarks = True  # Default: face landmarks ON
    
    # Create main window - resizable and fullscreen-capable
    cv2.namedWindow("Live Segmentation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Segmentation", width, height)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip warmup frames - don't render camera feed, just process
            if frame_count < warmup_frames:
                # Process frame but don't display (camera feed is hidden)
                if frame_count % process_every_n == 0:
                    timestamp_ms = int((time.time() - start_time) * 1000)
                    segmenter.process_frame(frame, timestamp_ms, processing_size)
                frame_count += 1
                continue
            
            # Process frame
            if frame_count % process_every_n == 0:
                timestamp_ms = int((time.time() - start_time) * 1000)
                segmenter.process_frame(frame, timestamp_ms, processing_size)
            
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
                except:
                    display_frame = np.zeros_like(frame)
            else:
                display_frame = np.zeros_like(frame)
            
            # Draw face landmarks if enabled
            if segmenter.enable_face_landmarks and segmenter.face_landmarker:
                try:
                    display_frame = segmenter.draw_face_landmarks(display_frame)
                except:
                    pass
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Only show text if toggle is on
            if show_text:
                mode = "Textures" if (has_mask and segmenter.use_textures) else "Mask"
                gpu = "GPU" if segmenter.use_gpu else "CPU"
                status = "✓" if has_mask else "⏳"
                
                cv2.putText(display_frame, f"{mode} | {gpu} {status} | FPS: {fps:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Live Segmentation", display_frame)
            
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
            elif key == ord('f'):
                # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty("Live Segmentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Fullscreen: ON")
                else:
                    cv2.setWindowProperty("Live Segmentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    # Restore window size
                    cv2.resizeWindow("Live Segmentation", width, height)
                    print("Fullscreen: OFF")
            elif key == ord('s'):
                cv2.imwrite(f"./captures/segmented_frame_{frame_count}.jpg", display_frame)
                print(f"Saved: ./captures/segmented_frame_{frame_count}.jpg")
            
            frame_count += 1
            
            if frame_count % 60 == 0:
                processed = frame_count // process_every_n
                proc_fps = processed / elapsed if elapsed > 0 else 0
                print(f"FPS: {fps:.1f} | Processed: {proc_fps:.1f} | Frames: {frame_count}")
    
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        segmenter.close()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    main()
