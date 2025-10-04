"""
Preprocessing Compatibility Layer for IDM-VTON on Hugging Face Spaces
Provides fallback implementations for OpenPose and Human Parsing when models fail to load
"""

import numpy as np
import torch
from PIL import Image
import warnings

# Try to import real preprocessing modules
try:
    from preprocess.humanparsing.run_parsing import Parsing as RealParsing
    PARSING_AVAILABLE = True
    print("✅ Real human parsing available")
except ImportError as e:
    print(f"⚠️  Human parsing import failed: {e}")
    PARSING_AVAILABLE = False

try:
    from preprocess.openpose.run_openpose import OpenPose as RealOpenPose
    OPENPOSE_AVAILABLE = True
    print("✅ Real OpenPose available")
except ImportError as e:
    print(f"⚠️  OpenPose import failed: {e}")
    OPENPOSE_AVAILABLE = False


class MockParsing:
    """Fallback implementation for human parsing"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        print("⚠️  Using mock human parsing - results will be simplified")
    
    def __call__(self, input_image):
        """Return dummy parsing results"""
        if isinstance(input_image, Image.Image):
            width, height = input_image.size
        else:
            height, width = 512, 384  # Default size
        
        # Create dummy parsing mask (all background)
        parsing_result = np.zeros((height, width), dtype=np.uint8)
        
        # Create dummy face mask
        face_mask = torch.zeros((height, width), dtype=torch.float32)
        
        # Convert to PIL Image with palette
        palette = self._get_palette(19)
        output_img = Image.fromarray(parsing_result)
        output_img.putpalette(palette)
        
        return output_img, face_mask
    
    def _get_palette(self, num_cls):
        """Generate color palette for segmentation"""
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette


class MockOpenPose:
    """Fallback implementation for OpenPose"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        print("⚠️  Using mock OpenPose - keypoints will be estimated")
        
        # Mock preprocessor for compatibility
        self.preprocessor = MockPreprocessor()
    
    def __call__(self, input_image, resolution=384):
        """Return dummy keypoints"""
        if isinstance(input_image, Image.Image):
            width, height = input_image.size
        else:
            height, width = 512, 384  # Default size
        
        # Generate dummy keypoints for 18 body joints
        # These are rough estimates for a standing person
        keypoints = []
        
        # Basic human pose keypoints (normalized to image size)
        # Order: nose, neck, right_shoulder, right_elbow, right_wrist,
        #        left_shoulder, left_elbow, left_wrist, right_hip, right_knee,
        #        right_ankle, left_hip, left_knee, left_ankle, right_eye,
        #        left_eye, right_ear, left_ear
        
        center_x = width // 2
        center_y = height // 2
        
        # Dummy keypoints for a standing person
        dummy_points = [
            [center_x, center_y - height * 0.35],      # nose
            [center_x, center_y - height * 0.25],      # neck
            [center_x + width * 0.15, center_y - height * 0.2],   # right_shoulder
            [center_x + width * 0.25, center_y - height * 0.05],  # right_elbow
            [center_x + width * 0.3, center_y + height * 0.1],    # right_wrist
            [center_x - width * 0.15, center_y - height * 0.2],   # left_shoulder
            [center_x - width * 0.25, center_y - height * 0.05],  # left_elbow
            [center_x - width * 0.3, center_y + height * 0.1],    # left_wrist
            [center_x + width * 0.1, center_y + height * 0.1],    # right_hip
            [center_x + width * 0.1, center_y + height * 0.3],    # right_knee
            [center_x + width * 0.1, center_y + height * 0.45],   # right_ankle
            [center_x - width * 0.1, center_y + height * 0.1],    # left_hip
            [center_x - width * 0.1, center_y + height * 0.3],    # left_knee
            [center_x - width * 0.1, center_y + height * 0.45],   # left_ankle
            [center_x + width * 0.05, center_y - height * 0.37],  # right_eye
            [center_x - width * 0.05, center_y - height * 0.37],  # left_eye
            [center_x + width * 0.08, center_y - height * 0.35],  # right_ear
            [center_x - width * 0.08, center_y - height * 0.35],  # left_ear
        ]
        
        # Ensure we have exactly 18 keypoints
        while len(dummy_points) < 18:
            dummy_points.append([0, 0])
        
        keypoints = {"pose_keypoints_2d": dummy_points[:18]}
        return keypoints


class MockPreprocessor:
    """Mock preprocessor for OpenPose compatibility"""
    
    def __init__(self):
        self.body_estimation = MockBodyEstimation()


class MockBodyEstimation:
    """Mock body estimation model"""
    
    def __init__(self):
        self.model = MockModel()


class MockModel:
    """Mock model that can be moved to device"""
    
    def to(self, device):
        """Mock to() method for device compatibility"""
        return self


def Parsing(gpu_id=0):
    """Factory function that returns real or mock parsing"""
    if PARSING_AVAILABLE:
        try:
            return RealParsing(gpu_id)
        except Exception as e:
            print(f"⚠️  Real parsing failed to initialize: {e}")
            print("🔄 Falling back to mock parsing...")
            return MockParsing(gpu_id)
    else:
        return MockParsing(gpu_id)


def OpenPose(gpu_id=0):
    """Factory function that returns real or mock OpenPose"""
    if OPENPOSE_AVAILABLE:
        try:
            return RealOpenPose(gpu_id)
        except Exception as e:
            print(f"⚠️  Real OpenPose failed to initialize: {e}")
            print("🔄 Falling back to mock OpenPose...")
            return MockOpenPose(gpu_id)
    else:
        return MockOpenPose(gpu_id)


# Export the factory functions
__all__ = ['Parsing', 'OpenPose', 'PARSING_AVAILABLE', 'OPENPOSE_AVAILABLE']
