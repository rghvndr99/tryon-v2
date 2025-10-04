"""
Compatibility layer for detectron2 imports.
Provides fallbacks when detectron2 is not available or has import issues.
"""

import sys
import warnings
import numpy as np
from PIL import Image
import cv2

# Try to import detectron2, provide fallbacks if it fails
try:
    from detectron2.config import CfgNode, get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.structures.instances import Instances
    from detectron2.utils.logger import setup_logger
    DETECTRON2_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Detectron2 import failed: {e}")
    print("🔄 Using fallback implementations...")
    DETECTRON2_AVAILABLE = False
    
    # Fallback implementations
    class CfgNode:
        def __init__(self):
            pass
        def merge_from_file(self, file):
            pass
        def merge_from_list(self, opts):
            pass
    
    def get_cfg():
        return CfgNode()
    
    def read_image(filename, format=None):
        """Fallback image reader"""
        img = cv2.imread(filename)
        if format == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    class DefaultPredictor:
        def __init__(self, cfg):
            print("⚠️  DefaultPredictor fallback - predictions will be empty")
            pass
        
        def __call__(self, image):
            # Return empty predictions
            return {"instances": Instances((image.shape[0], image.shape[1]))}
    
    class Instances:
        def __init__(self, image_size):
            self.image_size = image_size
            self._fields = {}
        
        def __len__(self):
            return 0
        
        def has(self, name):
            return name in self._fields
        
        def get(self, name):
            return self._fields.get(name, [])
        
        def set(self, name, value):
            self._fields[name] = value
    
    def setup_logger(name=None):
        pass

# Try to import densepose, provide fallbacks if it fails
try:
    from densepose import add_densepose_config
    from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
    from densepose.utils.logger import verbosity_to_level
    from densepose.vis.base import CompoundVisualizer
    from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
    from densepose.vis.densepose_outputs_vertex import (
        DensePoseOutputsTextureVisualizer,
        DensePoseOutputsVertexVisualizer,
        get_texture_atlas,
    )
    from densepose.vis.densepose_results import (
        DensePoseResultsContourVisualizer,
        DensePoseResultsFineSegmentationVisualizer,
        DensePoseResultsUVisualizer,
        DensePoseResultsVVisualizer,
    )
    from densepose.vis.densepose_results_textures import (
        DensePoseResultsVisualizerWithTexture,
        get_texture_atlases_dict,
    )
    from densepose.vis.extractor import (
        CompoundExtractor,
        DensePoseOutputsExtractor,
        DensePoseResultExtractor,
        create_extractor,
    )
    DENSEPOSE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  DensePose import failed: {e}")
    print("🔄 Using DensePose fallback implementations...")
    DENSEPOSE_AVAILABLE = False

    # Fallback implementations for densepose
    def add_densepose_config(cfg):
        pass

    class DensePoseChartPredictorOutput:
        def __init__(self, *args, **kwargs):
            pass

    class DensePoseEmbeddingPredictorOutput:
        def __init__(self, *args, **kwargs):
            pass

    def verbosity_to_level(verbosity):
        return 0

    class CompoundVisualizer:
        def __init__(self, *args, **kwargs):
            pass
        def visualize(self, *args, **kwargs):
            return None

    class ScoredBoundingBoxVisualizer:
        def __init__(self, *args, **kwargs):
            pass

    class DensePoseOutputsTextureVisualizer:
        def __init__(self, *args, **kwargs):
            pass

    class DensePoseOutputsVertexVisualizer:
        def __init__(self, *args, **kwargs):
            pass

    def get_texture_atlas(*args, **kwargs):
        return None

    def get_texture_atlases(*args, **kwargs):
        return None

    class DensePoseResultsContourVisualizer:
        def __init__(self, *args, **kwargs):
            pass

    class DensePoseResultsFineSegmentationVisualizer:
        def __init__(self, *args, **kwargs):
            pass

    class DensePoseResultsUVisualizer:
        def __init__(self, *args, **kwargs):
            pass

    class DensePoseResultsVVisualizer:
        def __init__(self, *args, **kwargs):
            pass

    class DensePoseResultsVisualizerWithTexture:
        def __init__(self, *args, **kwargs):
            pass

    def get_texture_atlases_dict(*args, **kwargs):
        return {}

    class CompoundExtractor:
        def __init__(self, *args, **kwargs):
            pass

    class DensePoseOutputsExtractor:
        def __init__(self, *args, **kwargs):
            pass

    class DensePoseResultExtractor:
        def __init__(self, *args, **kwargs):
            pass

    def create_extractor(*args, **kwargs):
        return CompoundExtractor()

# Additional compatibility functions
def convert_PIL_to_numpy(image, format="BGR"):
    """Convert PIL image to numpy array"""
    img_array = np.array(image)
    if format == "BGR" and len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array

def _apply_exif_orientation(image):
    """Apply EXIF orientation to PIL image"""
    try:
        from PIL.ExifTags import ORIENTATION
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(ORIENTATION)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except:
        pass  # If EXIF processing fails, just return original image
    return image
