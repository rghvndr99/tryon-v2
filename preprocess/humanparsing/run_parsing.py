import pdb
from pathlib import Path
import sys
import os
import torch
import numpy as np

# Try to import onnxruntime, provide fallback if it fails
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
    print("✅ ONNX Runtime loaded successfully")
except ImportError as e:
    print(f"⚠️  ONNX Runtime import failed: {e}")
    print("🔄 Using fallback implementation for human parsing...")
    ONNXRUNTIME_AVAILABLE = False

    # Fallback implementation
    class MockSessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self.execution_mode = None
        def add_session_config_entry(self, key, value):
            pass

    class MockInferenceSession:
        def __init__(self, model_path, sess_options=None, providers=None):
            print(f"⚠️  Mock ONNX session for {model_path}")
            pass

        def run(self, output_names, input_feed):
            # Return dummy output for human parsing
            # This is a fallback - real parsing would need the ONNX model
            height, width = 512, 384  # Default dimensions
            if input_feed and len(input_feed) > 0:
                input_tensor = list(input_feed.values())[0]
                if hasattr(input_tensor, 'shape') and len(input_tensor.shape) >= 2:
                    height, width = input_tensor.shape[-2:]

            # Return dummy segmentation mask
            dummy_output = np.zeros((1, 20, height, width), dtype=np.float32)
            return [dummy_output]

    class ort:
        SessionOptions = MockSessionOptions
        InferenceSession = MockInferenceSession

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

        class ExecutionMode:
            ORT_SEQUENTIAL = "ORT_SEQUENTIAL"

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference


class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        # Only set CUDA device if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if torch.cuda.is_available():
            session_options.add_session_config_entry('gpu_id', str(gpu_id))
        self.session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_atr.onnx'),
                                            sess_options=session_options, providers=['CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_lip.onnx'),
                                                sess_options=session_options, providers=['CPUExecutionProvider'])
        

    def __call__(self, input_image):
        # torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
