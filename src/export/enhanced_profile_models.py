import os
import logging
from dotenv import load_dotenv
from edgeimpulse.model import profile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("EDGE_IMPULSE_API_KEY")
model_path = "models/onnx/iris_model.onnx"  # Updated to use the existing model file

if not api_key:
    logger.error("❌ API key missing. Add EDGE_IMPULSE_API_KEY to your .env file")
    exit(1)

logger.info(f"Profiling model: {model_path}")
result = profile(model=model_path, api_key=api_key)

logger.info("\n📊 Model Profile Results:")
logger.info(f"RAM usage: {result.ram / 1024:.2f} KB")
logger.info(f"ROM size: {result.rom / 1024:.2f} KB")
logger.info(f"Inference time: {result.inference_time:.2f} ms")
logger.info(f"Target device: {result.target_device}")