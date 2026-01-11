import os
import sys
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths - Assuming this is run from root or we can adjust relative paths
# But imports need to work. 
# We assume the script utilizing this is running with project root in sys.path or as CWD.

def load_models():
    """
    Loads VAD, Speaker Embedding, and Enhancement models.
    Returns:
        vad_model, vad_utils, embedding_model, feature_extractor, enhancement_model
    """
    vad_model = None
    vad_utils = None
    embedding_model = None
    feature_extractor = None
    enhancement_model = None

    # 1. VAD Model
    try:
        logger.info("Loading Silero VAD Model...")
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False)
        vad_model = model
        vad_utils = utils
        logger.info("VAD Model Loaded.")
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}")
        # Depending on usage, we might want to raise or return None
        # For CLI/Server, critical failure.
        sys.exit(1)

    # 2. Speaker Embedding Model
    try:
        logger.info("Loading Speaker Embedding Model (ERes2Net)...")
        # Ensure we can import from Speaker/
        from Speaker.speakerlab.bin.infer import embedding as EmbeddingClass
        
        # Initialize with dummy to trigger model load
        # This relies on the class structure we modified earlier
        temp_embedder = EmbeddingClass(wav_file=None)
        embedding_model = temp_embedder.embedding_model
        feature_extractor = temp_embedder.feature_extractor
        logger.info("Speaker Embedding Model Loaded.")
    except Exception as e:
        logger.error(f"Failed to load Speaker Embedding model: {e}")
        logger.error("Please ensure './model/pretrained_eres2net_aug.ckpt' exists.")
        sys.exit(1)

    # 3. Enhancement Model (Optional)
    try:
        logger.info("Checking Enhancement Model (CMGAN)...")
        from CMGAN.inference import enhancement
        model_path = "./model/ckpt"
        if os.path.exists(model_path):
            enhancement_model = enhancement(noisy_path=None, model_path=model_path)
            logger.info("Enhancement Model (CMGAN) Loaded.")
        else:
            logger.warning(f"Enhancement model checkpoint not found at {model_path}. Running without enhancement.")
            enhancement_model = None
    except Exception as e:
        logger.error(f"Error during Enhancement model check: {e}")
        enhancement_model = None

    return vad_model, vad_utils, embedding_model, feature_extractor, enhancement_model
