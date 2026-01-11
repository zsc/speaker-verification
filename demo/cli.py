import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
import logging
from scipy.spatial.distance import cdist

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from demo.model_loader import load_models
from verification import enhance_and_embeding

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Speaker Verification CLI Demo")
    parser.add_argument("input_file", type=str, help="Path to the input audio file (WAV)")
    parser.add_argument("--threshold", type=float, default=0.5, help="VAD threshold (default: 0.5)")
    parser.add_argument("--min_silence", type=int, default=500, help="Min silence duration in ms (default: 500)")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # 1. Load Models
    logger.info("Initializing models...")
    vad_model, vad_utils, embedding_model, feature_extractor, enhancement_model = load_models()
    (get_speech_timestamps, _, read_audio, _, _) = vad_utils
    
    # 2. Load Audio
    logger.info(f"Loading audio: {args.input_file}")
    wav, sr = torchaudio.load(args.input_file)
    
    if sr != 16000:
        logger.info(f"Resampling from {sr}Hz to 16000Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
    
    # Ensure mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    # 3. VAD Segmentation
    logger.info("Running VAD...")
    # get_speech_timestamps expects (samples,) tensor for single channel usually, or just handled.
    # checking usage: expects tensor
    wav_sq = wav.squeeze()
    
    speech_timestamps = get_speech_timestamps(
        wav_sq, 
        vad_model, 
        threshold=args.threshold,
        min_silence_duration_ms=args.min_silence
    )
    
    if not speech_timestamps:
        logger.info("No speech detected.")
        sys.exit(0)
    
    logger.info(f"Detected {len(speech_timestamps)} speech segments.")
    
    embeddings = []
    ids = []
    
    # 4. Process Segments
    for i, ts in enumerate(speech_timestamps):
        start = ts['start']
        end = ts['end']
        duration = (end - start) / 16000
        logger.info(f"Processing Segment {i+1}: {start/16000:.2f}s - {end/16000:.2f}s ({duration:.2f}s)")
        
        segment = wav[:, start:end] # (1, T)
        
        # Verify/Enhance/Embed
        try:
            embedder = enhance_and_embeding(
                noisy=segment,
                enhancement_model=enhancement_model,
                embedding_model=embedding_model,
                feature_extractor=feature_extractor
            )
            embedding_tensor = embedder.encoder()
            embeddings.append(embedding_tensor.tolist())
            ids.append(i + 1)
        except Exception as e:
            logger.error(f"Failed to process segment {i+1}: {e}")

    # 5. Distance Matrix
    if len(embeddings) > 1:
        logger.info("Calculating Distance Matrix (Cosine)...")
        mat = cdist(embeddings, embeddings, metric='cosine')
        
        print("\nDistance Matrix:")
        # Print Header
        print("      ", end="")
        for id in ids:
            print(f" Seg {id:<3} ", end="")
        print()
        
        for r, row in enumerate(mat):
            print(f"Seg {ids[r]:<2} ", end="")
            for val in row:
                print(f" {val:.3f}   ", end="")
            print()
    else:
        logger.info("Not enough segments to calculate distance matrix.")

if __name__ == "__main__":
    main()
