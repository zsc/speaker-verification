import os
import sys
import torch
import numpy as np
import json
import logging
import base64
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from scipy.spatial.distance import cdist

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEMO_DIR = os.path.dirname(__file__)

# Add Root to sys.path to allow imports
sys.path.append(ROOT_DIR)

# Switch to Root dir for model loading compatibility
os.chdir(ROOT_DIR)

from verification import enhance_and_embeding

# Flask App
app = Flask(__name__, 
            template_folder=os.path.join(DEMO_DIR, 'templates'),
            static_folder=os.path.join(DEMO_DIR, 'static'))
socketio = SocketIO(app, cors_allowed_origins='*')

from demo.model_loader import load_models

# Global State
HISTORY = [] # List of {'id': int, 'embedding': list, 'timestamp': str}
MAX_HISTORY = 8
SEGMENT_COUNTER = 0

# Models
VAD_MODEL = None
VAD_UTILS = None
ENHANCEMENT_MODEL = None
EMBEDDING_MODEL = None
FEATURE_EXTRACTOR = None

# Audio Buffer for VAD
SAMPLE_RATE = 16000
WINDOW_SIZE_SAMPLES = 512  # 32ms at 16k

# Config (Can be updated from frontend)
CONFIG = {
    'vad_threshold': 0.5,
    'min_silence_duration_ms': 500,
    'speech_pad_ms': 30,
}

# State for current utterance
CURRENT_AUDIO_BUFFER = []
IS_SPEAKING = False
SILENCE_COUNTER = 0

def init_server_models():
    global VAD_MODEL, VAD_UTILS, EMBEDDING_MODEL, FEATURE_EXTRACTOR, ENHANCEMENT_MODEL
    
    VAD_MODEL, VAD_UTILS, EMBEDDING_MODEL, FEATURE_EXTRACTOR, ENHANCEMENT_MODEL = load_models()
    logger.info("Server models initialized.")

init_server_models()

def reset_vad_state():
    global VAD_ITERATOR, CURRENT_AUDIO_BUFFER, IS_SPEAKING, SILENCE_COUNTER
    VAD_ITERATOR = None # Reset VAD internal state
    CURRENT_AUDIO_BUFFER = []
    IS_SPEAKING = False
    SILENCE_COUNTER = 0
    # Also reset history? Maybe not.

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    reset_vad_state()
    emit('server_log', {'msg': 'Connected to Server. Models Ready.'})
    emit('config_sync', CONFIG)

@socketio.on('update_config')
def handle_config(data):
    global CONFIG
    CONFIG.update(data)
    emit('server_log', {'msg': f'Config Updated: {CONFIG}'})

@socketio.on('audio_chunk')
def handle_audio(data):
    global IS_SPEAKING, SILENCE_COUNTER, CURRENT_AUDIO_BUFFER, SEGMENT_COUNTER
    
    # Data is raw bytes (Float32)
    try:
        np_chunk = np.frombuffer(data, dtype=np.float32)
        chunk = torch.from_numpy(np_chunk)
    except Exception as e:
        # Fallback if data is JSON list
        chunk = torch.FloatTensor(data)
    
    # VAD Processing
    # Silero expects (batch, samples)
    if VAD_ITERATOR is None:
        # Reset iterator logic if needed, but silero `model(x, sr)` is stateless if we don't pass state?
        # Actually standard usage is `model(x, sr)` and it returns prob.
        # But for streaming, we might need `VADIterator` if using that class.
        # Let's stick to simple `model(x, sr)` which is stateless unless using the LSTM version carefully.
        # Wait, Silero V5 is LSTM and stateful?
        # "The model is not stateful" unless you use the JIT version with state.
        # Let's assume standard hub load returns the standard model.
        pass

    speech_prob = VAD_MODEL(chunk.unsqueeze(0), SAMPLE_RATE).item()
    
    # State Machine
    threshold = CONFIG['vad_threshold']
    min_silence_samples = (CONFIG['min_silence_duration_ms'] / 1000) * SAMPLE_RATE
    min_silence_frames = min_silence_samples / len(chunk)

    CURRENT_AUDIO_BUFFER.append(chunk)
    
    if speech_prob > threshold:
        if not IS_SPEAKING:
            IS_SPEAKING = True
            emit('vad_event', {'type': 'start', 'prob': speech_prob})
        SILENCE_COUNTER = 0
    else:
        if IS_SPEAKING:
            SILENCE_COUNTER += 1
            if SILENCE_COUNTER >= min_silence_frames:
                IS_SPEAKING = False
                emit('vad_event', {'type': 'end', 'prob': speech_prob})
                
                # Process the segment
                process_segment()
                CURRENT_AUDIO_BUFFER = [] # Clear buffer
                SILENCE_COUNTER = 0

def process_segment():
    global SEGMENT_COUNTER
    emit('server_log', {'msg': 'Processing Segment...'})
    emit('proc_event', {'type': 'start'})
    
    # Concatenate audio
    full_audio = torch.cat(CURRENT_AUDIO_BUFFER)
    
    # Check if audio is long enough (e.g. > 0.5s)
    if len(full_audio) < SAMPLE_RATE * 0.5:
        emit('server_log', {'msg': 'Segment too short, ignored.'})
        emit('proc_event', {'type': 'end'})
        return

    # Prepare for verification
    audio_input = full_audio.unsqueeze(0)
    
    try:
        # Extract Embedding
        # Pass pre-loaded models to avoid reloading
        embedder = enhance_and_embeding(
            noisy=audio_input,
            enhancement_model=ENHANCEMENT_MODEL,
            embedding_model=EMBEDDING_MODEL,
            feature_extractor=FEATURE_EXTRACTOR
        )
        embedding_tensor = embedder.encoder() # returns tensor
        embedding_list = embedding_tensor.tolist()
        
        SEGMENT_COUNTER += 1
        
        # Update History
        seg_data = {
            'id': SEGMENT_COUNTER,
            'embedding': embedding_list
        }
        
        update_history(seg_data)
        
    except Exception as e:
        logger.error(f"Error processing segment: {e}")
        emit('server_log', {'msg': f'Error: {str(e)}'})
    
    emit('proc_event', {'type': 'end'})

def update_history(new_seg):
    global HISTORY
    HISTORY.append(new_seg)
    if len(HISTORY) > MAX_HISTORY:
        HISTORY.pop(0)
    
    # Calculate Distance Matrix
    embeddings = [h['embedding'] for h in HISTORY]
    ids = [h['id'] for h in HISTORY]
    
    if len(embeddings) > 0:
        # Cosine Distance
        # cdist expects 2D array
        mat = cdist(embeddings, embeddings, metric='cosine')
        
        # Send to client
        emit('matrix_update', {
            'matrix': mat.tolist(),
            'ids': ids
        })

if __name__ == '__main__':
    port = 8000
    print(f"Starting server on http://localhost:{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
