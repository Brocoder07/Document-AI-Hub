import whisper
import os
import math
import torch
import concurrent.futures
from pydub import AudioSegment
import tempfile
import logging
import threading

# Setup Logger
logger = logging.getLogger(__name__)

# Thread-local storage to hold a separate model instance for each thread
_thread_local = threading.local()
_model_name = "base"

def get_device():
    """
    Returns 'cuda' if GPU is available, otherwise 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_whisper_model():
    """
    Loads the Whisper model into THREAD-LOCAL storage.
    This ensures each thread has its own independent model instance,
    allowing true parallel inference without race conditions or locks.
    """
    # Check if this specific thread already has a model loaded
    if not hasattr(_thread_local, "model"):
        device = get_device()
        logger.info(f"[Thread-{threading.get_ident()}] Loading Whisper model: {_model_name} on {device}...")
        _thread_local.model = whisper.load_model(_model_name, device=device)
        logger.info(f"[Thread-{threading.get_ident()}] Whisper model loaded.")
    
    return _thread_local.model

def _process_chunk(chunk_data):
    """
    Worker function to process a single audio chunk.
    Args:
        chunk_data: tuple (index, start_ms, end_ms, audio_segment, temp_dir)
    """
    i, start_ms, end_ms, audio_obj, temp_dir = chunk_data
    
    try:
        # 1. Extract Chunk
        chunk = audio_obj[start_ms:end_ms]
        
        # 2. Pre-process strictly for Whisper (16kHz, Mono)
        chunk = chunk.set_frame_rate(16000).set_channels(1)
        
        # 3. Export as WAV (Lossless)
        # Using WAV avoids compression artifacts that confuse the model
        chunk_filename = os.path.join(temp_dir, f"chunk_{i}.wav")
        chunk.export(chunk_filename, format="wav")
        
        # 4. Transcribe
        # Each thread calls this, getting its own PRIVATE model instance
        model = get_whisper_model()
        device = get_device()
        use_fp16 = (device == "cuda")
        
        # No Lock needed here anymore! True Parallelism.
        result = model.transcribe(chunk_filename, fp16=use_fp16)
            
        text = result.get("text", "").strip()
        
        logger.info(f"Completed chunk {i}")
        return (i, text)
        
    except Exception as e:
        logger.error(f"Error processing chunk {i}: {e}")
        return (i, "")

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the audio file at the given path.
    Uses parallel execution with independent model instances.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at {file_path}")
        
    # Load Audio
    try:
        logger.info(f"Loading audio file: {file_path}")
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        raise Exception(f"Failed to load audio file: {str(e)}")

    duration_ms = len(audio)
    chunk_size_ms = 10 * 60 * 1000 # 10 minutes
    
    # If short, process directly
    if duration_ms < chunk_size_ms:
        with tempfile.TemporaryDirectory() as temp_dir:
            return _process_chunk((0, 0, duration_ms, audio, temp_dir))[1]

    logger.info(f"Audio length: {duration_ms/1000/60:.2f} mins. Starting parallel processing...")
    
    total_chunks = math.ceil(duration_ms / chunk_size_ms)
    chunk_args = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(total_chunks):
            start_ms = i * chunk_size_ms
            end_ms = min((i + 1) * chunk_size_ms, duration_ms)
            chunk_args.append((i, start_ms, end_ms, audio, temp_dir))
        
        results = []
        
        # Use ThreadPoolExecutor for parallelism
        # max_workers=3 allows 3 chunks to be transcribed EXACTLY at the same time.
        # If you have a strong GPU (8GB+ VRAM), you can increase this to 4 or 5.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_chunk = {executor.submit(_process_chunk, arg): arg[0] for arg in chunk_args}
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    idx, text = future.result()
                    results.append((idx, text))
                except Exception as exc:
                    logger.error(f"Chunk {chunk_idx} generated an exception: {exc}")

    # Sort results by index to maintain order
    results.sort(key=lambda x: x[0])
    
    # Combine text
    final_text = " ".join([r[1] for r in results])
    return final_text