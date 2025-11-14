import whisper
import os

_model = None
_model_name = "tiny.en" # Use tiny for speed, or base.en

def get_whisper_model():
    """
    Loads the Whisper model into memory (singleton pattern).
    """
    global _model
    if _model is None:
        print(f"Loading Whisper model: {_model_name}...")
        _model = whisper.load_model(_model_name)
        print("Whisper model loaded.")
    return _model

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the audio file at the given path.
    This is a blocking, CPU-intensive operation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at {file_path}")
        
    model = get_whisper_model()
    
    # Run the transcription
    result = model.transcribe(file_path)
    
    return result.get("text", "")