# transcription.py
import whisper  # Make sure you have the OpenAI Whisper library installed
import gc

def transcribe_audio(audio_path):
    """
    Transcribe audio file using OpenAI's Whisper model.
    """
    model = whisper.load_model("base")  # You can choose a different model size if needed
    
    # Perform transcription
    result = model.transcribe(audio_path)
    
    return result['text']


gc.collect()