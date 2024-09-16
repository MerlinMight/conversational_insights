# data_ingestion.py
import ffmpeg
import os
import subprocess
from transcription import transcribe_audio  # Import the transcription function

def extract_audio_from_video(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path).run()
    return audio_path

def convert_audio_to_wav(file_path, output_path):
    ffmpeg.input(file_path).output(output_path).run()
    return output_path

def load_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.mp3', '.wav']:
        if file_ext != '.wav':
            audio_path = file_path.replace(file_ext, '.wav')
            convert_audio_to_wav(file_path, audio_path)
            return audio_path
        return file_path
    
    elif file_ext == '.mp4':
        audio_path = file_path.replace('.mp4', '.wav')
        extract_audio_from_video(file_path, audio_path)
        return audio_path
    
    else:
        raise ValueError("Unsupported file format")
    


def extract_audio_from_video(video_path, audio_path):
    """
    Extract audio from a video file using the ffmpeg command-line tool.
    
    Arguments:
    video_path -- path to the video file.
    audio_path -- path where the audio file will be saved.
    """
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        audio_path
    ]
    subprocess.run(command, check=True)


def transcribe_audio_file(audio_path):
    """
    Transcribe the audio file at the given path using the transcription function.
    """
    transcription = transcribe_audio(audio_path)
    return transcription
