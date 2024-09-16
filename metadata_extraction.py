import pydub
from pydub import AudioSegment
from pyAudioAnalysis import audioSegmentation as aS
import librosa
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import langdetect
import nltk
from transcription import transcribe_audio  # Import the transcription function

def extract_metadata(audio_path):
    """
    Extract metadata from an audio file including number of speakers, duration of speech per speaker,
    language, number of words spoken, frequency of certain keywords, and additional audio features.
    """
    # Load audio file with pydub
    audio = AudioSegment.from_wav(audio_path)
    duration = len(audio) / 1000.0  # Duration in seconds
    sample_rate = audio.frame_rate
    channels = audio.channels

    # Speaker diarization with a range of possible speakers
    possible_speakers = range(2, 10)  # Adjust range as needed
    best_segments = None
    best_num_speakers = 0

    for n_speakers in possible_speakers:
        try:
            segments, _, _ = aS.speaker_diarization(audio_path, n_speakers=n_speakers)
            if len(segments) > 0:
                best_segments = segments
                best_num_speakers = n_speakers
                break  # If valid segments are found, break the loop
        except Exception as e:
            continue  # Try the next number of speakers

    if best_segments is None:
        raise ValueError("No valid segments were returned from the diarization process.")

    # Determine the number of unique speakers
    unique_speakers = set(best_segments)
    num_speakers = len(unique_speakers)
    
    # Initialize speaker durations
    speaker_durations = defaultdict(float)
    
    if len(best_segments) > 0:
        start_time = 0
        current_speaker = best_segments[0]
        for i in range(1, len(best_segments)):
            if best_segments[i] != current_speaker:
                end_time = i * (duration / len(best_segments))
                speaker_durations[current_speaker] += (end_time - start_time)
                start_time = end_time
                current_speaker = best_segments[i]
        # Handle the last segment
        end_time = duration
        speaker_durations[current_speaker] += (end_time - start_time)
    
    # Transcribe audio using Whisper model
    transcription = transcribe_audio(audio_path)
    
    # Detect language
    try:
        language = langdetect.detect(transcription)
    except langdetect.lang_detect_exception.LangDetectException:
        language = 'unknown'

    # Tokenize and count words
    words = word_tokenize(transcription.lower())
    num_words = len([word for word in words if word.isalpha()])
    
    # Extract keywords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    word_freq = Counter(filtered_words)
    most_common_keywords = word_freq.most_common(10)

    # Load audio using librosa
    y, sr = librosa.load(audio_path, sr=None)

    # Extract additional features using librosa
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms_energy = np.mean(librosa.feature.rms(y=y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma_features = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    # Additional metadata
    metadata = {
        'duration(secs)': duration,
        'sample_rate': sample_rate,
        'channels': channels,
        'speaker_durations(secs)': {f'Speaker {i+1}': dur for i, dur in enumerate(speaker_durations.values())},
        'number_of_speakers': num_speakers,
        'language': language,
        'num_words': num_words,
        'most_common_keywords': most_common_keywords,
        'tempo': tempo,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'zero_crossing_rate': zero_crossing_rate,
        'rms_energy': rms_energy,
        'mfccs': mfccs.tolist(),  # Convert to list for better readability
        'chroma_features': chroma_features.tolist()  # Convert to list for better readability
    }

    return metadata
