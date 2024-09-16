# app.py

from flask import Flask, request, render_template, url_for
from module.data_ingestion import load_file, transcribe_audio_file
from topic_extraction import predict_topics_from_transcription, predefined_topics
from metadata_extraction import extract_metadata  # Import the metadata extraction function
from sentimental_analysis import analyze_sentiment_and_emotions
from insight_generation import generate_insights

import os



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = 'uploaded_file' + os.path.splitext(file.filename)[1]
        file.save(file_path)
        
        # Process the file
        audio_path = load_file(file_path)
        transcription = transcribe_audio_file(audio_path)

        # Extract metadata
        metadata = extract_metadata(audio_path)

        # Extract topics from transcription
        topics = predict_topics_from_transcription(transcription, predefined_topics)

        # Perform analysis
        sentiment_distribution, emotion_distribution = analyze_sentiment_and_emotions()
    
        # Paths to the saved plots
        sentiment_plot = url_for('static', filename='sentiment_plot.png')
        emotion_plots = [url_for('static', filename=f'emotion_plots/{filename}') for filename in os.listdir('static/emotion_plots')]


        # Call the generate_insights function to get the transcription and insights
        transcription, conversation_type, satisfaction_score, complaints, suggestions = generate_insights(transcription, sentiment_distribution)

 

        return render_template(
            'results.html', 
            transcription=transcription,
            metadata=metadata, 
            topics=topics,
            sentiment_plot=sentiment_plot, 
            emotion_plots=emotion_plots,
            conversation_type=conversation_type,
            satisfaction_score=satisfaction_score,
            complaints=complaints,
            suggestions=suggestions 
        )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)






