from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Load the sentiment and emotion analysis pipelines
sentiment_model = pipeline("sentiment-analysis")
emotion_model = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

def analyze_sentiment_and_emotions():
    # Example transcription (use your actual transcription here)
    transcription = """
    Good morning.
    Oh, hello, good morning.
    Can I help you?
    Yes, you can. Actually, I'm looking for a present for my brother.
    Right, for your brother.
    Yes.
    What about a t-shirt?
    No, not a t-shirt.
    How about a denim jacket?
    Oh, yes. Actually, that's lovely. I like that.
    What size does he take?
    He takes a small.
    Okay. How about this?
    Great. That's lovely. I'll take that one.
    It's 50 euros.
    Is there a discount?
    There is today and it's 40 euros with the discount.
    Fantastic. I'll take it.
    Super. Thank you. Would you like it gift wrapped?
    No, thank you. I would like to do it myself.
    Okay. So with the discount, that's 40 euros.
    40 euros?
    Yes.
    Here's 50.
    Thank you. And here's your 10 change.
    Thank you very much.
    You're welcome. And then here's your jacket.
    Thank you.
    Bye-bye.
    Bye-bye.
    Have a good day.
    Thank you.
    Bye-bye.
    Thank you.
    """

    def assign_speaker_segments(transcription):
        lines = [line.strip() for line in transcription.strip().split('\n') if line.strip()]
        speaker_segments = [1 if i % 2 == 0 else 2 for i in range(len(lines))]
        return speaker_segments

    def analyze_sentiment_per_speaker(transcription, speaker_segments):
        sentences = [sentence.strip() for sentence in transcription.split('.') if sentence.strip()]
        speaker_sentences = defaultdict(list)
        num_sentences = len(sentences)
        if len(speaker_segments) < num_sentences:
            speaker_segments += [speaker_segments[-1]] * (num_sentences - len(speaker_segments))
        for i, sentence in enumerate(sentences):
            if i < len(speaker_segments):
                speaker = f'Speaker {speaker_segments[i]}'
                speaker_sentences[speaker].append(sentence)
        speaker_sentiments = defaultdict(list)
        for speaker, sentences in speaker_sentences.items():
            for sentence in sentences:
                if sentence:
                    sentiment_result = sentiment_model(sentence)[0]
                    speaker_sentiments[speaker].append(sentiment_result)
        return dict(speaker_sentiments)

    def categorize_sentiments(speaker_sentiments):
        sentiment_distribution = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        for speaker, sentiments in speaker_sentiments.items():
            for sentiment in sentiments:
                label = sentiment['label'].lower()
                if label == 'positive':
                    sentiment_distribution[speaker]['positive'] += 1
                elif label == 'negative':
                    sentiment_distribution[speaker]['negative'] += 1
                else:
                    sentiment_distribution[speaker]['neutral'] += 1
        return sentiment_distribution

    def analyze_emotion_per_speaker(transcription, speaker_segments):
        sentences = [sentence.strip() for sentence in transcription.split('.') if sentence.strip()]
        speaker_sentences = defaultdict(list)
        num_sentences = len(sentences)
        if len(speaker_segments) < num_sentences:
            speaker_segments += [speaker_segments[-1]] * (num_sentences - len(speaker_segments))
        for i, sentence in enumerate(sentences):
            if i < len(speaker_segments):
                speaker = f'Speaker {speaker_segments[i]}'
                speaker_sentences[speaker].append(sentence)
        speaker_emotions = defaultdict(list)
        for speaker, sentences in speaker_sentences.items():
            for sentence in sentences:
                if sentence:
                    emotion_result = emotion_model(sentence)[0]
                    speaker_emotions[speaker].append(emotion_result)
        return dict(speaker_emotions)

    def categorize_emotions(speaker_emotions):
        emotion_distribution = defaultdict(lambda: defaultdict(int))
        for speaker, emotions in speaker_emotions.items():
            for emotion in emotions:
                label = emotion['label']
                emotion_distribution[speaker][label] += 1
        return emotion_distribution

    def save_sentiment_plot(sentiment_distribution):
        data = []
        for speaker, sentiments in sentiment_distribution.items():
            for sentiment, count in sentiments.items():
                data.append({'Speaker': speaker, 'Sentiment': sentiment.capitalize(), 'Count': count})
        df = pd.DataFrame(data)
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Speaker', y='Count', hue='Sentiment', data=df)
        plt.title('Sentiment Distribution Per Speaker')
        plt.xlabel('Speaker')
        plt.ylabel('Count')
        plt.legend(title='Sentiment')
        plt.savefig('static/sentiment_plot.png')
        plt.close()

    def save_emotion_plots(emotion_distribution):
        if not os.path.exists('static/emotion_plots'):
            os.makedirs('static/emotion_plots')
        for speaker, emotions in emotion_distribution.items():
            data = [{'Emotion': emotion, 'Count': count} for emotion, count in emotions.items()]
            df = pd.DataFrame(data)
            sns.set(style="whitegrid")
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Emotion', y='Count', data=df)
            plt.title(f'Emotion Distribution for {speaker}')
            plt.xlabel('Emotion')
            plt.ylabel('Count')
            plt.savefig(f'static/emotion_plots/{speaker}_emotion_plot.png')
            plt.close()

    speaker_segments = assign_speaker_segments(transcription)
    speaker_sentiments = analyze_sentiment_per_speaker(transcription, speaker_segments)
    sentiment_distribution = categorize_sentiments(speaker_sentiments)
    save_sentiment_plot(sentiment_distribution)
    speaker_emotions = analyze_emotion_per_speaker(transcription, speaker_segments)
    emotion_distribution = categorize_emotions(speaker_emotions)
    save_emotion_plots(emotion_distribution)

    return sentiment_distribution, emotion_distribution
