from transformers import pipeline, MarianMTModel, MarianTokenizer

# Load the Pegasus summarization model
summarizer = pipeline("summarization", model="google/pegasus-xsum")

def generate_summary(transcription):
    """
    Generate a summary of the transcription using Pegasus.
    
    Arguments:
    transcription -- the transcribed text.
    
    Returns:
    summary -- the summary of the transcription.
    """
    summary = summarizer(transcription, max_length=150, min_length=30, do_sample=False, temperature=0.7, top_p=0.9, repetition_penalty=1.2)
    return summary[0]['summary_text']

# Load the question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_question(context, question):
    """
    Answer a question about the context using a QA model.
    
    Arguments:
    context -- the text of the transcription.
    question -- the question to be answered.
    
    Returns:
    answer -- the answer to the question.
    """
    result = qa_pipeline({"question": question, "context": context})
    return result['answer']

# Load models for sentiment analysis and translation
sentiment_analyzer = pipeline("sentiment-analysis")
intent_recognizer = pipeline("zero-shot-classification")
translation_model_name = "Helsinki-NLP/opus-mt-en-de"  # English to German
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

def analyze_sentiment(text):
    """
    Analyze the sentiment of the text.
    
    Arguments:
    text -- the input text.
    
    Returns:
    sentiment_label -- the sentiment label (e.g., POSITIVE, NEGATIVE).
    sentiment_score -- the sentiment score.
    """
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

def recognize_intent(text, candidate_labels):
    """
    Recognize the intent of the text using zero-shot classification.
    
    Arguments:
    text -- the input text.
    candidate_labels -- list of possible intents.
    
    Returns:
    intent -- the recognized intent.
    """
    result = intent_recognizer(text, candidate_labels=candidate_labels)
    return result['labels'][0], result['scores'][0]

def translate_text(text, src_lang="en", tgt_lang="de"):
    """
    Translate the text using MarianMT model.
    
    Arguments:
    text -- the input text.
    src_lang -- source language code.
    tgt_lang -- target language code.
    
    Returns:
    translated_text -- the translated text.
    """
    inputs = translation_tokenizer.encode(text, return_tensors="pt")
    translated = translation_model.generate(inputs, max_length=100)
    translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def generate_report():
    """
    Generate a report based on the analysis.
    
    Returns:
    report -- the generated report.
    """
    transcription = """
Good morning. Oh, hello, good morning. Can I help you? Yes, you can. Actually, I'm looking for a present for my brother. 
Right, for your brother. Yes. What about a t-shirt? No, not a t-shirt. How about a denim jacket? Oh, yes. Actually, that's lovely. 
I like that. What size does he take? He takes a small. Okay. How about this? Great. That's lovely. I'll take that one. It's 50 euros. 
Is there a discount? There is today and it's 40 euros with the discount. Fantastic. I'll take it. Super. Thank you. Would you like it gift wrapped? 
No, thank you. I would like to do it myself. Okay. So with the discount, that's 40 euros. 40 euros? Yes. Here's 50. 
Thank you. And here's your 10 change. Thank you very much. You're welcome. And then here's your jacket. Thank you. Bye-bye. 
Bye-bye. Have a good day. Thank you. Bye-bye. Thank you.
"""

    question = "for whom the jacket is brought for?"

    sentiment_label, sentiment_score = analyze_sentiment(transcription)
    intent, intent_score = recognize_intent(transcription, ["complaint", "inquiry", "request", "feedback"])
    translated_text = translate_text(transcription)
    answer = answer_question(transcription, question)

    report = {
        "Sentiment Label": sentiment_label,
        "Sentiment Score": sentiment_score,
        "Intent": intent,
        "Translated Text": translated_text,
        "Answer": answer,
        "Transcription": transcription  # Add transcription to the report
    }
    return report
