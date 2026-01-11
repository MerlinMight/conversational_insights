from transformers import pipeline

transcription = """
Good morning. Oh, hello, good morning. Can I help you? Yes, you can. Actually, I'm looking for a present for my brother. 
Right, for your brother. Yes. What about a t-shirt? No, not a t-shirt. How about a denim jacket? Oh, yes. Actually, that's lovely. 
I like that. What size does he take? He takes a small. Okay. How about this? Great. That's lovely. I'll take that one. It's 50 euros. 
Is there a discount? There is today and it's 40 euros with the discount. Fantastic. I'll take it. Super. Thank you. Would you like it gift wrapped? 
No, thank you. I would like to do it myself. Okay. So with the discount, that's 40 euros. 40 euros? Yes. Here's 50. 
Thank you. And here's your 10 change. Thank you very much. You're welcome. And then here's your jacket. Thank you. Bye-bye. 
Bye-bye. Have a good day. Thank you. Bye-bye. Thank you.
"""
question = 'for whom the jacket is brought for?'

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
    result = qa_pipeline(question=question, context=context)
    return result['answer']
