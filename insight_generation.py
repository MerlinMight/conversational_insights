import pandas as pd
from collections import defaultdict

common_complaints_list = [
    "delivery", "customer service", "refund", "pricing", "product quality",
    "warranty", "out of stock", "return policy", "payment issues",
    "shipping delay", "damaged goods", "order tracking", "incorrect order",
    "product availability", "exchange policy", "discount issues",
    "loyalty program", "billing problems", "size or fit issues",
    "packaging issues", "communication delay"
]

def detect_conversation_type(num_speakers):
    if num_speakers == 2:
        return "customer_agent_conversation"
    elif num_speakers > 2:
        return "panel_discussion"
    else:
        return "unknown"

def calculate_customer_satisfaction(sentiment_distribution):
    total_positive = sum(s['positive'] for s in sentiment_distribution.values())
    total_negative = sum(s['negative'] for s in sentiment_distribution.values())
    total_sentiments = total_positive + total_negative

    if total_sentiments == 0:
        return 0

    satisfaction_score = (total_positive / total_sentiments) * 100
    return satisfaction_score

def generate_suggestions(sentiment_distribution):
    positive_sentiments = sum(s['positive'] for s in sentiment_distribution.values())
    negative_sentiments = sum(s['negative'] for s in sentiment_distribution.values())
    
    suggestions = []
    if positive_sentiments > negative_sentiments:
        suggestions.append("Continue to provide excellent service and consider additional perks for loyal customers.")
    if negative_sentiments > positive_sentiments:
        suggestions.append("Address any negative feedback promptly and work on improving the customer experience.")
    if positive_sentiments == 0 and negative_sentiments == 0:
        suggestions.append("Maintain current service standards but explore areas for potential enhancement.")
    
    return suggestions

def generate_improvement_suggestions(satisfaction_score):
    improvement_suggestions = []
    if satisfaction_score < 50:
        improvement_suggestions.append("Implement customer feedback mechanisms to identify specific areas for improvement.")
        improvement_suggestions.append("Enhance training for customer service representatives to better handle customer issues.")
        improvement_suggestions.append("Review and streamline complaint resolution processes to reduce response times.")
        improvement_suggestions.append("Consider offering compensatory measures, such as discounts or incentives, to dissatisfied customers.")
    
    return improvement_suggestions

def capture_complaints(transcription):
    complaints = []
    sentences = [sentence.strip() for sentence in transcription.split('.') if sentence.strip()]
    
    for sentence in sentences:
        for keyword in common_complaints_list:
            if keyword in sentence.lower():
                complaints.append(sentence)
                break
    
    return complaints

# Function to generate insights
def generate_insights(transcription, sentiment_distribution):
    
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
    # Example sentiment distribution for demo purposes
    sentiment_distribution = {
        "customer": {"positive": 5, "negative": 2},
        "agent": {"positive": 6, "negative": 1}
    }


    # Determine the number of speakers based on the sentiment distribution dictionary
    num_speakers = len(sentiment_distribution)

    # Detect the conversation type
    conversation_type = detect_conversation_type(num_speakers)
    print(f"Conversation Type: {conversation_type}")

    # Capture complaints
    complaints = capture_complaints(transcription)
    if complaints:
        print("Complaints:")
        for complaint in complaints:
             print(f"- {complaint}")
    else:
         print("No complaints!! Everything is Good.")

    # Calculate customer satisfaction score (only applicable for customer-agent conversations)
    if conversation_type == "customer_agent_conversation":
        satisfaction_score = calculate_customer_satisfaction(sentiment_distribution)
        print(f"Customer Satisfaction Score: {satisfaction_score:.2f}%")

    # Generate suggestions based on sentiment distribution
    suggestions = generate_suggestions(sentiment_distribution)
    print("Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")

    # Generate additional suggestions if satisfaction score is below 50%
    improvement_suggestions = generate_improvement_suggestions(satisfaction_score)
    if improvement_suggestions:
        print("Additional Improvement Suggestions:")
        for suggestion in improvement_suggestions:
            print(f"- {suggestion}")
    else:
        print("Suggestions are only applicable for customer-agent conversations.")

    
    num_speakers = len(sentiment_distribution)
    conversation_type = detect_conversation_type(num_speakers)
    satisfaction_score = calculate_customer_satisfaction(sentiment_distribution)
    complaints = capture_complaints(transcription)
    suggestions = generate_suggestions(sentiment_distribution)
    
    return transcription, conversation_type, satisfaction_score, complaints, suggestions
