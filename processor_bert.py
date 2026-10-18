from pathlib import Path
from sentence_transformers import SentenceTransformer
import joblib
# Load the sentence transformer model to compute log_message embeddings
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
# Load the saved classification model
model_path = Path(__file__).resolve().parent / "training" / "models" / "log_classifier.joblib"
classifier_model = joblib.load(model_path)
def classify_with_bert(log_message):
    '''Classify log messages using a BERT model. This function takes a log message as input and returns the predicted label based on the BERT model's output. If the model cannot confidently classify the message, it returns None.'''
    # Compute embeddings for the log_message
    message_embedding = transformer_model.encode(log_message)
    probabilities = classifier_model.predict_proba([message_embedding])[0]
    predicted_class = "Unknown"
    if max(probabilities) < 0.5:
        return "Unclassified"
        predicted_class = classifier_model.predict([message_embedding])[0]
    # Perform classification using the loaded model
    predicted_class = classifier_model.predict([message_embedding])[0]
    # Return the predicted class
    return predicted_class
if __name__ == "__main__":
    logs = [
        "alpha. osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE 404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        Label = classify_with_bert(log)
        print(log, "->", Label)
