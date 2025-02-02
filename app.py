import torch
import string
import unicodedata
import torch.nn as nn
from flask import Flask, render_template, request
from datasets import load_dataset

# Initialize Flask app
app = Flask(__name__)

# Load the emotion dataset
dataset = load_dataset("emotion")

# Vocabulary from training phase
all_letters = string.ascii_letters + " .,;!?'-"
alphabet = {char: idx for idx, char in enumerate(all_letters)}

# Emotion labels (Custom mapping: 0 - sadness, 1 - joy, etc.)
emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Model definition
class LastTimeStep(nn.Module):
    def __init__(self, rnn_layers, bidirectional):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional

    def forward(self, rnn_output):
        # rnn_output is a tuple: (output, hidden_state)
        output, hidden_state = rnn_output
        return output[:, -1, :]  # Extract the last timestep from the output


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Embedding(len(all_letters), 64),
    nn.RNN(64, 128, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2),
    LastTimeStep(rnn_layers=3, bidirectional=True),
    nn.Linear(128*2, len(emotion_map)),
)

# Load model to CPU if CUDA is not available
model.load_state_dict(torch.load('thirteenmodel.pth', map_location=torch.device('cpu')))

model.to(device)
model.eval()

# Function to preprocess text
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)

def classify_text(text):
    text = unicodeToAscii(text).lower()
    text_vec = torch.zeros(len(text), dtype=torch.long)
    for idx, char in enumerate(text):
        text_vec[idx] = alphabet.get(char, 0)  # Default to 0 if char not in vocab
    text_vec = text_vec.unsqueeze(0).to(device)  # Add batch dimension

    # Get prediction from model
    with torch.no_grad():
        output = model(text_vec)
        _, predicted = torch.max(output, 1)

    # Map the label index to emotion using the emotion_map
    predicted_emotion_index = predicted.cpu().numpy()[0]
    emotion = emotion_map.get(predicted_emotion_index, "Unknown")  # Default to "Unknown" if not found
    return emotion


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for handling form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get text from the form
        user_input = request.form['text']
        emotion = classify_text(user_input)
        return render_template('index.html', emotion=emotion, user_input=user_input)


if __name__ == "__main__":
    app.run(debug=True)









