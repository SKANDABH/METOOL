API_KEY = 'AIzaSyCh7lSEAnbl6mGE3YTAd-t4nILaoT2BY5I'
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import pytesseract
import threading
import speech_recognition as sr
import google.generativeai as genai
import torch
import torch.nn as nn
import string
import time

from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import nltk
# nltk.download('punkt_tab')
# nltk.download('punkt')  # This is the main resource for tokenization
# nltk.download('stopwords')  # For common stop words, if needed


app = Flask(__name__)


genai.configure(api_key=API_KEY)
modelgemini = genai.GenerativeModel('gemini-1.5-flash')

# Initialize hand detector
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

canvas = None
prev_pos = None
voice_text = ""
ai_response = ""
# ai_response = "hiii"
lock = threading.Lock()

recognizer = sr.Recognizer()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model_responce=""

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        return fingers, hand['lmList']
    return None, []


import base64
from PIL import Image
import cv2
import numpy as np

def sendToAi(modelgemini, canvas, voice_text):
    global ai_response
    pil_image = Image.fromarray(
        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))  # Convert canvas to RGB and then to PIL Image
    response = modelgemini.generate_content([voice_text, pil_image])
    print(response.text)
    ai_response = response.text










brush_color = (255, 0, 0)
brush_size = 5


def draw(canvas, info, prev_pos, brush_color, brush_size):
    fingers, lmList = info
    current_position = None

    if fingers == [0, 1, 0, 0, 0]:
        if lmList and len(lmList) > 8:
            current_position = lmList[8][0:2]
            if prev_pos is not None:
                cv2.line(canvas, prev_pos, current_position, brush_color, brush_size)
            prev_pos = current_position

    elif fingers == [1, 1, 0, 0, 1]:
        global voice_text
        sendToAi(modelgemini, canvas, voice_text)  # Send canvas and voice text

    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(canvas)

    return current_position, canvas

# Voice listening function
def listen_to_voice():
    global voice_text
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Optional: calibrate for noise
        print("Listening for voice...")
        while True:
            try:
                # print("Listening for voice...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # Increase timeout
                recognized_text = recognizer.recognize_google(audio)
                print(f"Recognized Voice: {recognized_text}")
                voice_text = "find the are of circle"
                with lock:
                    # voice_text = recognized_text
                    voice_text="find the are of circle"
            except sr.UnknownValueError:
                print("Could not understand the audio.")
                continue
            except sr.RequestError as e:
                print(f"API error: {e}")
                continue
            except sr.WaitTimeoutError:
                print("Timeout: No voice detected.")

# Start the voice thread
voice_thread = threading.Thread(target=listen_to_voice, daemon=True)
voice_thread.start()

# Text cleaning function for model input
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    return tokens

# Read and process text data for LSTM model
with open("math.txt", "r",encoding='utf-8') as al:
    data = al.read()

cleaned_data = clean_text(data)

# Create vocabulary
vocab = set(cleaned_data)
vocab.add(' ')  # Add space to vocabulary
vocab_size = len(vocab)
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}

with open("vocab.txt", "w",encoding='utf-8') as f:
    for word in vocab:
        f.write(word + "\n")

# Prepare text data for model training
text_as_int = [word_to_index.get(word, word_to_index[' ']) for word in cleaned_data]  # Handle unknown words
seq_length = 100
input_texts = []
output_texts = []

for i in range(len(text_as_int) - seq_length):
    input_texts.append(text_as_int[i:i + seq_length])
    output_texts.append(text_as_int[i + seq_length])

X = torch.tensor(input_texts, dtype=torch.long)
y = torch.tensor(output_texts, dtype=torch.long)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


batch_size = 64
dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_length):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


embed_size = 256
hidden_size = 256
model = LSTMModel(vocab_size, embed_size, hidden_size, seq_length)


state_dict = torch.load('modeltorchmath45.pth', map_location=torch.device('cpu'), weights_only=True)
current_state_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in current_state_dict and v.size() == current_state_dict[k].size()}
model.load_state_dict(filtered_state_dict, strict=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_text(model, start_string, length=4, temperature=1.0):
    model.eval()
    cleaned_start_string = clean_text(start_string)
    input_indices = [word_to_index.get(word, word_to_index[' ']) for word in cleaned_start_string]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

    generated_text =" "

    with torch.no_grad():
        for _ in range(length):
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output / temperature, dim=-1)
            predicted_index = torch.multinomial(probabilities, 1).item()

            predicted_word = index_to_word.get(predicted_index, ' ')
            generated_text += " " + predicted_word

            input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[predicted_index]], device=device)), dim=1)

    print(generated_text)
    return generated_text


def draw_text_multiline(canvas, text, position, font, font_scale, font_color, thickness, line_width):
    x, y = position
    words = text.split(' ')
    line = ""
    y_offset = 0

    for word in words:
        test_line = f"{line} {word}".strip()
        (text_width, text_height), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if text_width <= line_width:
            line = test_line
        else:
            # Draw the current line and start a new one
            cv2.putText(canvas, line, (x, y + y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)
            y_offset += text_height + 5
            line = word

    # Draw the last line
    if line:
        cv2.putText(canvas, line, (x, y + y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)


def generate_frames():
    global brush_color, brush_size, voice_text, ai_response,model_responce
    prev_pos, canvas = None, None
    max_text_width = 600
    y_offset = 50
    combined_text = ""
    last_combined_time = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        fingers, lmList = getHandInfo(img)
        if canvas is None or fingers == [1, 0, 0, 0, 1]:
            canvas = np.zeros_like(img)


        if voice_text:
            draw_text_multiline(canvas, voice_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                                max_text_width)
        if voice_text and not model_responce:
            model_responce = generate_text(model, voice_text)
            # model_responce="with diameter 5cm"

        text=voice_text+model_responce
        if model_responce:
            draw_text_multiline(canvas, model_responce, (50, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (169, 169, 169), 2, max_text_width)
        current_time = time.time()
        if fingers and fingers.count(1) == 4  and (current_time - last_combined_time >= 7):
            combined_text = f"{voice_text} {model_responce}".strip()
            voice_text = combined_text
            model_responce = ""
            last_combined_time = current_time
        elif fingers == [1, 1, 0, 0, 1]:
            sendToAi(modelgemini, canvas, text)
        if fingers == [1, 0, 0, 0, 1]:
            voice_text = ""
            model_response = ""
            combined_text = ""


        # if combined_text:
        #     draw_text_multiline(canvas, combined_text, (50, y_offset + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #                         brush_color, 2, max_text_width)


            # if fingers and fingers.count(1) == 4:
        #     combined_text = f"{voice_text} {model_response}".strip()
        #     voice_text = combined_text
        #     model_response = ""
            # if fingers:
        #     prev_pos, canvas = draw(canvas, (fingers, lmList), prev_pos, brush_color, brush_size)
        #     if voice_text and not model_responce:
        #
        #     if fingers.count(1) == 4:
        #         text_color = (169, 169, 169)
        #         if model_responce:
        #             draw_text_multiline(canvas, model_responce, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2,
        #                                 max_text_width)

        ret, buffer = cv2.imencode('.jpg', canvas)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/update_tool_settings', methods=['POST'])
def update_tool_settings():
    global brush_color, brush_size

    data = request.json
    brush_color = tuple(int(data['color'][i:i+2], 16) for i in (1, 3, 5))
    brush_size = int(data['brush_size'])

    return jsonify({"status": "success", "color": brush_color, "brush_size": brush_size})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/get_voice_input')
def get_voice_input():
    global voice_text
    global  ai_response
    print(ai_response)

    return jsonify({
        "ai_response": ai_response,
        "voice_text": voice_text
    })


# @app.route('/get_voice_input')
# def get_voice_input():
#
#     voice_text="Hi this is ai "
#
#


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
