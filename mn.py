API_KEY = 'AIzaSyCh7lSEAnbl6mGE3YTAd-t4nILaoT2BY5I'
import cv2

from flask import Flask, render_template, Response, jsonify, request
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import pytesseract
import threading
import speech_recognition as sr
import google.generativeai as genai
import string
import time

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


from PIL import Image
import cv2


def sendToAi(modelgemini, canvas, voice_text):
    global ai_response
    pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


    input_payload = {
        "text": voice_text,
        "image": pil_image
    }
    response = modelgemini.generate_content([voice_text, pil_image])
    print(response.text)
    ai_response = response.text

brush_color = (255, 255, 0)
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

flag=0

def listen_to_voice():
    global voice_text
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Optional: calibrate for noise
        print("Listening for voice...")
        while True:
            if flag == 0:
                print("Voice recognition stopped.")
                break
            try:
                # print("Listening for voice...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # Increase timeout
                recognized_text = recognizer.recognize_google(audio)
                print(f"Recognized Voice: {recognized_text}")
                voice_text = "find the are of circle"
                with lock:
                    voice_text = recognized_text

            except sr.UnknownValueError:
                print("Could not understand the audio.")
                continue
            except sr.RequestError as e:
                print(f"API error: {e}")
                continue
            except sr.WaitTimeoutError:
                print("Timeout: No voice detected.")

voice_thread = threading.Thread(target=listen_to_voice)
voice_thread.start()


def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    return tokens


import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

with open("math.txt", 'r', encoding='utf-8') as myfile:
    mytext = myfile.read()

mytokenizer = Tokenizer()
mytokenizer.fit_on_texts([mytext])
total_words = len(mytokenizer.word_index) + 1

my_input_sequences = []
for line in mytext.split('\n'):
    token_list = mytokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        my_n_gram_sequence = token_list[:i + 1]
        my_input_sequences.append(my_n_gram_sequence)

max_sequence_len = max([len(seq) for seq in my_input_sequences])

input_sequences = np.array(pad_sequences(my_input_sequences, maxlen=max_sequence_len, padding='pre'))

model = load_model('ac.h5')
def predict_next_words(model,input_text, predict_next_words=3):
    token_list = mytokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    reverse_word_index = {index: word for word, index in mytokenizer.word_index.items()}

    for _ in range(predict_next_words):
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]

        output_word = reverse_word_index.get(predicted_index, "")

        if not output_word:
            break

        input_text += " " + output_word

        token_list = pad_sequences([mytokenizer.texts_to_sequences([input_text])[0]], maxlen=max_sequence_len - 1,
                                   padding='pre')

    return input_text

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

            cv2.putText(canvas, line, (x, y + y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)
            y_offset += text_height + 5
            line = word


    if line:
        cv2.putText(canvas, line, (x, y + y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)


def generate_frames():
    prev_text=""
    global brush_color, brush_size, voice_text, ai_response,model_responce
    prev_pos, canvas = None, None
    max_text_width = 900
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
            clean_canvas = np.zeros_like(img)

        if fingers:
            prev_pos, canvas = draw(canvas, (fingers, lmList), prev_pos, brush_color, brush_size)

        if prev_text and voice_text and prev_text != voice_text:
            voice_text=prev_text+" "+voice_text

        if voice_text:
            draw_text_multiline(canvas, voice_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                                max_text_width)

        if voice_text and not model_responce:
            model_responce = predict_next_words(model,voice_text, predict_next_words=6)


        if model_responce:
            draw_text_multiline(canvas, model_responce, (50, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,

                                (169, 169, 169), 2, max_text_width)

        current_time = time.time()
        if fingers and fingers.count(1) == 4  and (current_time - last_combined_time >= 7):
            combined_text = f"{model_responce}".strip()
            voice_text = combined_text
            combined_text =""
            model_responce = ""
            canvas = clean_canvas.copy()
            draw_text_multiline(canvas, voice_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                                max_text_width)

            last_combined_time = current_time

        if fingers == [0, 0, 1, 1, 1] and (current_time - last_combined_time >= 7):
            model_responce =""
            canvas = np.zeros_like(img)
            canvas = clean_canvas.copy()
            draw_text_multiline(canvas, voice_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                                max_text_width)

            last_combined_time = current_time
        prev_text = voice_text
        if fingers == [1, 1, 1, 1, 1]:
            voice_text = ""
            model_responce = ""
            prev_text = ""
            combined_text = ""
            canvas = np.zeros_like(img)
            clean_canvas = np.zeros_like(img)


        combined_img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        _, buffer = cv2.imencode('.jpg', combined_img)
        frame = buffer.tobytes()

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jepg\r\n\r\n' + frame + b'\r\n')
# -------------------------------------------------------------------------------------------------






import subprocess
import datetime
import webbrowser
import yt_dlp
import wikipedia
import  threading
import pyttsx3
import speech_recognition as sr
import win32com.client as win32
import pyautogui
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename

recognizer = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
rate = engine.getProperty('rate')
engine.setProperty('rate', 150)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


from gtts import gTTS
from playsound import playsound
import uuid

from time import sleep

def speak(text):
    print(text)

    def _speak():
        try:
            # Create a unique file name to avoid conflicts
            temp_filename = f"temp_{uuid.uuid4().hex}.mp3"

            # Convert text to speech
            tts = gTTS(text=text, lang='en')
            tts.save(temp_filename)

            # Play the speech
            playsound(temp_filename)

            # Wait a bit before deleting the file to ensure it has played
            sleep(2)

            # Remove the temporary file
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error in TTS: {e}")
    # _speak()

    # Start the speaking process in a new thread
    tts_thread = threading.Thread(target=_speak)
    tts_thread.daemon = True
    tts_thread.start()

def get_command():
    with sr.Microphone() as source:
        print("Listening for command...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

        try:
            print("Recognizing command...")
            command = recognizer.recognize_google(audio).lower()
            print(f"User said: {command}")
            return command

        except sr.UnknownValueError:
            speak("Sorry, I did not understand that.")
        except sr.RequestError:
            speak("Sorry, my speech service is down.")
        return ""

def listen():
    with sr.Microphone() as source:
        print("Listening for wake word...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Adjust faster
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # Limit listening time

        try:
            print("Recognizing wake word...")
            command = recognizer.recognize_google(audio).lower()
            print(f"Wake word detected: {command}")

            if "nova" in command:
                print("How can I help you?")
                speak("how can I help")
                # threading.Thread(target=speak("How can I help you?")).start()
                return get_command()

        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            speak("Sorry, my speech service is down.")
        return ""
def process_command(command):
    if "what is the time" in command or "what's the time" in command:
        current_time = datetime.datetime.now().strftime("%H:%M")
        speak(f"The time is {current_time}")

    elif "can you search for" in command and "in youtube" in command:
        search = command.replace("can you search for", "").replace("in youtube", "").strip()
        search_in_yt(search)

    elif "play" in command or "can you play" in command:
        song = command.replace("play", "").strip()
        play_youtube_music(song)

    elif "open calculator" in command:
        open_calculator()

    elif "pause" in command:
        pause_video()

    elif "open notepad" in command:
        type_in_notepad()

    elif "search in wikipedia" in command or "search on wikipedia":
        speak("What do you want to search in Wikipedia?")
        query = get_command().lower()
        if query:
            result = search_on_wikipedia(query)
            speak(f"According to Wikipedia: {result}")

    elif "stop" in command or "exit" in command:
        speak("Goodbye!")
        return False

    else:
        speak("I'm sorry, I can't perform that task yet.")

    return True



def main():
    try:
        speak("Hello, I am Nova! Say 'Hey Nova!' to activate me.")
        while flag == 0:
            if flag==0:
                command = listen()

                print(command)
                if command:
                    if not process_command(command):
                        break
    except KeyboardInterrupt:
        speak("Goodbye! Program interrupted.")
    except Exception as e:
        print(f"Error in main loop: {e}")
        speak("An unexpected error occurred. Exiting now.")
    finally:
        speak("Shutting down Nova.")


def listen_for_command():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening for command...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        try:
            command = recognizer.recognize_google(audio).lower()
            print("Command:", command)
            return command
        except sr.UnknownValueError:
            speak("Sorry, I didn't understand.")
        except sr.RequestError:
            speak("Network error.")
    return ""

def perform_calculator_operation(command, operator_key):
    try:
        words = command.split()
        numbers = [word for word in words if word.replace('.', '', 1).isdigit()]
        if len(numbers) == 2:
            calculator_window = pyautogui.getWindowsWithTitle("Calculator")
            if calculator_window:
                calculator_window[0].activate()

            pyautogui.typewrite(numbers[0], interval=0)
            pyautogui.press(operator_key)
            pyautogui.typewrite(numbers[1], interval=0)
            pyautogui.press("enter")
            speak("Operation performed. Check the calculator for the result.")
        else:
            speak("I need exactly two numbers to perform this operation.")
    except Exception as e:
        print(f"Error in calculator operation: {e}")
        speak("Sorry, I couldn't perform the operation.")

def open_calculator():
    speak("Opening Calculator. Please tell me the operation you want to perform.")
    os.startfile("calc.exe")

    while True:
        command = get_command()
        if command:
            if "add" in command or "plus" in command:
                perform_calculator_operation(command, "+")
            elif "subtract" in command or "minus" in command:
                perform_calculator_operation(command, "-")
            elif "multiply" in command or "times" in command:
                perform_calculator_operation(command, "*")
            elif "divide" in command or "by" in command:
                perform_calculator_operation(command, "/")
            elif "close calculator" in command or "exit calculator" in command:
                speak("Closing calculator.")


                try:

                    result = subprocess.run(['tasklist'], capture_output=True, text=True)
                    if 'calc.exe' in result.stdout:
                        subprocess.run(['taskkill', '/f', '/im', 'calc.exe'])
                        speak("Calculator closed.")
                    else:
                        speak("Calculator is not open.")
                except Exception as e:
                    print(f"Error: {e}")
                    speak("An error occurred while trying to close the calculator.")
                break
        else:
            speak("I didn't understand the operation. Please try again.")

def search_on_wikipedia(query):
    results = wikipedia.summary(query,sentences=2)
    return results

def search_in_yt(query):
    try:
        search_url = f"https://www.youtube.com/results?search_query={query}"
        speak(f"Here are your search results for {query} on YouTube.")
        webbrowser.open_new(search_url)
    except Exception as e:
        print("Error:", e)
        speak("Sorry, I couldn't search YouTube.")

def play_youtube_music(query):
    try:
        ydl_opts = {"format": "bestaudio/best", "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_result = ydl.extract_info(f"ytsearch:{query}", download=False)
            if search_result['entries']:
                video_url = search_result["entries"][0]["webpage_url"]
                speak(f"Playing the first result on YouTube.")
                webbrowser.open(video_url)
            else:
                speak("Sorry, no results found.")
    except Exception as e:
        print("Error:", e)
        speak("I couldn't play the music.")



def pause_video():
    speak("Pausing the video.")
    pyautogui.press('k')



def type_in_notepad():
    speak("Opening Notepad.")
    notepad_path = r"C:\Windows\System32\notepad.exe"
    os.startfile(notepad_path)
    speak("Notepad is now open. You can start speaking, and I will type your words. Say 'save it' to stop.")

    text_to_write = ''
    while True:
        command = listen_for_command()
        if command:
            if "save it" in command:
                speak("Saving the document and stopping.")
                break
            else:
                text_to_write += command + ' '
                pyautogui.typewrite(command + ' ', interval=0.05)  # Faster typing speed
                print(f"Typed: {command}")
        else:
            speak("You didn't say anything.")
            continue
def control_presentation(presentation, command):
    try:
        slides = presentation.Slides
        if "next slide" in command or "forward" in command:
            pyautogui.press('right')
            speak("Moving forward to the next slide.")
        elif "previous slide" in command or "back" in command:
            pyautogui.press('left')
            speak("Going back to the previous slide.")
        elif "start" in command or "presentation" in command:
            presentation.SlideShowSettings.Run()
            speak("Starting the slideshow.")
        elif "end presentation" in command or "stop" in command:
            presentation.SlideShowWindow.View.Exit()
            speak("Ending the slideshow.")
        elif "exit" in command or "quit" in command:
            speak("Exiting the program. Goodbye!")
            return "exit"
        else:
            speak("I didn't understand the command.")
    except Exception as e:
        print(f"Error controlling presentation: {e}")
        speak("An error occurred while controlling the presentation.")
    return "continue"


def open_presentation(file_path):
    try:
        powerpoint = win32.Dispatch("PowerPoint.Application")
        presentation = powerpoint.Presentations.Open(file_path)
        powerpoint.Visible = True
        print(f"Presentation '{file_path}' opened successfully.")
        speak("Presentation opened. Ready to start.")
        return presentation
    except Exception as e:
        speak(f"Error opening the presentation: {str(e)}")
        print(f"Error: {str(e)}")
        return None

UPLOAD_FOLDER = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload_file_post():

    ppt_file = request.files.get('file')


    if not ppt_file:
        return jsonify({"error": "No file uploaded."}), 400


    if not ppt_file.filename.endswith('.pptx'):
        return jsonify({"error": "Invalid file type. Please upload a .pptx file."}), 400


    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(ppt_file.filename))
    ppt_file.save(filepath)


    try:
        if os.name == 'nt':
            open_presentation(filepath)
            # subprocess.Popen(['start', filepath], shell=True)
            presentation = open_presentation(filepath)
            if presentation is None:
                return jsonify({"error": "Failed to open the presentation."}), 500

            # Directly trigger control functionality
            speak("Presentation opened. Now starting control functionality.")
            while True:
                command = listen_for_command()
                if command:
                    action = control_presentation(presentation, command)
                    if action == "exit":
                        return jsonify({"message": "Presentation closed."})
                else:
                    speak("No command received. Please try again.")

        elif os.name == 'posix':
            subprocess.Popen(['open', filepath])
        return jsonify({"message": f"File uploaded and opened: {ppt_file.filename}"}), 200
    except Exception as e:
        return jsonify({"error": f"Error opening the file: {str(e)}"}), 500

@app.route('/control/<file>', methods=['GET'])
def control_presentation_view(file):
    # Construct file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        speak("Opening the presentation.")
        presentation = open_presentation(file_path)
        if presentation is None:
            return jsonify(
                {"error": "Failed to open the presentation. Please check the file format or PowerPoint setup."}), 500

        speak("You can now give voice commands to control the presentation.")

        while True:
            command = listen_for_command()
            if command:
                action = control_presentation(presentation, command)
                if action == "exit":
                    return jsonify({"message": "Presentation closed."})
            else:
                speak("No command received. Please try again.")

    except Exception as e:
        print(f"Error controlling the presentation: {e}")
        return jsonify({"error": str(e)}), 500






# -----------------------------------------------------------------------------------------
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

print(flag)
@app.route('/toggle_voice', methods=['POST'])
def toggle_voice():
    global flag, voice_thread

    data = request.json
    flag = data.get('flag', 0)
    status = "enabled" if flag == 1 else "disabled"
    print(flag)
    if flag == 1:
        voice_thread = threading.Thread(target=listen_to_voice)
        voice_thread.start()
    else:
        flag = 0
        voice_thread = threading.Thread(target=main)
        voice_thread.start()
    return jsonify({"message": f"Voice recognition is {status}."})
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    voice_thread = threading.Thread(target=main)
    voice_thread.start()
    app.run(debug=True, threaded=True)
