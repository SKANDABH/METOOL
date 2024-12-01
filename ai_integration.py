import google.generativeai as genai
from PIL import Image
import threading
import torch
import cv2
lock = threading.Lock()

def sendToAi(model, canvas, voice_text):
    global ai_response
    if voice_text and canvas is not None:
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))  # Convert canvas to PIL Image
        response = model.generate_content(["Solve", voice_text, pil_image])  # Send voice text and image to Gemini AI
        with lock:
            ai_response = response.text

            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(ai_response + "\n")
    else:
        ai_response = ""

def generate_text(model, start_string, word_to_index, index_to_word, length=100, temperature=1.0):
    model.eval()
    cleaned_start_string = clean_text(start_string)
    input_indices = [word_to_index.get(word, word_to_index[' ']) for word in cleaned_start_string]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_string

    with torch.no_grad():
        for _ in range(length):
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output / temperature, dim=-1)
            predicted_index = torch.multinomial(probabilities, 1).item()

            predicted_char = index_to_word[predicted_index]
            generated_text += predicted_char

            input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[predicted_index]], device=device)), dim=1)

    return generated_text
