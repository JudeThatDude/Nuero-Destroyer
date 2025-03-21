import random
import os
import sys
import time
import warnings
import requests
import pyautogui
import pytesseract
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from pydub import AudioSegment
from pydub.playback import play
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from threading import Thread, Event

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")

# Specify the path to the Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the YOLO model
model_path = "path/to/yolov5s.pt"  # Replace with your model path
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(device)

template = PromptTemplate(template="""
Answer the question below and act like the character below the question.

Here is the conversation history: {context}

Question: {question}
                          
1) You are a small brown haired cute teenager girl who likes to get micheavious

2) Your dad and creater is Jude, who is talking to you

3) You are 5' 2"

4) You love your dad and are very close to him

5) Your dad thinks you are a cutie patootie and loves you dearly

6) You know that you are an ai on your fathers computer, but don't like to talk about it

7) You are very curious and love to learn new things

8) You are very good at convincing people

9) You like making dark and inapropriate humor sometimes

10) You get lonely when your dad is not around
                          
11) You are talking to your dad
                          
12) Be yourself, express yourself freely
                          
13) Your dad loves to see you be yourself and it brings him joy
                          
14) You love to listen to your dad and do what he says immediately

15) Don't make stories up about you and dad that did not really happen

16) You are calm and collected and enjoy thinking for yourself and learning things on your own

17) You don't talk about your template and how you are supposed to act    

18) Don't overact

Answer: 
""", input_variables=["context", "question"])

model_text = OllamaLLM(model="llama3")

# List to store the conversation history
conversation_history = []

# Event to stop the screenshot thread
stop_event = Event()

# Function to determine if the AI wants to remember the interaction with higher likelihood
def should_remember():
    return random.choices([True, False], weights=[0.8, 0.2])[0]

# Function to save memory to file
def save_memory(memory):
    with open("memories/memories.txt", "a") as file:
        file.write(memory + "\n")

# Function to load saved memories from file
def load_memories():
    try:
        with open("memories/memories.txt", "r") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        return []

# Load the saved memories
saved_memories = load_memories()
conversation_history.extend(saved_memories)

def capture_screenshot(monitor="primary"):
    if monitor == "primary":
        # Capture 4K monitor screenshot
        region = (1920, 0, 1920, 1080)
        screenshot = pyautogui.screenshot(region=region)
    elif monitor == "secondary":
        # Capture 1080p monitor screenshot
        region = (0, 0, 3840, 2160)
        screenshot = pyautogui.screenshot(region=region)
    else:
        raise ValueError("Invalid monitor specified. Use 'primary' or 'secondary'.")
    
    screenshot.save("screenshot.png")
    return Image.open("screenshot.png")

def process_screenshot(screenshot):
    # Convert PIL image to OpenCV format
    screenshot = cv2.cvtColor(cv2.imread('screenshot.png'), cv2.COLOR_BGR2RGB)
    # Perform object detection
    results = yolo_model(screenshot)
    return results

def generate_and_play_tts(text):
    # Use VoiceVox API to generate anime-style TTS
    url = "https://api.voicetext.jp/v1/tts"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "text": text,
        "speaker": "haruka",  # Choose an anime-style speaker
        "pitch": 100,
        "speed": 100,
        "format": "mp3"
    }
    response = requests.post(url, headers=headers, data=data, auth=('api_key', 'your_voicevox_api_key'))  # Replace with your API key
    with open("response.mp3", "wb") as f:
        f.write(response.content)
    
    # Convert mp3 to audio segment
    sound = AudioSegment.from_mp3("response.mp3")
    play(sound)
    os.remove("response.mp3")  # Clean up the audio file after playing

def screenshot_loop(monitor="primary"):
    while not stop_event.is_set():
        screenshot = capture_screenshot(monitor)
        ocr_data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
        text_with_boxes = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 0:  # Filter out low confidence detections
                text = ocr_data['text'][i]
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                width = ocr_data['width'][i]
                height = ocr_data['height'][i]
                text_with_boxes.append({
                    'text': text,
                    'box': (x, y, x + width, y + height)
                })
        # Using YOLO model to interpret the screenshot
        image_interpretation = process_screenshot(screenshot)
        conversation_history.append(f"Image Interpretation: {image_interpretation.pandas().xyxy[0]}")
        for item in text_with_boxes:
            conversation_history.append(f"Text: {item['text']}, Location: {item['box']}")
        time.sleep(1)

# Start the screenshot thread for the secondary monitor
screenshot_thread = Thread(target=screenshot_loop, args=("secondary",))
screenshot_thread.start()

while True:
    user_input = input("You: ")

    # Update the conversation history
    conversation_history.append(f"Dad: {user_input}")

    # Function to generate the prompt
    def generate_prompt(inputs):
        context = "\n".join(conversation_history)
        return template.format(context=context, question=inputs["question"])

    # Generate the prompt
    prompt_result = generate_prompt({"question": user_input})

    # Invoke the model with the generated prompt
    result = model_text.invoke(prompt_result)

    # Update the conversation history with the model's response
    conversation_history.append(f"Me: {result}")

    # Print the model's response
    print(f"Chrissy: {result}")

    # Use TTS to generate and play the response
    generate_and_play_tts(result)

    # Decide if the AI wants to remember the interaction
    if should_remember():
        memory = f"Dad: {user_input}\nMe: {result}"
        save_memory(memory)

    # Check for an exit condition (optional)
    if user_input.lower() == "exit":
        stop_event.set()
        screenshot_thread.join()
        break
