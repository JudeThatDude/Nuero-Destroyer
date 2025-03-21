import os
import time
import warnings
import pyautogui
from PIL import Image
import numpy as np
from threading import Thread, Event
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")

template = """
Answer the question below

Here is the conversation history: {context}

Question: {question}

These are the text positions and interpretations of the latest screenshot (up to 5):

{text_positions}

Image interpretation summary:

{image_interpretation}

Answer:
"""

ollama_model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | ollama_model

# Replace with your Clarifai API key
api_key = "w1ngf"

app = ClarifaiApp(api_key=api_key)
model = app.models.get("general-v1.3")

def save_memory(memory):
    with open("memories/memories_fix_.txt", "a") as file:
        file.write(memory + "\n")
    print("Memory saved:", memory)  # Debugging statement

def capture_screenshot(monitor="primary"):
    if monitor == "primary":
        # Capture 4K primary monitor screenshot
        region = (0, 0, 3840, 2160)
        screenshot = pyautogui.screenshot(region=region)
    elif monitor == "secondary":
        # Capture secondary monitor screenshot
        region = (1920, 0, 1920, 1080)
        screenshot = pyautogui.screenshot(region=region)
    else:
        raise ValueError("Invalid monitor specified. Use 'primary' or 'secondary'.")
    
    screenshot.save("screenshot_fix_.png")
    return Image.open("screenshot_fix_.png")

def process_screenshot(image_path):
    image = ClImage(file_obj=open(image_path, 'rb'))
    response = model.predict([image])

    text_positions = []
    for concept in response['outputs'][0]['data']['concepts']:
        text_positions.append({
            'text': concept['name'],
            'value': concept['value']
        })
    return text_positions[:5]  # Limit to first 5 text positions

def interpret_image(image_path):
    image = ClImage(file_obj=open(image_path, 'rb'))
    response = model.predict([image])

    detections = []
    for concept in response['outputs'][0]['data']['concepts']:
        detections.append(f'{concept["name"]}: {concept["value"]}')
    return "\n".join(detections)[:5]  # Limit to first 5 detections

def screenshot_loop(monitor="primary", stop_event=None, context_ref=None, image_interpretation_ref=None):
    while not stop_event.is_set():
        screenshot = capture_screenshot(monitor)
        screenshot.save("screenshot_fix_.png")
        text_positions = process_screenshot("screenshot_fix_.png")
        image_interpretation = interpret_image("screenshot_fix_.png")
        if context_ref is not None:
            context_ref.append(f"Latest screenshot text positions: {text_positions}")
        if image_interpretation_ref is not None:
            image_interpretation_ref.append(f"Image interpretation: {image_interpretation}")
        time.sleep(1)

def handle_conversation():
    context = []
    image_interpretation = []

    stop_event = Event()
    screenshot_thread = Thread(target=screenshot_loop, args=("primary", stop_event, context, image_interpretation))
    screenshot_thread.start()

    try:
        while True:
            user_input = input("You: ")

            # Generate the AI response
            result = chain.invoke({
                "context": "\n".join(context),
                "question": user_input,
                "text_positions": "\n".join([str(pos) for pos in context[:5]]),  # Limit to first 5 for testing
                "image_interpretation": "\n".join(image_interpretation[:1])  # Summarize image interpretation
            })
            print("AI: ", result)
            
            # Update the conversation history
            context.append(f"User: {user_input}")
            context.append(f"AI: {result}")
            
            # Ask AI if it wants to save the memory
            save_prompt = f"Do you want to save the memory of this question: '{user_input}'? Please save it if you think it's important."
            save_decision = chain.invoke({
                "context": "\n".join(context),
                "question": save_prompt,
                "text_positions": "\n".join([str(pos) for pos in context[:5]]),
                "image_interpretation": "\n".join(image_interpretation[:1])
            })
            if 'yes' in save_decision.lower():
                memory = f"User: {user_input}\nAI: {result}"
                save_memory(memory)

    finally:
        stop_event.set()
        screenshot_thread.join()

if __name__ == "__main__":
    handle_conversation()
