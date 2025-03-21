import asyncio
import websockets
import json
import time
import math
import requests
import sys
import aiofiles
import os
from filelock import FileLock

# Configuration
API_URL = "ws://127.0.0.1:8001"  # WebSocket server (e.g., VTube Studio)
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API
CONVERSATION_HISTORY_FILE = "conversation_history.json"
CONVERSATION_HISTORY_LOCK = "conversation_history.lock"

# Lock for thread-safe access to conversation history
conversation_lock = FileLock(CONVERSATION_HISTORY_LOCK)

def query_ollama(prompt, model_name="llama3.1:8b", max_tokens=100, temperature=0.7):
    """
    Query the Ollama API for a response (synchronous).
    
    Args:
        prompt (str): The prompt to send to the AI.
        model_name (str): The model to use (default: llama3.1:8b).
        max_tokens (int): Maximum number of tokens in the response.
        temperature (float): Controls randomness in the response.
    
    Returns:
        str: The AI's response.
    """
    data = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(OLLAMA_URL, json=data)
    return response.json().get("text", "").strip()

def load_conversation_history():
    """
    Load conversation history from file with validation.
    
    Returns:
        list: A list of conversation history entries, or an empty list if validation fails.
    """
    if os.path.exists(CONVERSATION_HISTORY_FILE):
        try:
            with open(CONVERSATION_HISTORY_FILE, "r") as f:
                history = json.load(f)
            # Ensure history is a list
            if not isinstance(history, list):
                return []
            # Check each item is a dict with 'role' and 'content'
            for item in history:
                if not isinstance(item, dict) or 'role' not in item or 'content' not in item:
                    return []
            return history
        except json.JSONDecodeError:
            # Handle invalid JSON (e.g., empty file or malformed data)
            return []
    return []

def save_conversation_history(history):
    """
    Save conversation history to file.
    
    Args:
        history (list): The conversation history to save.
    """
    with open(CONVERSATION_HISTORY_FILE, "w") as f:
        json.dump(history, f)

def coordinate_generator():
    """
    Generate smooth, continuous movements for the character.
    
    Yields:
        dict: A dictionary containing position parameters for character movement.
    """
    start_time = time.time()
    while True:
        current_time = time.time() - start_time
        head_x = 20 * math.sin(1.0 * current_time) + 5 * math.sin(3.0 * current_time)
        head_y = 15 * math.sin(1.2 * current_time) + 3 * math.sin(2.5 * current_time)
        body_tilt = 10 * math.sin(0.8 * current_time) + 2 * math.sin(2.0 * current_time)
        body_swing = 10 * math.cos(0.9 * current_time) + 2 * math.cos(2.2 * current_time)
        step = 15 * math.sin(0.6 * current_time) + 3 * math.sin(1.8 * current_time)
        face_lean = 5 * math.cos(1.0 * current_time) + 1 * math.cos(3.0 * current_time)
        body_lean = 5 * math.sin(1.1 * current_time) + 1 * math.sin(2.5 * current_time)
        positions = {
            "HeadX": head_x,
            "HeadY": head_y,
            "BodyTilt": body_tilt,
            "BodySwing": body_swing,
            "Step": step,
            "FaceLean": face_lean,
            "BodyLean": body_lean
        }
        yield positions

async def movement_task(websocket):
    """
    Continuously update character movements via WebSocket.
    
    Args:
        websocket: The WebSocket connection to send updates to.
    """
    coord_gen = coordinate_generator()
    while True:
        positions = next(coord_gen)
        parameter_values = [
            {"id": "HeadX", "value": positions['HeadX']},
            {"id": "HeadY", "value": positions['HeadY']},
            {"id": "BodyTilt", "value": positions['BodyTilt']},
            {"id": "BodySwing", "value": positions['BodySwing']},
            {"id": "Step", "value": positions['Step']},
            {"id": "FaceLean", "value": positions['FaceLean']},
            {"id": "BodyLean", "value": positions['BodyLean']}
        ]
        set_param_request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "set_params",
            "messageType": "InjectParameterDataRequest",
            "data": {"parameterValues": parameter_values}
        }
        await websocket.send(json.dumps(set_param_request))
        await websocket.recv()  # Wait for acknowledgment
        await asyncio.sleep(0.02)  # 50 Hz update rate

async def conversation_task():
    """
    Handle user input and AI responses without blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    history = load_conversation_history()
    while True:
        # Get user input asynchronously using run_in_executor
        user_input = await loop.run_in_executor(None, input, "You: ")
        history.append({"role": "user", "content": user_input})
        
        # Prepare prompt with conversation history
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history]) + "\nassistant:"
        
        # Query AI in a separate thread to avoid blocking
        ai_response = await asyncio.to_thread(query_ollama, prompt)
        print(f"AI: {ai_response}")
        history.append({"role": "assistant", "content": ai_response})
        
        # Save updated history to file with thread safety
        with conversation_lock:
            save_conversation_history(history)

async def main():
    """
    Main function to run movement and conversation tasks concurrently.
    """
    async with websockets.connect(API_URL) as websocket:
        # Start both tasks
        movement = asyncio.create_task(movement_task(websocket))
        conversation = asyncio.create_task(conversation_task())
        
        # Run both tasks indefinitely
        await asyncio.gather(movement, conversation)

if __name__ == "__main__":
    asyncio.run(main())