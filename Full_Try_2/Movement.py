import asyncio
import websockets
import json
import re
import aiohttp
import random
from collections import deque
from datetime import datetime
import os

# Constants
API_URL = "ws://127.0.0.1:8001"
TOKEN_FILE = "auth_token.txt"
RECONNECT_DELAY = 1.0
MAX_DELTA = 1.0

# Parameter ranges
PARAM_RANGES = {
    "HeadX": (-20, 20), "HeadY": (-20, 20), "BodyTilt": (-20, 20),
    "BodySwing": (-25, 25), "Step": (-40, 40), "FaceLean": (-15, 15),
    "BodyLean": (-15, 15), "EyesX": (-1.5, 1.5), "EyesY": (-1.5, 1.5),
    "ArmLeftX": (-40, 40), "ArmRightX": (-40, 40),
    "EyebrowLeft": (-15, 15), "EyebrowRight": (-15, 15),
    "BrowAngleLeft": (-40, 40), "BrowAngleRight": (-40, 40),
    "HandLeftX": (-40, 40), "HandRightX": (-40, 40),
    "ShouldersY": (-15, 15)
}

# Utility Functions

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def log(message):
    #print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
    return

async def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["response"]
    except Exception as e:
        log(f"Error querying Ollama: {e}")
        return None

def interpolate(current, target, param, speed=0.12, max_delta=MAX_DELTA):
    t = clamp(speed, 0, 1)
    eased_t = t * t * (3 - 2 * t)
    delta = (target - current) * eased_t
    limited_delta = clamp(delta, -max_delta, max_delta)
    min_val, max_val = PARAM_RANGES[param]
    return clamp(current + limited_delta, min_val, max_val)

def blend_targets(current_target, new_target, blend_factor=0.85):
    return tuple(current * (1 - blend_factor) + new * blend_factor for current, new in zip(current_target, new_target))

# Token Management

def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            token = f.read().strip()
            log("Loaded stored authentication token.")
            return token
    return None

def save_token(token):
    with open(TOKEN_FILE, "w") as f:
        f.write(token)
    log("Saved new authentication token.")

# Conversation History

def load_conversation_history():
    try:
        if not os.path.exists("conversation_history.json"):
            return []
        with open("conversation_history.json", "r") as f:
            content = f.read().strip()
            if not content:
                return []
            history = json.loads(content)
            if not isinstance(history, list):
                log("Conversation history is not a list. Using empty history.")
                return []
            return history[-5:]
    except json.JSONDecodeError as e:
        log(f"Error decoding conversation history: {e}. Using empty history.")
        return []
    except Exception as e:
        log(f"Unexpected error loading conversation history: {e}. Using empty history.")
        return []

def get_latest_ai_message(history):
    """Extracts the most recent Assistant message."""
    for entry in reversed(history):
        if isinstance(entry, str) and entry.startswith("Assistant:"):
            # Remove "Assistant: " prefix and return the content
            return entry[len("Assistant: "):].strip()
    return None

# Target Generator
async def target_generator(target_queue):
    last_target = tuple(0.0 for _ in PARAM_RANGES)
    param_list = list(PARAM_RANGES.keys())
    param_str = ", ".join([f"{param.lower()} between {PARAM_RANGES[param][0]} and {PARAM_RANGES[param][1]}" for param in param_list])
    format_str = ",".join(param_list).lower()

    while True:
        try:
            conversation_history = load_conversation_history()
            latest_ai_message = get_latest_ai_message(conversation_history)

            if latest_ai_message:
                prompt = (
                    f"The main AI just said: '{latest_ai_message}'\n"
                    f"Based on this, suggest values for {param_str} to reflect the AI's expression, mood, or actions (e.g., **nod**, **raise eyebrows**, **interested**). "
                    f"Make movements dynamic and varied for an active avatar. "
                    f"Provide your suggestion in the format '{format_str}'."
                )
                log(f"Prioritizing AI message: '{latest_ai_message}'")
            else:
                prompt = (
                    f"No recent AI message found. Suggest random values for {param_str}. "
                    f"Make movements dynamic and varied for an active avatar. "
                    f"Provide your suggestion in the format '{format_str}'."
                )
                log("No AI message in history. Using random fallback.")

            response = await query_ollama(prompt)
            if response and (match := re.match(r'^' + ','.join([r'(-?\d*\.?\d+)' for _ in param_list]) + r'$', response.replace(" ", ""))):
                target_values = [float(match.group(i + 1)) for i in range(len(param_list))]
                for i, param in enumerate(param_list):
                    target_values[i] = clamp(target_values[i], PARAM_RANGES[param][0], PARAM_RANGES[param][1])
            else:
                log("Invalid response format from Ollama. Using random fallback.")
                target_values = [random.uniform(PARAM_RANGES[param][0] * 0.8, PARAM_RANGES[param][1] * 0.8) for param in param_list]

            blended_target = blend_targets(last_target, tuple(target_values))
            while len(target_queue) > 1:
                target_queue.popleft()
            target_queue.append(blended_target)
            last_target = blended_target
            log(f"Target added: {', '.join([f'{param}={val:.2f}' for param, val in zip(param_list, blended_target)])}")
        except Exception as e:
            log(f"Error in target_generator: {e}")
        await asyncio.sleep(0.5)

# Parameter Generator
def parameter_generator(target_queue):
    current_values = {param: 0.0 for param in PARAM_RANGES}
    target_values = {param: 0.0 for param in PARAM_RANGES}

    while True:
        while target_queue:
            new_targets = target_queue.popleft()
            for i, param in enumerate(PARAM_RANGES.keys()):
                target_values[param] = new_targets[i]
        
        for param in PARAM_RANGES:
            current_values[param] = interpolate(current_values[param], target_values[param], param)
        
        yield tuple(current_values[param] for param in PARAM_RANGES)

# Main Function
async def main():
    target_queue = deque()
    param_gen = parameter_generator(target_queue)
    token = load_token()

    while True:
        try:
            async with websockets.connect(API_URL, ping_interval=20, ping_timeout=40) as websocket:
                if not token:
                    auth_token_request = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": "authenticate_token",
                        "messageType": "AuthenticationTokenRequest",
                        "data": {"pluginName": "start pyvts", "pluginDeveloper": "Genteki", "pluginIconURL": ""}
                    }
                    await websocket.send(json.dumps(auth_token_request))
                    auth_token_response = json.loads(await websocket.recv())
                    token = auth_token_response.get("data", {}).get("authenticationToken")
                    if not token:
                        log("Failed to retrieve authentication token.")
                        await asyncio.sleep(RECONNECT_DELAY)
                        continue
                    save_token(token)
                else:
                    log("Using stored authentication token.")

                auth_request = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "authenticate",
                    "messageType": "AuthenticationRequest",
                    "data": {"pluginName": "start pyvts", "pluginDeveloper": "Genteki", "authenticationToken": token}
                }
                await websocket.send(json.dumps(auth_request))
                auth_response = json.loads(await websocket.recv())
                if not auth_response.get("data", {}).get("authenticated", False):
                    log(f"Authentication failed: {auth_response.get('data', {}).get('reason', 'Unknown reason')}")
                    if "token expired" in auth_response.get('data', {}).get("reason", "").lower():
                        log("Token expired. Requesting a new token.")
                        token = None
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue
                log("Authenticated successfully.")

                for param in PARAM_RANGES.keys():
                    create_param_request = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": f"create_param_{param}",
                        "messageType": "ParameterCreationRequest",
                        "data": {"parameterName": param, "min": PARAM_RANGES[param][0], "max": PARAM_RANGES[param][1], "defaultValue": 0}
                    }
                    await websocket.send(json.dumps(create_param_request))
                    response = json.loads(await websocket.recv())
                    if response.get("data", {}).get("errorID") == 352:
                        log(f"{param} already exists.")
                    elif "errorID" in response.get("data", {}):
                        log(f"Error creating {param}: {response['data'].get('message', 'Unknown error')}")
                    else:
                        log(f"{param} created successfully.")

                asyncio.create_task(target_generator(target_queue))
                log("Target generator started.")

                while websocket.open:
                    try:
                        params = next(param_gen)
                        params_dict = dict(zip(PARAM_RANGES.keys(), params))
                        parameter_values = [{"id": k, "value": v} for k, v in params_dict.items()]
                        set_param_request = {
                            "apiName": "VTubeStudioPublicAPI",
                            "apiVersion": "1.0",
                            "requestID": "set_params",
                            "messageType": "InjectParameterDataRequest",
                            "data": {"parameterValues": parameter_values}
                        }
                        await websocket.send(json.dumps(set_param_request))
                        log(f"Parameters updated: {', '.join([f'{k}={v:.2f}' for k, v in params_dict.items()])}")
                    except Exception as e:
                        log(f"Error in parameter update: {e}")
                    await asyncio.sleep(0.033)

        except websockets.ConnectionClosed as e:
            log(f"Connection closed: code={e.code}, reason={e.reason}")
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            log(f"Unexpected error: {e}")
            await asyncio.sleep(RECONNECT_DELAY)

# Entry Point
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Script terminated by user.")
    except Exception as e:
        log(f"Script terminated: {e}")