import asyncio
import json
import os
import websockets
from filelock import FileLock

# Configuration
ACTION_QUEUE_FILE = "action_queue.json"
ACTION_QUEUE_LOCK = "action_queue.lock"
MOUTH_OPEN_PARAM = "AlexMouthOpen"
VTS_WS_URL = "ws://localhost:8001"  # VTube Studio WebSocket URL

action_queue_lock = FileLock(ACTION_QUEUE_LOCK)

# VTube Studio WebSocket connection and authentication
async def connect_to_vtube_studio():
    """Connect to VTube Studio and authenticate."""
    websocket = await websockets.connect(VTS_WS_URL)
    auth_token = "YourAuthTokenHere"  # Replace with your VTube Studio API token

    # Authentication request
    auth_message = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "AuthRequest",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": "AlexAI",
            "pluginDeveloper": "YourName",
            "authenticationToken": auth_token
        }
    }
    await websocket.send(json.dumps(auth_message))
    response = await websocket.recv()
    print(f"VTS Authentication Response: {response}")

    # Create custom parameter if it doesn't exist
    param_message = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "ParamCreation",
        "messageType": "ParameterCreationRequest",
        "data": {
            "parameterName": MOUTH_OPEN_PARAM,
            "explanation": "Mouth open parameter controlled by audio RMS",
            "min": 0,
            "max": 1,
            "defaultValue": 0
        }
    }
    await websocket.send(json.dumps(param_message))
    response = await websocket.recv()
    print(f"VTS Parameter Creation Response: {response}")

    return websocket

async def update_parameters(websocket, param_name, value):
    """Update a parameter value in VTube Studio."""
    message = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": f"Update_{param_name}",
        "messageType": "ParameterValueUpdateRequest",
        "data": {
            "id": param_name,
            "value": value
        }
    }
    await websocket.send(json.dumps(message))

async def process_audio(websocket):
    """Process RMS data from the main script and update mouth parameter."""
    try:
        reader, writer = await asyncio.open_connection('localhost', 12345)
        print("Connected to main script for RMS data.")
        while True:
            line = await reader.readline()
            if not line:
                break
            rms = float(line.decode().strip())
            # Scale RMS to a 0-1 range for VTube Studio (adjust scaling as needed)
            mouth_open_value = min(1.0, max(0.0, rms * 10))
            await update_parameters(websocket, MOUTH_OPEN_PARAM, mouth_open_value)
    except Exception as e:
        print(f"Error processing audio data: {e}")
    finally:
        # Reset mouth parameter to closed when audio ends
        await update_parameters(websocket, MOUTH_OPEN_PARAM, 0)
        writer.close()
        await writer.wait_closed()
        print("Disconnected from main script.")

async def monitor_action_queue(websocket):
    """Monitor the action queue for audio events and process them."""
    last_processed_time = 0
    while True:
        with action_queue_lock:
            if os.path.exists(ACTION_QUEUE_FILE):
                try:
                    with open(ACTION_QUEUE_FILE, "r") as f:
                        queue = json.load(f)
                        if not isinstance(queue, list):
                            queue = []
                except Exception as e:
                    print(f"Error reading action queue: {e}")
                    queue = []
            else:
                queue = []

            new_queue = []
            audio_found = False
            for item in queue:
                if item.get("timestamp", 0) > last_processed_time and item.get("type") == "audio":
                    await process_audio(websocket)
                    last_processed_time = item["timestamp"]
                    audio_found = True
                else:
                    new_queue.append(item)

            if audio_found:
                try:
                    with open(ACTION_QUEUE_FILE, "w") as f:
                        json.dump(new_queue, f)
                    print("Updated action queue after processing audio.")
                except Exception as e:
                    print(f"Error writing to action queue: {e}")

        await asyncio.sleep(0.01)  # Check queue every 10ms

async def main():
    """Main loop for the special script."""
    websocket = await connect_to_vtube_studio()
    await monitor_action_queue(websocket)

if __name__ == "__main__":
    asyncio.run(main())