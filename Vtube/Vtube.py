import asyncio
import websockets
import json
import pyautogui
import random
import math
import time

API_URL = "ws://127.0.0.1:8001"

# Function to map mouse position to parameter values
def map_mouse_to_param(x, y, screen_width, screen_height):
    head_x = (x / screen_width) * 80 - 40  # Increase range to [-40, 40] for more pronounced head turn
    head_y = ((screen_height - y) / screen_height) * 60 - 30  # Reverse y and map to range [-30, 30]
    body_tilt = (x / screen_width) * 30 - 15  # Map to range [-15, 15]
    body_swing = ((screen_height - y) / screen_height) * 40 - 20  # Reverse y and map to range [-20, 20]
    step = (x / screen_width) * 60 - 30  # Increase range to [-30, 30] for more pronounced stepping
    face_lean = (x / screen_width) * 20 - 10  # Reduce range to [-10, 10] for face lean
    body_lean = (x / screen_width) * 20 - 10  # Reduce range to [-10, 10] for body lean
    return head_x, head_y, body_tilt, body_swing, step, face_lean, body_lean

async def main():
    screen_width, screen_height = pyautogui.size()

    async with websockets.connect(API_URL) as websocket:
        try:
            # Request authentication token
            auth_token_request = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "authenticate",
                "messageType": "AuthenticationTokenRequest",
                "data": {
                    "pluginName": "start pyvts",
                    "pluginDeveloper": "Genteki",
                    "pluginIconURL": "",
                }
            }
            await websocket.send(json.dumps(auth_token_request))
            auth_token_response = await websocket.recv()
            auth_token_data = json.loads(auth_token_response)
            print(f"Authentication token response: {auth_token_response}")

            # Extract the token from the response
            if "data" in auth_token_data and "authenticationToken" in auth_token_data["data"]:
                token = auth_token_data["data"]["authenticationToken"]
            else:
                print("Failed to retrieve authentication token.")
                return

            # Authenticate using the token
            auth_request = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "authenticate",
                "messageType": "AuthenticationRequest",
                "data": {
                    "pluginName": "start pyvts",
                    "pluginDeveloper": "Genteki",
                    "authenticationToken": token
                }
            }
            await websocket.send(json.dumps(auth_request))
            auth_response = await websocket.recv()
            auth_data = json.loads(auth_response)
            print(f"Authentication response: {auth_response}")

            # Check if authentication is successful
            if "data" in auth_data and auth_data["data"].get("authenticated", False):
                print("Authenticated successfully.")
            else:
                print(f"Authentication failed: {auth_data['data']['reason']}")
                return

            # Create custom parameters
            custom_parameters = [
                {"parameterName": "HeadX", "min": -15, "max": 15, "defaultValue": 0},  # Increase range to [-40, 40] for head turn
                {"parameterName": "HeadY", "min": -15, "max": 15, "defaultValue": 0},
                {"parameterName": "BodyTilt", "min": -15, "max": 15, "defaultValue": 0},
                {"parameterName": "BodySwing", "min": -20, "max": 20, "defaultValue": 0},
                {"parameterName": "Step", "min": -30, "max": 30, "defaultValue": 0},
                {"parameterName": "FaceLean", "min": -10, "max": 10, "defaultValue": 0},  # Reduce range to [-10, 10] for face lean
                {"parameterName": "BodyLean", "min": -10, "max": 10, "defaultValue": 0}  # Reduce range to [-10, 10] for body lean
            ]

            for param in custom_parameters:
                create_param_request = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "create_param_" + param["parameterName"],
                    "messageType": "ParameterCreationRequest",
                    "data": param
                }
                await websocket.send(json.dumps(create_param_request))
                create_param_response = await websocket.recv()
                print(f"Parameter creation response for {param['parameterName']}: {create_param_response}")

            # Continuously update the custom parameter values based on mouse position
            while True:
                mouse_x, mouse_y = pyautogui.position()
                head_x, head_y, body_tilt, body_swing, step, face_lean, body_lean = map_mouse_to_param(
                    mouse_x, mouse_y, screen_width, screen_height)

                parameter_values = [
                    {"id": "HeadX", "value": head_x},
                    {"id": "HeadY", "value": head_y},
                    {"id": "BodyTilt", "value": body_tilt},
                    {"id": "BodySwing", "value": body_swing},
                    {"id": "Step", "value": step},
                    {"id": "FaceLean", "value": face_lean},
                    {"id": "BodyLean", "value": body_lean}
                ]

                set_param_request = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "set_params",
                    "messageType": "InjectParameterDataRequest",
                    "data": {"parameterValues": parameter_values}
                }
                await websocket.send(json.dumps(set_param_request))
                await websocket.recv()

                await asyncio.sleep(0.05)  # Adjust the update interval as needed

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
