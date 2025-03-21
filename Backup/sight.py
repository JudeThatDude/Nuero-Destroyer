import mss
from PIL import Image
import time
import json
import os
from datetime import datetime
import requests
import tempfile
import base64
from io import BytesIO

# Ollama API endpoint (ensure Ollama is running locally with LLaVA:7B)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Function to convert image to base64 with downscaling
def image_to_base64(img):
    # Downscale to 672x672 (reasonable size for LLaVA, reduces processing time)
    img = img.resize((672, 672), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Query Ollama API with LLaVA:7B (single call)
def query_ollama_with_image(image_base64):
    payload = {
        "model": "llava:7b",
        "prompt": "Describe this image in two parts: 1) the overall scene in a short sentence, and 2) focus on the most eye-catching object or feature and describe it in a short sentence.",
        "images": [image_base64],
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        response_text = response.json()["response"].strip()
        # Split response into scene and focus parts
        parts = response_text.split('\n')
        scene_desc = parts[0].strip() if len(parts) > 0 else "No scene description available."
        focus_desc = parts[1].strip() if len(parts) > 1 else "No focus description available."
        # Extract focus target from the second sentence (simple heuristic)
        focus_target = focus_desc.split()[1] if len(focus_desc.split()) > 1 else "unknown"
        return scene_desc, focus_target, focus_desc
    except Exception as e:
        return "No scene description available.", "unknown", "No focus description available."

# Analyze the entire screen with LLaVA:7B
def analyze_screen_with_llava(sct, monitor):
    screenshot = sct.grab(monitor)
    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    image_base64 = image_to_base64(img)
    
    # Single query for both scene and focus
    scene_description, focus_target, focused_description = query_ollama_with_image(image_base64)
    
    # Focused region is approximated as center (since no crop is sent)
    width, height = monitor['width'], monitor['height']
    crop_size = min(width, height) // 2
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    
    return {
        "scene_description": scene_description,
        "focus_target": focus_target,
        "focused_description": focused_description,
        "focused_region": f"{left},{top},{crop_size}x{crop_size}"
    }

# Main function to capture and analyze the screen
def capture_and_recognize():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        
        while True:
            try:
                start_time = time.time()
                timestamp = datetime.now().isoformat()

                # Analyze the full screen with LLaVA:7B
                analysis = analyze_screen_with_llava(sct, monitor)

                # Prepare JSON summary
                summary = {
                    "timestamp": timestamp,
                    "resolution": f"{monitor['width']}x{monitor['height']}",
                    "type": "focused_scene_analysis",
                    "scene_description": analysis["scene_description"],
                    "focus_target": analysis["focus_target"],
                    "focused_description": analysis["focused_description"],
                    "focused_region": analysis["focused_region"]
                }

                # Write to JSON file atomically
                with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
                    json.dump(summary, temp_file, indent=4)
                os.replace(temp_file.name, 'screen_summary.json')

                # Wait to maintain 1-second intervals
                time.sleep(max(0, 1 - (time.time() - start_time)))
            except KeyboardInterrupt:
                print("Script stopped by user.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    capture_and_recognize()