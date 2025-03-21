import re
import requests
import json

def query_ollama(prompt, model_name="llama3.1:8b", max_tokens=10, temperature=0.1):
    """
    Send a prompt to the Ollama API and return the generated text.
    
    Parameters:
    - prompt: The text prompt to send to the model.
    - model_name: The name of the model to use (default: "llama3.1:8b").
    - max_tokens: Maximum number of tokens to generate (default: 10 for short responses).
    - temperature: Controls randomness (default: 0.1 for consistent output).
    
    Returns:
    - The generated text, or None if the request fails.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False  # Get the full response at once
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        return json.loads(response.text)["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return None

def coordinate_generator(screen_width=3840, screen_height=2160, steps_per_move=30):
    """
    Generator that yields smoothly updating coordinates based on AI-suggested targets via Ollama.
    
    Parameters:
    - screen_width: Maximum x-coordinate (default: 3840 for 4K).
    - screen_height: Maximum y-coordinate (default: 2160 for 4K).
    - steps_per_move: Steps to interpolate between positions (default: 30).
    
    Yields:
    - (x, y): Current position as a tuple of floats.
    """
    # Start at the center
    current_x = screen_width / 2
    current_y = screen_height / 2
    target_x = current_x
    target_y = current_y
    steps_to_target = 0

    while True:
        if steps_to_target == 0:
            # Ask the AI for a new target
            prompt = (
                f"Current position: ({current_x:.2f},{current_y:.2f}). "
                f"Suggest a new position where x is between 0 and {screen_width}, "
                f"and y is between 0 and {screen_height}, in the format 'x,y'."
            )
            response = query_ollama(prompt)
            if response:
                # Extract coordinates (integers or floats)
                match = re.search(r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)', response)
                if match:
                    target_x = float(match.group(1))
                    target_y = float(match.group(2))
                    diff_x = target_x - current_x
                    diff_y = target_y - current_y
                    steps_to_target = steps_per_move
                    increment_x = diff_x / steps_per_move
                    increment_y = diff_y / steps_per_move
                else:
                    print(f"Invalid response from AI: {response}")
            else:
                print("Failed to get response from Ollama.")

        if steps_to_target > 0:
            # Move towards the target
            current_x += increment_x
            current_y += increment_y
            steps_to_target -= 1

        # Clamp coordinates to screen boundaries
        current_x = max(0, min(current_x, screen_width))
        current_y = max(0, min(current_y, screen_height))

        # Yield the current position
        yield current_x, current_y

# Example usage
if __name__ == "__main__":
    # Note: Ensure Ollama is running (e.g., 'ollama serve') and the model is pulled (e.g., 'ollama pull llama3.1:8b')
    gen = coordinate_generator(screen_width=3840, screen_height=2160)

    # Test the generator with 100 steps
    for i in range(100):
        x, y = next(gen)
        print(f"Step {i}: Position: ({x:.2f}, {y:.2f})")