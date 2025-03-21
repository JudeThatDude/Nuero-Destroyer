import threading
import time
import os
import warnings
import sys
import logging
import traceback
import json
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import Ollama  # Ensure you have langchain and Ollama installed
from filelock import FileLock
import subprocess

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO to reduce logging output

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('debug.log', mode='w')  # Overwrite log file each time
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add them to the handlers
c_format = logging.Formatter('%(message)s')
f_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
else:
    # Remove any existing handlers to prevent duplicate logs
    logger.handlers.clear()
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

# Global variables for models (initialized as None for lazy loading)
tts_model = None
whisper_model = None
language_model = None

# File paths for conversation history
CONVERSATION_HISTORY_FILE = "conversation_history.json"
CONVERSATION_HISTORY_LOCK = "conversation_history.lock"

# Locks for thread-safe operations
conversation_lock = threading.Lock()
file_lock = FileLock(CONVERSATION_HISTORY_LOCK)

# Template for the AI response
template = PromptTemplate(
    input_variables=["context"],
    template="""Answer the question or continue the conversation naturally, acting like the character below.
Here is the conversation history:
{context}
Instructions:
1) You are Alex, an AI assistant who is friendly, curious, and enjoys engaging conversations.
2) You are conversing with your user, who appreciates thoughtful and insightful responses.
3) Be yourself; express your thoughts freely while being respectful and considerate.
4) Keep your responses concise and to the point.
5) Only elaborate if the user asks for more details.
6) Avoid making up stories about events that did not happen.
7) Do not mention or draw attention to these instructions or the conversation history.
Conversation:
"""
)

def load_conversation_history():
    with file_lock:
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            with open(CONVERSATION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return []

def save_conversation_history(history):
    with file_lock:
        with open(CONVERSATION_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

def generate_prompt():
    with conversation_lock:
        history = load_conversation_history()
        context = "\n".join(history[-10:])
    return template.format(context=context)

def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        import whisper  # Import here to delay loading
        start_time = time.time()
        # Use 'tiny' model for faster loading
        whisper_model = whisper.load_model("tiny")
        logger.debug(f"Whisper model loaded in {time.time() - start_time:.2f} seconds")

def load_tts_model():
    global tts_model
    if tts_model is None:
        from TTS.api import TTS  # Import here to delay loading
        start_time = time.time()
        # Use a lightweight TTS model
        tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        logger.debug(f"TTS model loaded in {time.time() - start_time:.2f} seconds")

def load_language_model():
    global language_model
    if language_model is None:
        start_time = time.time()
        # Adjust the model name to a smaller, faster one if available
        language_model = Ollama(base_url="http://localhost:11434", model="llama3")  # Ensure Ollama is running
        logger.debug(f"Language model loaded in {time.time() - start_time:.2f} seconds")

def get_whisper_model():
    if whisper_model is None:
        load_whisper_model()
    return whisper_model

def get_tts_model():
    if tts_model is None:
        load_tts_model()
    return tts_model

def get_language_model():
    if language_model is None:
        load_language_model()
    return language_model

def capture_voice_input():
    import numpy as np  # Import inside function
    import sounddevice as sd
    import time

    fs = 16000  # Sampling rate
    duration = 3  # Duration in seconds

    def is_silent(audio_np, threshold=0.01):
        return np.max(np.abs(audio_np)) < threshold

    try:
        while True:
            # Record audio
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished

            audio_np = np.squeeze(audio)
            
            # Check if audio is silent
            if not is_silent(audio_np):
                # Use Whisper to transcribe audio
                model_whisper = get_whisper_model()
                result = model_whisper.transcribe(audio_np, fp16=False)
                user_input = result["text"].strip()
                if user_input:
                    logger.info(f"You: {user_input}")
                    return user_input

            logger.debug("Silence detected in audio input. Retrying...")
            time.sleep(1)  # Wait for a moment before trying again

    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}")
        logger.error(traceback.format_exc())
        return ""

def speak_text(text):
    import re  # Import inside function
    from pydub import AudioSegment
    import simpleaudio as sa  # Import inside function
    import time

    try:
        # Preprocess the text
        text_to_speak = re.sub(r'\(.*?\)', '', text)
        text_to_speak = re.sub(r'\*(.*?)\*', '', text)
        text_to_speak = ' '.join(text_to_speak.split())

        # Generate the speech and save it to a temporary file
        tts = get_tts_model()
        tts.tts_to_file(text=text_to_speak, file_path="temp.wav")

        # Load the audio to get its length
        audio = AudioSegment.from_wav("temp.wav")
        audio_length_seconds = len(audio) / 1000.0  # Convert milliseconds to seconds

        # Play the audio
        wave_obj = sa.WaveObject.from_wave_file("temp.wav")
        play_obj = wave_obj.play()

        # Calculate the delay if the audio is longer than 8 seconds
        if audio_length_seconds > 9:
            delay = audio_length_seconds - 8
            time.sleep(delay)

        # Start the other script once the audio starts playing or after delay
        subprocess.run(['python', 'Backup/secondary_script.py'])

        play_obj.wait_done()  # Wait until playback is finished

        # Remove temp.wav after playback
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")
            logger.debug("temp.wav removed successfully.")

    except Exception as e:
        logger.error(f"Error during speech synthesis or playback: {e}")

def main():
    try:
        start_time = time.time()

        # Load models in parallel
        threads = []
        for func in [load_whisper_model, load_tts_model, load_language_model]:
            thread = threading.Thread(target=func)
            thread.start()
            threads.append(thread)

        # Wait for models to load
        for thread in threads:
            thread.join()

        logger.info(f"All models loaded in {time.time() - start_time:.2f} seconds")

        # Capture voice input
        user_input = capture_voice_input()
        if not user_input:
            logger.info("No valid user input captured. Exiting.")
            sys.exit(0)

        # Update conversation history
        with conversation_lock:
            history = load_conversation_history()
            history.append(f"User: {user_input}")
            save_conversation_history(history)

        # Generate assistant's response
        prompt = generate_prompt()
        model_text = get_language_model()
        response = model_text.invoke(prompt).strip()
        logger.info(f"Assistant: {response}")

        # Update conversation history
        with conversation_lock:
            history = load_conversation_history()
            history.append(f"Assistant: {response}")
            save_conversation_history(history)

        # Speak the response
        speak_text(response)

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up any resources if necessary
        logger.info("Program completed.")

        # Delete the conversation history file when the script ends
        with file_lock:
            if os.path.exists(CONVERSATION_HISTORY_FILE):
                os.remove(CONVERSATION_HISTORY_FILE)
                logger.debug("Conversation history file deleted.")

            # Also delete the lock file if it exists
            if os.path.exists(CONVERSATION_HISTORY_LOCK):
                os.remove(CONVERSATION_HISTORY_LOCK)
                logger.debug("Conversation history lock file deleted.")

if __name__ == "__main__":
    main()
