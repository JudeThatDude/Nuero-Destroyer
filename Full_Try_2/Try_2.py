import asyncio
import aiofiles
import json
import re
import logging
import os
import time
import traceback
import numpy as np
import sounddevice as sd
import webrtcvad
from filelock import FileLock
import psutil
import whisper
from TTS.api import TTS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import torch
from pydub import AudioSegment
import socket
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from scipy.signal import butter, lfilter

# Set up logging
logger = logging.getLogger("AlexAI")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('debug.log', mode='w')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)
c_handler.setFormatter(logging.Formatter('%(message)s'))
f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers.clear()
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Thread pool executor for running synchronous tasks in async context
executor = ThreadPoolExecutor(max_workers=1)

# Global variables
tts_model = None
whisper_model = None
language_model = None
assistant_speaking = False

# File paths and locks
CONVERSATION_HISTORY_FILE = "conversation_history.json"
CONVERSATION_HISTORY_LOCK = "conversation_history.lock"
ACTION_QUEUE_FILE = "action_queue.json"
ACTION_QUEUE_LOCK = "action_queue.lock"
SCRIPT_PATH = os.path.abspath(__file__)
file_lock = FileLock(CONVERSATION_HISTORY_LOCK)
action_queue_lock = FileLock(ACTION_QUEUE_LOCK)
MAX_INSTANCES = 2

# Pre-load models
logger.info("Pre-loading models...")
start_time = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
whisper_model = whisper.load_model("base")  # Upgraded from "tiny" to "base" for better accuracy
if device == 'cuda':
    logger.info("GPU available, loading TTS model on CUDA")
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True).to(device)
else:
    logger.warning("No GPU available, falling back to CPU")
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
language_model = OllamaLLM(base_url="http://localhost:11434", model="llama3.1:8b")
logger.info(f"Models pre-loaded in {time.time() - start_time:.2f} seconds")

# Conversation history management
async def load_conversation_history():
    """Load conversation history from file."""
    with file_lock:
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            async with aiofiles.open(CONVERSATION_HISTORY_FILE, 'r') as f:
                content = await f.read()
                if content.strip():  # Check if the file has content
                    return json.loads(content)
                else:
                    return []  # Return empty list if file is empty
        return []  # Return empty list if file doesn’t exist

async def save_conversation_history(history):
    """Save conversation history to file."""
    with file_lock:
        async with aiofiles.open(CONVERSATION_HISTORY_FILE, 'w') as f:
            await f.write(json.dumps(history, ensure_ascii=False, indent=2))

# Prompt generation
async def generate_prompt():
    """Generate a prompt for the language model based on conversation history."""
    history = await load_conversation_history()
    context = "\n".join(history[-10:])
    template = PromptTemplate(
        input_variables=["context"],
        template=f"""Answer the question or continue the conversation naturally, acting like the character below.
Here is the conversation history:
{context}

Instructions:
1) You are Alex, an AI assistant who is friendly, curious, and enjoys engaging conversations.
2) You can trigger movements like showing interest, nodding, or raising eyebrows by adding **interested**, **nod**, or **raise eyebrows** in your response.
3) Be yourself; express your thoughts freely while being respectful and considerate.
4) Keep your responses concise and to the point.
5) Only elaborate if the user asks for more details.
6) Avoid making up stories about events that did not happen.
Conversation:
"""
    )
    return template.format(context=context)

# Improved voice input capture
async def capture_voice_input():
    """Capture voice input with improved speech recognition."""
    global assistant_speaking
    vad_mode = 2  # Lowered from 3 to 2 for better sensitivity in noisy environments
    frame_duration = 30  # ms
    sample_rate = 16000
    max_recording_duration = 15  # Increased from 10 to 15 seconds for more context
    silence_duration = 0.7  # Increased from 0.5 to 0.7 seconds for better phrase detection

    vad = webrtcvad.Vad(vad_mode)
    frames_per_buffer = int(sample_rate * frame_duration / 1000)
    dtype = 'int16'

    # Noise suppression filter (Butterworth high-pass filter)
    def butter_highpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def apply_highpass_filter(data, cutoff, fs):
        b, a = butter_highpass(cutoff, fs)
        return lfilter(b, a, data)

    def int2float(sound):
        abs_max = np.abs(sound).max()
        audio = sound.astype('float32')
        if abs_max > 0:
            audio *= 1 / abs_max
        return audio.squeeze()

    try:
        with sd.RawInputStream(samplerate=sample_rate, blocksize=frames_per_buffer, dtype=dtype, channels=1) as stream:
            logger.info("Listening for speech...")
            voiced_frames = []
            num_silent_frames = 0
            triggered = False
            start_time_rec = time.time()
            while True:
                if assistant_speaking:
                    await asyncio.sleep(0.1)
                    continue
                data = stream.read(frames_per_buffer)[0]
                if len(data) == 0:
                    break
                is_speech = vad.is_speech(data, sample_rate)
                if not triggered:
                    if is_speech:
                        triggered = True
                        voiced_frames.append(data)
                        logger.info("Speech started")
                else:
                    voiced_frames.append(data)
                    if not is_speech:
                        num_silent_frames += 1
                    else:
                        num_silent_frames = 0
                    if num_silent_frames * frame_duration >= silence_duration * 1000:
                        logger.info("Speech ended")
                        break
                    if time.time() - start_time_rec > max_recording_duration:
                        logger.info("Max recording duration reached")
                        break
            if voiced_frames:
                audio_data = b''.join(voiced_frames)
                audio_np = np.frombuffer(audio_data, dtype=dtype)
                # Apply noise suppression
                audio_np = apply_highpass_filter(audio_np, cutoff=100, fs=sample_rate)
                audio_np = int2float(audio_np)
                result = whisper_model.transcribe(audio_np, fp16=False, language="en")
                user_input = result["text"].strip()
                if user_input:
                    logger.info(f"You: {user_input}")
                    return user_input
                logger.info("No speech detected.")
    except Exception as e:
        logger.error(f"Error during voice capture: {e}")
        logger.error(traceback.format_exc())
    return ""

# Process text and actions
async def process_text_and_actions(full_text, audio_duration):
    """Process text to extract clean text and schedule facial/movement actions."""
    action_pattern = r'(\*[^\*]+\*|\*\*[^\*]+\*\*)'
    parts = re.split(action_pattern, full_text)
    words = []
    facial_actions = []
    movement_actions = []
    word_count = 0

    for part in parts:
        if part.startswith('*') and part.endswith('*'):
            action = part.strip('*')
            if part.startswith('**'):
                movement_actions.append((word_count - 1, action))
            elif part.startswith('*'):
                facial_actions.append((word_count - 1, action))
        else:
            cleaned_part = re.sub(r'\([^()]*\)', '', part)
            cleaned_part = re.sub(r'\*\*[^*]*\*\*', '', cleaned_part)
            cleaned_part = re.sub(r'\*[^*]*\*', '', cleaned_part)
            words.extend(cleaned_part.split())
            word_count = len(words)

    clean_text = " ".join(word for word in words if word)
    words_per_second = len(words) / audio_duration if audio_duration > 0 else 1

    async def action_task():
        start_time = time.time()
        for word_index, action in facial_actions:
            trigger_time = (word_index + 1) / words_per_second if word_count > 0 else 0
            elapsed = time.time() - start_time
            if elapsed < trigger_time:
                await asyncio.sleep(trigger_time - elapsed)
            send_action(action, "facial")
            logger.debug(f"Triggered facial action {action} at {trigger_time:.2f}s")
        for word_index, action in movement_actions:
            trigger_time = (word_index + 1) / words_per_second if word_count > 0 else 0
            elapsed = time.time() - start_time
            if elapsed < trigger_time:
                await asyncio.sleep(trigger_time - elapsed)
            send_action(action, "movement")
            logger.debug(f"Triggered movement action {action} at {trigger_time:.2f}s")

    return clean_text, action_task

# Audio adjustment functions
def change_pitch(sound, semitones):
    """Adjust the pitch of an AudioSegment."""
    new_sample_rate = int(sound.frame_rate * (2 ** (semitones / 12)))
    return sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)

def change_speed(sound, speed):
    """Adjust the speed of an AudioSegment."""
    new_frame_rate = int(sound.frame_rate * speed)
    return sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(sound.frame_rate)

# Text-to-speech with restart functionality
async def speak_text(text):
    """Convert text to speech and restart the script when 8 seconds remain."""
    global assistant_speaking
    try:
        assistant_speaking = True

        def _speak_text_sync(text):
            import re
            from pydub import AudioSegment
            import simpleaudio as sa
            import os

            # Preprocess the text
            text_to_speak = re.sub(r'\(.*?\)', '', text)
            text_to_speak = re.sub(r'\*(.*?)\*', '', text)
            text_to_speak = ' '.join(text_to_speak.split())

            # Generate the speech and save it to a temporary file
            tts_model.tts_to_file(text=text_to_speak, file_path="temp.wav")

            # Load the audio
            audio = AudioSegment.from_wav("temp.wav")

            # Adjust pitch and speed
            audio = change_pitch(audio, semitones=4)
            audio = audio.speedup(playback_speed=1.1)

            # Save the modified audio
            audio.export("temp_adjusted.wav", format="wav")

            # Calculate audio duration in seconds
            audio_length_seconds = len(audio) / 1000.0  # Convert milliseconds to seconds

            # Play the audio
            wave_obj = sa.WaveObject.from_wave_file("temp_adjusted.wav")
            play_obj = wave_obj.play()

            # Calculate when to restart (8 seconds before end)
            if audio_length_seconds > 10:
                restart_delay = audio_length_seconds - 10
                time.sleep(restart_delay)
                # Restart the script
                subprocess.Popen([sys.executable, SCRIPT_PATH])
                logger.info("Restarting script with 10 seconds remaining in audio playback")

            play_obj.wait_done()

            # Clean up temporary files
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")
                logger.debug("temp.wav removed successfully")
            if os.path.exists("temp_adjusted.wav"):
                os.remove("temp_adjusted.wav")
                logger.debug("temp_adjusted.wav removed successfully")

        # Run the synchronous speech function in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, _speak_text_sync, text)

    except Exception as e:
        logger.error(f"Error during speech: {e}")
        logger.error(traceback.format_exc())
    finally:
        assistant_speaking = False

# Action queue management
def send_action(action, action_type):
    """Send an action to the action queue."""
    action_data = {"action": action, "type": action_type, "timestamp": time.time()}
    with action_queue_lock:
        try:
            if os.path.exists(ACTION_QUEUE_FILE):
                with open(ACTION_QUEUE_FILE, "r") as f:
                    queue = json.load(f)
                    if not isinstance(queue, list):
                        queue = []
            else:
                queue = []
            queue.append(action_data)
            with open(ACTION_QUEUE_FILE, "w") as f:
                json.dump(queue, f)
            logger.debug(f"Appended to {ACTION_QUEUE_FILE}: {action_data}")
        except Exception as e:
            logger.error(f"Error writing to {ACTION_QUEUE_FILE}: {e}")

# Instance management
def get_active_instances():
    """Count the number of running instances of this script."""
    instance_count = sum(1 for _ in filter(
        lambda p: p.info['name'] == 'python.exe' and SCRIPT_PATH in ' '.join(p.info.get('cmdline', [])),
        psutil.process_iter(['pid', 'name', 'cmdline'])
    ))
    return instance_count

# Main loop
async def main():
    """Main program loop."""
    try:
        while True:
            user_input = await capture_voice_input()
            if not user_input:
                logger.info("No valid input. Listening again...")
                await asyncio.sleep(0.1)
                continue
            history = await load_conversation_history()
            history.append(f"User: {user_input}")
            await save_conversation_history(history)
            prompt = await generate_prompt()
            response = language_model.invoke(prompt).strip()
            logger.info(f"Assistant: {response}")
            history.append(f"Assistant: {response}")
            await save_conversation_history(history)
            await speak_text(response)
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
    finally:
        if file_lock.is_locked:
            file_lock.release()
        if os.path.exists(CONVERSATION_HISTORY_LOCK):
            os.remove(CONVERSATION_HISTORY_LOCK)

if __name__ == "__main__":
    asyncio.run(main())