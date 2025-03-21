import asyncio
import aiofiles
import json
import re
from langchain_ollama import OllamaLLM
from filelock import FileLock
from pydub import AudioSegment
import simpleaudio as sa
import subprocess
import psutil
import logging
import traceback
import os
from functools import lru_cache
from collections import deque
import webrtcvad
import numpy as np
import sounddevice as sd
import time
from TTS.api import TTS
import whisper
from langchain.prompts import PromptTemplate

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('debug.log', mode='w')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(message)s')
f_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.handlers.clear()
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Global variables
tts_model = None
whisper_model = None
language_model = None
assistant_speaking = False

# File paths
CONVERSATION_HISTORY_FILE = "conversation_history.json"
CONVERSATION_HISTORY_LOCK = "conversation_history.lock"
ACTION_QUEUE_FILE = "action_queue.json"
ACTION_QUEUE_LOCK = "action_queue.lock"
SECONDARY_SCRIPT_PATH = "Backup/Backup.py"

# Locks
file_lock = FileLock(CONVERSATION_HISTORY_LOCK)
action_queue_lock = FileLock(ACTION_QUEUE_LOCK)

# Prompt template
template = PromptTemplate(
    input_variables=["context"],
    template="""Answer the question or continue the conversation naturally, acting like the character below.
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

# Pronunciation dictionary (expand as needed)
pronunciation_dict = {
    "schedule": "SK EH JH UW L",
    "anime": "AE N AH M EY",
    "NeuroSama": "N UH R OH S AE M AH"  # Example for custom pronunciation
}

@lru_cache(maxsize=128)
def get_phonemes(word):
    """Retrieve phonemes for a word from the pronunciation dictionary."""
    return pronunciation_dict.get(word.lower(), None)

async def load_conversation_history():
    """Load conversation history asynchronously."""
    with file_lock:
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            async with aiofiles.open(CONVERSATION_HISTORY_FILE, 'r') as f:
                return json.loads(await f.read())
        return []

async def save_conversation_history(history):
    """Save conversation history asynchronously."""
    with file_lock:
        async with aiofiles.open(CONVERSATION_HISTORY_FILE, 'w') as f:
            await f.write(json.dumps(history, ensure_ascii=False, indent=2))

async def generate_prompt():
    """Generate the prompt using the latest conversation history."""
    history = await load_conversation_history()
    context = "\n".join(history[-10:])
    return template.format(context=context)

async def load_whisper_model():
    """Load the Whisper model asynchronously."""
    global whisper_model
    if whisper_model is None:
        start_time = time.time()
        whisper_model = whisper.load_model("tiny")
        logger.debug(f"Whisper model loaded in {time.time() - start_time:.2f} seconds")

async def load_tts_model():
    """Load the VITS TTS model for natural, synthetic speech."""
    global tts_model
    if tts_model is None:
        start_time = time.time()
        tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)
        logger.debug(f"TTS model (VITS) loaded in {time.time() - start_time:.2f} seconds")

async def load_language_model():
    """Load the language model asynchronously."""
    global language_model
    if language_model is None:
        start_time = time.time()
        language_model = OllamaLLM(base_url="http://localhost:11434", model="llama3.1:8b")
        logger.debug(f"Language model loaded in {time.time() - start_time:.2f} seconds")

async def get_whisper_model():
    """Get or load the Whisper model."""
    if whisper_model is None:
        await load_whisper_model()
    return whisper_model

async def get_tts_model():
    """Get or load the TTS model."""
    if tts_model is None:
        await load_tts_model()
    return tts_model

async def get_language_model():
    """Get or load the language model."""
    if language_model is None:
        await load_language_model()
    return language_model

async def capture_voice_input():
    """Capture voice input using VAD and transcribe it."""
    global assistant_speaking
    vad_mode = 3
    frame_duration = 30
    sample_rate = 16000
    max_recording_duration = 10
    silence_duration = 0.5

    vad = webrtcvad.Vad(vad_mode)
    frames_per_buffer = int(sample_rate * frame_duration / 1000)
    dtype = 'int16'

    def int2float(sound):
        abs_max = np.abs(sound).max()
        audio = sound.astype('float32')
        if abs_max > 0:
            audio *= 1 / abs_max
        return audio.squeeze()

    try:
        with sd.RawInputStream(samplerate=sample_rate, blocksize=frames_per_buffer, dtype=dtype, channels=1) as stream:
            logger.info("Listening for speech...")
            while True:
                if assistant_speaking:
                    await asyncio.sleep(0.1)
                    continue
                ring_buffer = deque(maxlen=int(sample_rate * max_recording_duration / frames_per_buffer))
                triggered = False
                voiced_frames = []
                num_silent_frames = 0
                start_time_rec = time.time()

                while True:
                    if assistant_speaking:
                        break
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
                if assistant_speaking:
                    continue
                if voiced_frames:
                    audio_data = b''.join(voiced_frames)
                    audio_np = np.frombuffer(audio_data, dtype=dtype)
                    audio_np = int2float(audio_np)
                    model_whisper = await get_whisper_model()
                    result = model_whisper.transcribe(audio_np, fp16=False)
                    user_input = result["text"].strip()
                    if user_input:
                        logger.info(f"You: {user_input}")
                        return user_input
                    logger.info("No speech detected.")
    except Exception as e:
        logger.error(f"Error during voice capture: {e}")
        logger.error(traceback.format_exc())
        return ""

async def process_text_and_actions(full_text, audio_duration):
    """Process text and schedule actions based on timing."""
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

async def speak_text(text):
    """Generate and play TTS with a voice similar to NeuroSama using VITS."""
    global assistant_speaking

    # Helper function to adjust pitch
    def change_pitch(sound, semitones):
        """Shift the pitch of the audio by a number of semitones."""
        new_sample_rate = int(sound.frame_rate * (2 ** (semitones / 12)))
        return sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)

    # Helper function to adjust speed
    def change_speed(sound, speed):
        """Change the speed of the audio (speed < 1 slows down, speed > 1 speeds up)."""
        new_frame_rate = int(sound.frame_rate * speed)
        return sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(sound.frame_rate)

    # Helper function to check if a script is running
    def is_script_running(script_path):
        script_name = os.path.abspath(script_path)
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and script_name in ' '.join(cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False

    try:
        assistant_speaking = True

        # Clean up the text by removing parentheses, bold, and italic markers
        text_to_speak = re.sub(r'\([^()]*\)', '', text)
        text_to_speak = re.sub(r'\*\*[^*]*\*\*', '', text_to_speak)
        text_to_speak = re.sub(r'\*[^*]*\*', '', text_to_speak)
        text_to_speak = ' '.join(text_to_speak.split())

        # Generate the initial TTS audio using VITS
        tts = await get_tts_model()
        tts.tts_to_file(text=text_to_speak, file_path="temp.wav")
        audio = AudioSegment.from_wav("temp.wav")

        # Apply a slight pitch shift for a synthetic lift
        audio = change_pitch(audio, semitones=1)

        # Keep the speed at normal (1.0x) for a calm delivery
        audio = change_speed(audio, 1.0)

        # Add a subtle reverb for a polished, digital feel
        delay_ms = 20
        echo_volume_reduction = 15  # dB reduction for reverb
        silent_delay = AudioSegment.silent(duration=delay_ms)
        echo_audio = audio - echo_volume_reduction
        echo = silent_delay + echo_audio
        audio_with_reverb = audio.overlay(echo)
        audio_with_reverb = audio_with_reverb.normalize()

        # Calculate the final duration for action timing
        final_duration = len(audio_with_reverb) / 1000.0

        # Export the processed audio and play it
        audio_with_reverb.export("temp_adjusted.wav", format="wav")
        wave_obj = sa.WaveObject.from_wave_file("temp_adjusted.wav")
        play_task = asyncio.to_thread(wave_obj.play().wait_done)
        _, action_task = await process_text_and_actions(text, final_duration)
        action_task_func = action_task

        # Run playback and actions concurrently
        await asyncio.gather(play_task, action_task_func())

        # Handle the secondary script
        if not is_script_running(SECONDARY_SCRIPT_PATH):
            subprocess.Popen(['python', SECONDARY_SCRIPT_PATH])
            logger.info(f"Started secondary script: {SECONDARY_SCRIPT_PATH}")
        else:
            logger.debug("Secondary script already running.")

        assistant_speaking = False

        # Clean up temporary files
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")
        if os.path.exists("temp_adjusted.wav"):
            os.remove("temp_adjusted.wav")

    except Exception as e:
        logger.error(f"Error during speech: {e}")
        logger.error(traceback.format_exc())
        assistant_speaking = False

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

async def main():
    """Main execution loop."""
    try:
        start_time = time.time()
        await asyncio.gather(
            load_whisper_model(),
            load_tts_model(),
            load_language_model()
        )
        logger.info(f"All models loaded in {time.time() - start_time:.2f} seconds")

        user_input = await capture_voice_input()
        if not user_input:
            logger.info("No valid input. Exiting.")
        else:
            history = await load_conversation_history()
            history.append(f"User: {user_input}")
            await save_conversation_history(history)
            prompt = await generate_prompt()
            language_model_instance = await get_language_model()
            response = language_model_instance.invoke(prompt).strip()
            logger.info(f"Assistant: {response}")
            history = await load_conversation_history()
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