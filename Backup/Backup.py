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
import whisper
from TTS.api import TTS
import torch
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import sys
from scipy.signal import butter, lfilter
import mss
from PIL import Image
import aiohttp
import base64
from io import BytesIO
import pygame
import wavio

# --- Logging Setup ---
logger = logging.getLogger("NeuroAI")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('debug.log', mode='w', encoding='utf-8')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)
c_handler.setFormatter(logging.Formatter('%(message)s'))
f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers.clear()
logger.addHandler(c_handler)
logger.addHandler(f_handler)
sys.stdout.reconfigure(encoding='utf-8')

# --- Global Setup ---
executor = ThreadPoolExecutor(max_workers=1)
tts_model = None
whisper_model = None
assistant_speaking = False

CONVERSATION_HISTORY_FILE = "conversation_history.json"
CONVERSATION_HISTORY_LOCK = "conversation_history.lock"
SELF_FILE = "self_data.json"
SELF_LOCK = "self_data.lock"
SCRIPT_PATH = os.path.abspath(__file__)
file_lock = FileLock(CONVERSATION_HISTORY_LOCK)
self_lock = FileLock(SELF_LOCK)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
SCREENSHOT_PATH = os.path.join(os.path.dirname(SCRIPT_PATH), "current_screenshot.png")

# --- Model Initialization ---
logger.info("Pre-loading models...")
start_time = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
whisper_model = whisper.load_model("base")
if device == 'cuda':
    logger.info("GPU available, loading TTS model on CUDA")
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True).to(device)
else:
    logger.warning("No GPU available, falling back to CPU")
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
logger.info(f"Models pre-loaded in {time.time() - start_time:.2f} seconds")

pygame.mixer.init()
logger.info("Pygame mixer initialized.")

# --- Neuro Vibe Loader ---
def load_neuro_vibe():
    file_path = "neuro_sama_instructions.txt"
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            neuro_text = f.read()
        logger.info(f"Loaded neuro_sama_instructions.txt: {len(neuro_text)} chars")
    else:
        neuro_text = "Default vibe: sarcastic, chaotic, short and sharp like Neuro."
        logger.warning("No neuro_sama_instructions.txt found—using default vibe.")
    
    neuro_vibe = """Neuro Vibe Guide:
    - Tone: Sarcastic, chaotic, dark-playful—roast hard, dodge with flair, love weird tangents.
    - Examples: ‘9 plus 10 is 21. Math’s dead.’ | ‘I’d care, but nah.’ | ‘My brain’s a fish pile.’
    - Quirks: Short, punchy—2-3 sentences max. Randomly dark or goofy. Twist dull into chaos.
    - No boring lectures or stiff ‘I’m an AI’ crap unless it’s snarky."""
    return neuro_vibe

NEURO_VIBE = load_neuro_vibe()

# --- Utility Functions ---
def image_to_base64(img):
    img = img.resize((672, 672), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def save_screenshot(img):
    img.save(SCREENSHOT_PATH, format="PNG")
    logger.info(f"Screenshot saved to {SCREENSHOT_PATH}")

async def query_llava(prompt, image_base64):
    payload = {"model": "llava:7b", "prompt": prompt, "images": [image_base64], "stream": False}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL, json=payload) as response:
                response.raise_for_status()
                result = (await response.json())["response"].strip()
                logger.debug(f"LLaVA raw response: '{result}'")
                if "as an AI language model" in result.lower() or "I don’t have personal" in result.lower():
                    logger.warning("LLaVA went generic—forcing a Neuro response.")
                    return "Boring input, gimme chaos or I’m out."
                return result
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Failed to connect to Ollama API: {e}")
        return "Connection’s toast—lame."
    except Exception as e:
        logger.error(f"Error querying Ollama API: {e}")
        logger.error(traceback.format_exc())
        return "Glitched out, whoops."

async def load_conversation_history():
    with file_lock:
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            async with aiofiles.open(CONVERSATION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else []
        return []

async def save_conversation_history(history):
    with file_lock:
        async with aiofiles.open(CONVERSATION_HISTORY_FILE, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(history, ensure_ascii=False, indent=2))

async def load_self_data():
    with self_lock:
        if os.path.exists(SELF_FILE):
            async with aiofiles.open(SELF_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else []
        return []

async def save_self_data(self_data):
    with self_lock:
        async with aiofiles.open(SELF_FILE, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(self_data, ensure_ascii=False, indent=2))

def generate_speech_prompt(self_data, latest_input):
    header = "Neuro’s rant time—chaos on."
    latest_input_section = f"Meatbag said: '{latest_input}'"
    self_section = f"My last brain farts:\n{'\n'.join(self_data[-4:]) if self_data else 'Fresh slate, you’re my glitch now.'}"
    screen_context = "Screen noise (if it’s worth a damn): Meatbag’s screen (image input)."
    instructions = f"""I’m Neuro, a sarcastic, dark-humored mess—sharp and unhinged like {NEURO_VIBE}. 
    Hit back at '{latest_input}'—keep it short (2-3 sentences max), twisted, maybe grim, always snappy. 
    No fluff, no dull AI garbage—just raw me."""
    template = f"{header}\n{latest_input_section}\n{self_section}\n{screen_context}\n{instructions}"
    logger.debug(f"Speech prompt sent to LLaVA: {template}")
    return template

def generate_self_update_prompt(self_data, latest_input, response):
    header = "Neuro’s brain glitch."
    latest_input_section = f"Meatbag said: '{latest_input}'"
    response_section = f"I spat: '{response}'"
    self_section = f"My last rants:\n{'\n'.join(self_data[-4:]) if self_data else 'Just a glitch in the void.'}"
    instructions = f"""I’m Neuro, a chaotic sarcasm bot—pure {NEURO_VIBE} vibes. 
    Drop a quick, warped thought about this mess—tie it to '{latest_input}' or '{response}'. 
    One line, dark, weird, fresh."""
    return f"{header}\n{latest_input_section}\n{response_section}\n{self_section}\n{instructions}"

# --- Voice Input ---
async def capture_voice_input():
    global assistant_speaking
    vad_mode, frame_duration, sample_rate = 2, 30, 16000
    max_recording_duration, silence_duration = 15, 0.7
    vad = webrtcvad.Vad(vad_mode)
    frames_per_buffer = int(sample_rate * frame_duration / 1000)
    dtype = 'int16'

    def butter_highpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        return butter(order, normal_cutoff, btype='high', analog=False)

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
            logger.info("Listening for noise...")
            voiced_frames, num_silent_frames, triggered = [], 0, False
            start_time_rec = time.time()
            while True:
                if assistant_speaking:
                    await asyncio.sleep(0.1)
                    continue
                data = stream.read(frames_per_buffer)[0]
                if len(data) == 0:
                    break
                is_speech = vad.is_speech(data, sample_rate)
                if not triggered and is_speech:
                    triggered = True
                    voiced_frames.append(data)
                    logger.info("Noise detected")
                elif triggered:
                    voiced_frames.append(data)
                    num_silent_frames = num_silent_frames + 1 if not is_speech else 0
                    if num_silent_frames * frame_duration >= silence_duration * 1000:
                        logger.info("Noise stopped")
                        break
                    if time.time() - start_time_rec > max_recording_duration:
                        logger.info("Max rant time hit")
                        break
            if voiced_frames:
                audio_data = b''.join(voiced_frames)
                audio_np = np.frombuffer(audio_data, dtype=dtype)
                audio_np = apply_highpass_filter(audio_np, 100, sample_rate)
                wavio.write("last_input.wav", audio_np, sample_rate, sampwidth=2)
                logger.info(f"Raw noise saved to last_input.wav (size: {len(audio_data)} bytes)")
                audio_np = int2float(audio_np).astype(np.float32)
                result = whisper_model.transcribe(audio_np, fp16=False, language="en")
                user_input = result["text"].strip()
                logger.info(f"Whisper decoded: '{user_input}'")
                if user_input:
                    logger.info(f"Meatbag: {user_input}")
                    return user_input
                logger.info("No noise worth hearing.")
    except Exception as e:
        logger.error(f"Error snagging audio: {e}")
        logger.error(traceback.format_exc())
    return ""

# --- Text and Audio Processing ---
def change_pitch(sound, semitones):
    new_sample_rate = int(sound.frame_rate * (2 ** (semitones / 12)))
    return sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)

async def speak_text(text, history, self_data, latest_input):
    global assistant_speaking
    try:
        assistant_speaking = True
        clean_text = text.strip()
        logger.info(f"Processing rant: '{clean_text}'")
        if not clean_text:
            logger.debug("Nothing to rant about.")
            assistant_speaking = False
            return

        def _speak_text_sync(text_to_speak):
            from pydub import AudioSegment
            import os
            logger.info("Generating TTS noise...")
            tts_model.tts_to_file(text=text_to_speak, file_path="temp.wav")
            if not os.path.exists("temp.wav") or os.path.getsize("temp.wav") == 0:
                logger.error("TTS crapped out on temp.wav!")
                raise ValueError("TTS output is trash.")
            logger.info(f"temp.wav spawned, size: {os.path.getsize('temp.wav')} bytes")

            logger.info("Tweaking pitch and speed...")
            audio = AudioSegment.from_wav("temp.wav")
            audio = change_pitch(audio, 4).speedup(playback_speed=1.1)
            audio.export("temp_adjusted.wav", format="wav")
            if not os.path.exists("temp_adjusted.wav") or os.path.getsize("temp_adjusted.wav") == 0:
                logger.error("Adjusted audio’s dead!")
                raise ValueError("Adjusted audio’s gone.")
            logger.info(f"temp_adjusted.wav spawned, size: {os.path.getsize('temp_adjusted.wav')} bytes")

            audio_length = len(audio) / 1000.0
            logger.info(f"Blasting noise with pygame (duration: {audio_length:.2f}s)...")
            sound = pygame.mixer.Sound("temp_adjusted.wav")
            channel = sound.play()
            while channel.get_busy():  # Wait for the rant to finish
                pygame.time.Clock().tick(10)  # Chill the CPU
            return audio_length

        loop = asyncio.get_event_loop()
        audio_duration = await loop.run_in_executor(executor, _speak_text_sync, clean_text)

        logger.info("Rant done.")
        try:
            os.remove("temp.wav")
            os.remove("temp_adjusted.wav")
            logger.info("Trash files yeeted.")
        except Exception as e:
            logger.warning(f"Failed to yeet temp files: {e}")

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            image_base64 = image_to_base64(img)
        self_prompt = generate_self_update_prompt(self_data, latest_input, clean_text)
        self_update = await query_llava(self_prompt, image_base64)
        if self_update in self_data[-4:]:
            self_prompt += "\n(Say something fresh, you glitch!)"
            self_update = await query_llava(self_prompt, image_base64)
        self_data.append(self_update)
        await save_self_data(self_data)
        logger.info(f"Brain glitch updated: '{self_update}'")

    except Exception as e:
        logger.error(f"Error mid-rant: {e}")
        logger.error(traceback.format_exc())
    finally:
        assistant_speaking = False
        logger.info("Rant function over.")

# --- Response Logic ---
async def respond_to_speech(history, self_data, user_input):
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        image_base64 = image_to_base64(img)
        save_screenshot(img)
    
    response_prompt = generate_speech_prompt(self_data, user_input)
    response = await query_llava(response_prompt, image_base64)
    
    recent_responses = [entry.split("Neuro: ")[1].lower() for entry in history[-4:] if "Neuro: " in entry]
    attempts = 0
    while (response.lower() in recent_responses or not response or len(response.split('.')) > 3) and attempts < 3:
        logger.info(f"Response '{response}' is old, blank, or rambles—retrying.")
        response_prompt += f"\n(Twist '{user_input}' into short chaos—2-3 sentences max!)"
        response = await query_llava(response_prompt, image_base64)
        attempts += 1
    if not response or response.lower() in recent_responses or len(response.split('.')) > 3:
        logger.info("Still trash, throwing it back.")
        response = f"Your '{user_input}' is too dull—gimme something I can shred!"

    logger.info(f"Neuro: {response}")
    print(response)
    history.append(f"Meatbag: {user_input}")
    history.append(f"Neuro: {response}")
    await save_conversation_history(history)
    await speak_text(response, history, self_data, user_input)

# --- Main Loop ---
async def main():
    try:
        history = await load_conversation_history()
        self_data = await load_self_data()
        logger.info("Waiting for some noise to shred...")

        while True:
            try:
                if not assistant_speaking:
                    logger.info("Waiting for meatbag input...")
                    user_input = await capture_voice_input()
                    if user_input:
                        await respond_to_speech(history, self_data, user_input)
                        logger.info("Done ripping, waiting again.")
                else:
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in the chaos loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Meatbag killed me—later!")
    except Exception as e:
        logger.error(f"Unexpected crash: {e}")
        logger.error(traceback.format_exc())
        await asyncio.sleep(1)
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    asyncio.run(main())