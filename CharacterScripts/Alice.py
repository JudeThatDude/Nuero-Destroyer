import threading
import time
import random
import os
import warnings
import numpy as np
import sys
import re
import sounddevice as sd  # For audio recording
import simpleaudio as sa  # For audio playback
import whisper  # For speech recognition using OpenAI's Whisper
import queue
import logging
import traceback

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from TTS.api import TTS  # Text-to-Speech library
from pydub import AudioSegment

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")

# Suppress specific library warnings
warnings.filterwarnings("ignore", category=UserWarning, module='TTS')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('debug.log')
c_handler.setLevel(logging.INFO)  # Only show INFO level and above in the console
f_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to the file

# Create formatters and add them to the handlers
c_format = logging.Formatter('%(message)s')
f_format = logging.Formatter('%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Set the path to ffmpeg for pydub
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Adjust this path to your actual ffmpeg.exe location
if not os.path.isfile(ffmpeg_path):
    raise FileNotFoundError(f"ffmpeg not found at {ffmpeg_path}")
AudioSegment.converter = ffmpeg_path  # Set the ffmpeg path for pydub

# Initialize the TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Initialize the Whisper model for speech recognition
model_whisper = whisper.load_model("base")  # Options: tiny, base, small, medium, large

# Initialize the language model
model_text = OllamaLLM(model="llama3")  # Adjust the model name as needed

# Conversation history
conversation_history = []

# Lock for synchronizing access to conversation history
conversation_lock = threading.Lock()

# Flags to manage speech states
is_chrissy_speaking = threading.Event()
is_user_speaking = threading.Event()

# Shutdown flag
shutdown_flag = threading.Event()

# Queue for user input
user_input_queue = queue.Queue()

# Queue for speech playback
speech_queue = queue.Queue()

# Template for the AI response
template = PromptTemplate(
    template="""
Answer the question or continue the conversation naturally, acting like the character below.

Here is the conversation history:
{context}

Instructions:
1) Your name is Alice
2) Your are very funny
3) You are kinda smart
4) You have a thick butt
5) You are couragous
6) You don't like the color blue
7) You have 5 older sisters
8) You like pokemon
9) You play video games
10) You like to garden

Conversation:
""",
    input_variables=["context"]
)

def generate_prompt():
    with conversation_lock:
        context = "\n".join(conversation_history[-10:])  # Limit context to last 10 exchanges
    return template.format(context=context)

def capture_voice_input():
    fs = 16000  # Sampling rate
    duration = 3  # Duration in seconds
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished

        audio_np = np.squeeze(audio)

        # Implement silence detection
        silence_threshold = 0.01  # Adjust threshold as needed
        if np.max(np.abs(audio_np)) < silence_threshold:
            # Audio is silent, return empty string
            return ""

        # Use Whisper to transcribe audio
        audio_padded = whisper.pad_or_trim(audio_np)
        mel = whisper.log_mel_spectrogram(audio_padded).to(model_whisper.device)
        options = whisper.DecodingOptions(language='en', fp16=False)
        result = whisper.decode(model_whisper, mel, options)
        user_input = result.text.strip()
        if user_input:
            print(f"You: {user_input}")
        return user_input
    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}")
        traceback.print_exc()
        return ""

def speak_text(text):
    # Enqueue the text to be spoken
    speech_queue.put(text)
    logger.debug(f"Text enqueued for speech: {text}")

def speech_playback_thread():
    while not shutdown_flag.is_set():
        try:
            text = speech_queue.get(timeout=1)  # Wait for the next text to speak
        except queue.Empty:
            continue  # No text to speak, check shutdown_flag again

        try:
            # Preprocess the text
            text_to_speak = re.sub(r'\(.*?\)', '', text)
            text_to_speak = re.sub(r'\*(.*?)\*', '', text_to_speak)
            text_to_speak = ' '.join(text_to_speak.split())

            # Generate the speech and save it to a temporary file
            tts.tts_to_file(text=text_to_speak, file_path="temp.wav")

            # Load the audio file
            try:
                audio = AudioSegment.from_file("temp.wav", format="wav")
            except Exception as e:
                logger.error(f"Error loading temp.wav: {e}")
                traceback.print_exc()
                continue  # Skip to the next iteration

            # Adjust pitch and speed
            def change_pitch(sound, semitones):
                new_sample_rate = int(sound.frame_rate * (2 ** (semitones / 12)))
                return sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)

            audio = change_pitch(audio, semitones=3)
            audio = audio.speedup(playback_speed=1.1)

            # Export the modified audio
            try:
                audio.export("temp_modified.wav", format="wav")
            except Exception as e:
                logger.error(f"Error exporting temp_modified.wav: {e}")
                traceback.print_exc()
                continue  # Skip to the next iteration

            # Play the audio and wait until it's done
            try:
                wave_obj = sa.WaveObject.from_wave_file("temp_modified.wav")
                play_obj = wave_obj.play()
                logger.debug("Audio playback started.")
                while play_obj.is_playing():
                    if shutdown_flag.is_set():
                        play_obj.stop()
                        break
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error during audio playback: {e}")
                traceback.print_exc()
                continue  # Skip to the next iteration

            # Clean up temporary files
            try:
                os.remove("temp.wav")
                logger.debug("temp.wav removed successfully.")
            except OSError as e:
                logger.error(f"Error removing temp.wav: {e}")
            try:
                os.remove("temp_modified.wav")
                logger.debug("temp_modified.wav removed successfully.")
            except OSError as e:
                logger.error(f"Error removing temp_modified.wav: {e}")

            logger.debug("Finished playback. Waiting for next text to speak.")
        except Exception as e:
            logger.error(f"An error occurred in speech_playback_thread: {e}")
            traceback.print_exc()
        finally:
            speech_queue.task_done()

def chrissy_speaking_thread():
    while not shutdown_flag.is_set():
        try:
            # Wait for a random interval between 60 and 120 seconds
            sleep_time = random.randint(60, 120)
            for _ in range(sleep_time * 10):
                if shutdown_flag.is_set():
                    break
                time.sleep(0.1)

            # Decide randomly whether Chrissy should speak
            if random.random() < 0.3:  # 30% chance to speak
                # Check if Chrissy or Dad is already speaking
                if not is_chrissy_speaking.is_set() and not is_user_speaking.is_set():
                    is_chrissy_speaking.set()
                    logger.debug("is_chrissy_speaking set to True")

                    try:
                        # Generate Chrissy's message
                        prompt = generate_prompt()
                        logger.debug("Invoking language model for Chrissy's response...")
                        result = model_text.invoke(prompt).strip()
                        logger.debug(f"Language model returned: {result}")

                        # Update conversation history
                        with conversation_lock:
                            conversation_history.append(f"Chrissy: {result}")

                        # Print and speak Chrissy's response
                        print(f"Chrissy: {result}")
                        speak_text(result)

                    finally:
                        is_chrissy_speaking.clear()
                        logger.debug("is_chrissy_speaking set to False")
            else:
                pass  # Chrissy decided not to speak this time
        except Exception as e:
            logger.error(f"Exception in chrissy_speaking_thread: {e}")
            traceback.print_exc()
            is_chrissy_speaking.clear()
            logger.debug("is_chrissy_speaking set to False")

def user_listening_thread():
    while not shutdown_flag.is_set():
        try:
            # Non-blocking listening for user input
            user_input = capture_voice_input()
            if user_input:
                user_input_queue.put(user_input)
            else:
                # Sleep briefly to reduce CPU usage
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Exception in user_listening_thread: {e}")
            traceback.print_exc()

def user_processing_thread():
    while not shutdown_flag.is_set():
        try:
            if not is_chrissy_speaking.is_set():
                try:
                    user_input = user_input_queue.get(timeout=1)
                except queue.Empty:
                    continue  # No user input to process, keep looping

                is_user_speaking.set()
                logger.debug("is_user_speaking set to True")

                try:
                    if user_input.lower() in ["exit", "quit", "goodbye"]:
                        logger.info("Goodbye!")
                        shutdown_flag.set()
                        break  # Exit the thread's loop

                    # Update conversation history
                    with conversation_lock:
                        conversation_history.append(f"User: {user_input}")

                    # Generate Chrissy's response
                    prompt = generate_prompt()
                    logger.debug("Invoking language model for Chrissy's response...")
                    result = model_text.invoke(prompt).strip()
                    logger.debug(f"Language model returned: {result}")

                    # Update conversation history
                    with conversation_lock:
                        conversation_history.append(f"Alice: {result}")

                    # Print and speak Chrissy's response
                    print(f"Alice: {result}")
                    speak_text(result)
                finally:
                    is_user_speaking.clear()
                    logger.debug("is_user_speaking set to False")
            else:
                # Chrissy is speaking; wait before processing user input
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Exception in user_processing_thread: {e}")
            traceback.print_exc()
            is_user_speaking.clear()
            logger.debug("is_user_speaking set to False")

if __name__ == "__main__":
    try:
        # Start Chrissy's speaking thread
        chrissy_thread = threading.Thread(target=chrissy_speaking_thread, name='ChrissyThread')
        chrissy_thread.start()
        logger.info("ChrissyThread started.")

        # Start user listening thread
        user_listen_thread = threading.Thread(target=user_listening_thread, name='UserListenThread')
        user_listen_thread.start()
        logger.info("UserListenThread started.")

        # Start user processing thread
        user_process_thread = threading.Thread(target=user_processing_thread, name='UserProcessThread')
        user_process_thread.start()
        logger.info("UserProcessThread started.")

        # Start speech playback thread
        speech_thread = threading.Thread(target=speech_playback_thread, name='SpeechPlaybackThread')
        speech_thread.start()
        logger.info("SpeechPlaybackThread started.")

        # Keep the main thread alive until shutdown_flag is set
        while not shutdown_flag.is_set():
            time.sleep(1)
            # Check if threads are alive
            if not chrissy_thread.is_alive():
                logger.error("ChrissyThread has stopped unexpectedly.")
            if not user_listen_thread.is_alive():
                logger.error("UserListenThread has stopped unexpectedly.")
            if not user_process_thread.is_alive():
                logger.error("UserProcessThread has stopped unexpectedly.")
            if not speech_thread.is_alive():
                logger.error("SpeechPlaybackThread has stopped unexpectedly.")
    except Exception as e:
        logger.error(f"Exception in main thread: {e}")
        traceback.print_exc()
    except KeyboardInterrupt:
        logger.info("\nProgram terminated by keyboard interrupt.")
        shutdown_flag.set()
    finally:
        # Ensure all threads are properly closed
        shutdown_flag.set()
        chrissy_thread.join()
        user_listen_thread.join()
        user_process_thread.join()
        speech_thread.join()
        logger.info("All threads have been terminated. Exiting program.")
