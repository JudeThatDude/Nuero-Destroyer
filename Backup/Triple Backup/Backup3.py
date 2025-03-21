import threading
import time
import random
import os
import warnings
import sys
import queue
import logging
import traceback
import re

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('debug.log', mode='w')  # Overwrite log file each time
c_handler.setLevel(logging.INFO)  # Show INFO level and above in the console
f_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to the file

# Create formatters and add them to the handlers
c_format = logging.Formatter('%(message)s')
f_format = logging.Formatter('%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Global variables for models (initialized as None for lazy loading)
tts_model = None
whisper_model = None
language_model = None

# Conversation history
conversation_history = []
conversation_lock = threading.Lock()  # Lock for synchronizing access to conversation history

# Flags to manage speech states
is_speaking = threading.Event()

# Queues for user input and speech playback
user_input_queue = queue.Queue()
speech_queue = queue.Queue()

# Shared variable to store the last audio duration
last_audio_duration = 0
duration_lock = threading.Lock()

# Template for the AI response
from langchain_core.prompts import PromptTemplate  # Import here for optimized imports

template = PromptTemplate(
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
""",
    input_variables=["context"]
)

def generate_prompt():
    with conversation_lock:
        context = "\n".join(conversation_history[-10:])
    return template.format(context=context)

def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        import whisper  # Import here to delay loading
        start_time = time.time()
        whisper_model = whisper.load_model("base")  # Use 'tiny' or 'base' for faster loading
        logger.debug(f"Whisper model loaded in {time.time() - start_time:.2f} seconds")

def load_tts_model():
    global tts_model
    if tts_model is None:
        from TTS.api import TTS  # Import here to delay loading
        start_time = time.time()
        tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        logger.debug(f"TTS model loaded in {time.time() - start_time:.2f} seconds")

def load_language_model():
    global language_model
    if language_model is None:
        from langchain_ollama import OllamaLLM  # Import here to delay loading
        start_time = time.time()
        language_model = OllamaLLM(model="llama3")  # Adjust the model name as needed
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

    fs = 16000  # Sampling rate
    duration = 3  # Duration in seconds
    try:
        # Record audio
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished

        audio_np = np.squeeze(audio)

        # Implement silence detection
        silence_threshold = 0.01  # Adjust threshold as needed
        if np.max(np.abs(audio_np)) < silence_threshold:
            # Audio is silent, return empty string
            logger.debug("Silence detected in audio input.")
            return ""

        # Use Whisper to transcribe audio
        model_whisper = get_whisper_model()
        import whisper  # Import inside function
        audio_padded = whisper.pad_or_trim(audio_np)
        mel = whisper.log_mel_spectrogram(audio_padded).to(model_whisper.device)
        options = whisper.DecodingOptions(language='en', fp16=False)
        result = whisper.decode(model_whisper, mel, options)
        user_input = result.text.strip()
        if user_input:
            logger.info(f"You: {user_input}")
        return user_input
    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}")
        logger.error(traceback.format_exc())
        return ""

def speak_text(text):
    import re  # Import inside function
    is_speaking.set()
    speech_queue.put(text)
    logger.debug(f"Text enqueued for speech: {text}")

def speech_playback_thread():
    global last_audio_duration  # Declare that we are using the global variable
    import simpleaudio as sa  # Import inside function
    from pydub import AudioSegment
    logger.info("speech_playback_thread started.")
    try:
        text = speech_queue.get()
        logger.debug("is_speaking set to True")

        try:
            # Preprocess the text
            text_to_speak = re.sub(r'\(.*?\)', '', text)
            text_to_speak = re.sub(r'\*(.*?)\*', '', text)
            text_to_speak = ' '.join(text_to_speak.split())

            # Generate the speech and save it to a temporary file
            tts = get_tts_model()
            tts.tts_to_file(text=text_to_speak, file_path="temp.wav")

            # Load the audio file and get its duration
            audio_segment = AudioSegment.from_file("temp.wav", format="wav")
            audio_duration = audio_segment.duration_seconds

            # Store the duration in a shared variable
            with duration_lock:
                last_audio_duration = audio_duration
                logger.debug(f"Assistant speech duration: {audio_duration} seconds")

            # Play the audio
            wave_obj = sa.WaveObject.from_wave_file("temp.wav")
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until playback is finished

            # Remove temp.wav after playback
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")
                logger.debug("temp.wav removed successfully.")

        except Exception as e:
            logger.error(f"Error during speech synthesis or playback: {e}")
            logger.error(traceback.format_exc())

        finally:
            is_speaking.clear()
            logger.debug("is_speaking set to False")
            speech_queue.task_done()

    except Exception as e:
        logger.error(f"An error occurred in speech_playback_thread: {e}")
        logger.error(traceback.format_exc())

def user_listening_thread():
    global last_audio_duration  # Declare that we are using the global variable
    logger.info("user_listening_thread started.")
    try:
        if not is_speaking.is_set():
            # Retrieve and reset the duration of the last assistant speech
            with duration_lock:
                duration = last_audio_duration
                last_audio_duration = 0  # Reset after reading

            if duration > 0:
                logger.debug(f"Waiting for {duration} seconds before starting to listen.")
                time.sleep(duration)

            # Capture voice input once
            user_input = capture_voice_input()
            if user_input:
                user_input_queue.put(user_input)
            else:
                logger.debug("No user input captured.")
    except Exception as e:
        logger.error(f"Exception in user_listening_thread: {e}")
        logger.error(traceback.format_exc())

def user_processing_thread():
    logger.info("user_processing_thread started.")
    try:
        # Process one user input
        user_input = user_input_queue.get(timeout=10)  # Timeout added to prevent indefinite blocking
        logger.debug(f"User input retrieved: {user_input}")

        # Update conversation history
        with conversation_lock:
            conversation_history.append(f"User: {user_input}")

        # Generate the assistant's response
        prompt = generate_prompt()
        logger.debug("Invoking language model for assistant's response...")
        model_text = get_language_model()
        result = model_text.invoke(prompt).strip()
        logger.debug(f"Language model returned: {result}")

        # Update conversation history
        with conversation_lock:
            conversation_history.append(f"Assistant: {result}")

        # Print and speak the assistant's response
        logger.info(f"Assistant: {result}")
        speak_text(result)

        # Wait for the speech to finish
        speech_queue.join()

        # Mark the user input as processed
        user_input_queue.task_done()

    except queue.Empty:
        logger.debug("User input queue is empty. Exiting user_processing_thread.")
    except Exception as e:
        logger.error(f"Exception in user_processing_thread: {e}")
        logger.error(traceback.format_exc())

def assistant_speaking_thread():
    logger.info("assistant_speaking_thread started.")
    try:
        # Decide whether the assistant should speak
        if random.random() < 0.3:  # 30% chance to speak
            if not is_speaking.is_set():
                logger.debug("Assistant decides to speak.")

                # Generate the assistant's message
                prompt = generate_prompt()
                logger.debug("Invoking language model for assistant's spontaneous response...")
                model_text = get_language_model()
                result = model_text.invoke(prompt).strip()
                logger.debug(f"Language model returned: {result}")

                # Update conversation history
                with conversation_lock:
                    conversation_history.append(f"Assistant: {result}")

                # Print and speak the assistant's response
                logger.info(f"Assistant: {result}")
                speak_text(result)

                # Wait for the speech to finish
                speech_queue.join()
            else:
                logger.debug("Assistant wanted to speak but is_speaking is set.")
        else:
            logger.debug("Assistant decided not to speak this time.")
    except Exception as e:
        logger.error(f"Exception in assistant_speaking_thread: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        start_time = time.time()

        # Start model loading in parallel
        model_threads = []
        for func in [load_whisper_model, load_tts_model, load_language_model]:
            thread = threading.Thread(target=func)
            thread.start()
            model_threads.append(thread)

        # Wait for models to load
        for thread in model_threads:
            thread.join()
        logger.info(f"All models loaded in {time.time() - start_time:.2f} seconds")

        # Start speech playback thread
        speech_thread = threading.Thread(target=speech_playback_thread, name='SpeechPlaybackThread')
        speech_thread.start()
        logger.info("SpeechPlaybackThread started.")

        # Start assistant speaking thread
        assistant_speak_thread = threading.Thread(target=assistant_speaking_thread, name='AssistantSpeakThread')
        assistant_speak_thread.start()
        logger.info("AssistantSpeakThread started.")

        # Start user listening thread
        user_listen_thread = threading.Thread(target=user_listening_thread, name='UserListenThread')
        user_listen_thread.start()
        logger.info("UserListenThread started.")

        # Start user processing thread
        user_process_thread = threading.Thread(target=user_processing_thread, name='UserProcessThread')
        user_process_thread.start()
        logger.info("UserProcessThread started.")

        # Wait for the threads to complete
        assistant_speak_thread.join()
        user_listen_thread.join()
        user_process_thread.join()
        speech_thread.join()

        logger.info("All threads have completed. Exiting program.")

    except Exception as e:
        logger.error(f"Unhandled exception in main thread: {e}")
        logger.error(traceback.format_exc())
    except KeyboardInterrupt:
        logger.info("\nProgram terminated by keyboard interrupt.")
    finally:
        logger.info("Shutting down program.")
