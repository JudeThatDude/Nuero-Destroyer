import os
import sys
import logging
import traceback
import time
import re
import warnings
from TTS.api import TTS
import playsound

# Suppress warnings from pydub and other libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('audio_test.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add them to the handlers
c_format = logging.Formatter('%(message)s')
f_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Set the path to ffmpeg for pydub
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Adjust this path to your actual ffmpeg.exe location
if not os.path.isfile(ffmpeg_path):
    logger.error(f"ffmpeg not found at {ffmpeg_path}")
    sys.exit(1)

# Import AudioSegment after setting ffmpeg path
from pydub import AudioSegment
AudioSegment.converter = ffmpeg_path  # Set ffmpeg path directly

def main():
    try:
        # Initialize the TTS model
        logger.debug("Initializing TTS model...")
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

        # Text to synthesize
        text = "Hello, this is a test of the text-to-speech system."

        # Preprocess the text
        text_to_speak = re.sub(r'\(.*?\)', '', text)
        text_to_speak = re.sub(r'\*(.*?)\*', '', text_to_speak)
        text_to_speak = ' '.join(text_to_speak.split())

        # Generate the speech and save it to a temporary file
        logger.debug("Generating speech...")
        tts.tts_to_file(text=text_to_speak, file_path="temp.wav")
        logger.debug("Speech generation complete. Saved to temp.wav")

        # Load the audio file
        logger.debug("Loading temp.wav...")
        audio = AudioSegment.from_file("temp.wav", format="wav")
        logger.debug("temp.wav loaded successfully.")

        # Adjust pitch and speed
        def change_pitch(sound, semitones):
            new_sample_rate = int(sound.frame_rate * (2 ** (semitones / 12)))
            return sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)

        logger.debug("Adjusting pitch and speed...")
        audio = change_pitch(audio, semitones=3)
        audio = audio.speedup(playback_speed=1.1)
        logger.debug("Pitch and speed adjusted.")

        # Export the modified audio
        logger.debug("Exporting modified audio to temp_modified.wav...")
        audio.export("temp_modified.wav", format="wav")
        logger.debug("Modified audio saved to temp_modified.wav.")

        # Play the audio using playsound
        try:
            from playsound import playsound
            logger.debug("Playing audio with playsound...")
            playsound("temp_modified.wav")
            logger.debug("Audio playback finished.")
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            traceback.print_exc()

        # Small delay before deletion
        time.sleep(0.5)

        # Clean up temporary files
        logger.debug("Cleaning up temporary files...")
        try:
            os.remove("temp.wav")
            logger.debug("temp.wav removed successfully.")
        except Exception as e:
            logger.error(f"Error removing temp.wav: {e}")
        try:
            os.remove("temp_modified.wav")
            logger.debug("temp_modified.wav removed successfully.")
        except Exception as e:
            logger.error(f"Error removing temp_modified.wav: {e}")

        logger.info("Audio test completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred in the audio test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
