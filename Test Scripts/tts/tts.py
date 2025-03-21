from TTS.api import TTS

# Initialize the English TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Text to synthesize
text = "Hello, I'm Chrissy, your AI assistant. How can I help you today?"

# Generate speech and save to a file
tts.tts_to_file(text=text, file_path="chrissy_voice.wav")
