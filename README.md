So, the main scripts are in the Backup folder and the main script is backup.py, a lot still needs fixing. So if you want to help, there is a discord. https://discord.gg/4Mu9wjhu

Setup Instructions for NeuroVTS
This guide outlines the steps to set up the NeuroVTS project, an AI-driven avatar controller integrating voice recognition, text-to-speech, and VTube Studio movements. Follow these instructions to prepare your environment from scratch. Python 3.10 or higher is required, along with a microphone and speakers.

Step 1: Environment Preparation
Install Python: Download and install Python 3.10+ from python.org, ensuring pip is included in the installation.
Create a Virtual Environment (Recommended): Open a terminal and run:
Linux/Mac: python -m venv neuro_chaos then source neuro_chaos/bin/activate
Windows: python -m venv neuro_chaos then neuro_chaos\Scripts\activate
Update pip: Ensure you have the latest version by running pip install --upgrade pip.
Step 2: Install Python Dependencies
Install the required Python libraries by executing the following command in your terminal:

pip install asyncio aiofiles json regex logging numpy sounddevice webrtcvad filelock whisper torch pydub scipy mss pillow aiohttp base64 pygame wavio websockets

Additional setup is needed for specific dependencies:

PyTorch with GPU Support (Optional): After the initial install, check GPU availability by running python -c "import torch; print(torch.cuda.is_available())". If False and you have an NVIDIA GPU, uninstall and reinstall with CUDA support:
pip uninstall torch
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118 (adjust cu118 to match your CUDA version).
If no GPU is available, the default torch install suffices.
OpenAI Whisper: Install from the GitHub repository:
pip install git+https://github.com/openai/whisper.git
Coqui TTS: Install the TTS library:
pip install TTS
The first run may automatically download the model tts_models/en/ljspeech/tacotron2-DDC. If it fails, manually trigger the download with tts --model_name tts_models/en/ljspeech/tacotron2-DDC --list_models.
Audio Dependencies:
webrtcvad: On Linux, you may need to install a C compiler first: sudo apt-get install libatlas-base-dev gfortran.
sounddevice: Requires PortAudio:
Linux: sudo apt-get install libportaudio2
Mac: brew install portaudio
Windows: Typically works without additional setup.
Step 3: Configure External Services
Ollama with LLaVA:
Download and install Ollama from ollama.ai.
Start the Ollama server: ollama serve
Pull the LLaVA model: ollama pull llava:7b
Verify it’s running at http://localhost:11434.
VTube Studio:
Install VTube Studio (available on Steam or other platforms).
Enable the API: Launch VTS, go to Settings, and activate the “VTube Studio API” on port 8001.
Start VTube Studio and load an avatar before running the script.
Step 4: Prepare the Script
Save the Code: Copy the neuro_vts.py script into your project directory.
Customize Neuro’s Personality (Optional): Create a file named neuro_sama_instructions.txt in the same directory with custom instructions for Neuro’s tone. If omitted, a default sarcastic personality is used.
Step 5: Run the Application
Activate the Virtual Environment (if created):
Linux/Mac: source neuro_chaos/bin/activate
Windows: neuro_chaos\Scripts\activate
Execute the Script: Run python neuro_vts.py.
Initial execution may take time to download models and authenticate with VTube Studio.
Step 6: Troubleshooting
Log File: Check debug.log in the project directory for detailed error messages.
Audio Issues: Ensure your microphone and speakers are connected and set as default devices.
VTube Studio Not Responding: Confirm VTS is running, the API is enabled, and port 8001 is accessible.
Model Loading Errors: Verify Ollama is active and the LLaVA model is downloaded.
System Requirements
Hardware: A modern CPU is sufficient; an NVIDIA GPU enhances performance for TTS and Whisper.
Operating System: Tested on Windows and Linux; macOS may require audio library adjustments.
Usage
Once running, speak to Neuro via your microphone. It will respond with sarcastic commentary and animate the VTube Studio avatar accordingly. Terminate the script with Ctrl+C.

For issues or enhancements, please open a GitHub issue. Contributions are welcome!
