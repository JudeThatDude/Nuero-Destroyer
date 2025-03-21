import os

ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Adjust this path to your actual ffmpeg.exe location

if not os.path.isfile(ffmpeg_path):
    logger.error(f"ffmpeg not found at {ffmpeg_path}")
    sys.exit(1)  # Commented out for testing

print("hi")