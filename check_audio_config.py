"""
Test script to verify audio transcription handling
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Check configuration
print("=== Audio Transcription Configuration ===\n")

aipipe_token = os.getenv("AIPIPE_TOKEN")

# Check if local whisper is installed
try:
    import whisper
    whisper_installed = True
except ImportError:
    whisper_installed = False

print(f"AIPIPE_TOKEN present: {'✅' if aipipe_token else '❌'}")
print(f"Local Whisper installed: {'✅' if whisper_installed else '❌'}")

# Check ffmpeg
import subprocess
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    ffmpeg_installed = True
except:
    ffmpeg_installed = False

print(f"ffmpeg installed: {'✅' if ffmpeg_installed else '❌'}")

print("\n=== Current Setup ===\n")

if whisper_installed:
    print("✅ Local Whisper transcription is ENABLED (FREE!)")
    print("   - Audio files will be converted using ffmpeg")
    print("   - Transcribed using local Whisper model")
    print("   - First run will download model (~140MB for 'base' model)")
    print("   - Transcription added to analysis context")
    print("\n📊 Model Info:")
    print("   - Using 'base' model (good balance of speed/accuracy)")
    print("   - Options: tiny, base, small, medium, large")
    print("   - Model stored in: ~/.cache/whisper/")
else:
    print("⚠️  Local Whisper is NOT installed")
    print("✅ Fallback logic will be used for audio-based quizzes")
    print("\nHow it works:")
    print("  1. Audio files are detected but not transcribed")
    print("  2. For CSV calculations with cutoffs:")
    print("     - Calculates BOTH sum > cutoff AND sum <= cutoff")
    print("     - Checks context for hints (below/above/less/greater)")
    print("     - Defaults to sum <= cutoff if no hints")

print("\n=== Installation ===\n")
if not whisper_installed:
    print("To install local Whisper:")
    print("  pip install openai-whisper")
    print("\nOR if already installed, activate your venv:")
    print("  source venv/bin/activate")

if not ffmpeg_installed and whisper_installed:
    print("\n⚠️  ffmpeg is required for audio conversion")
    print("Install with: brew install ffmpeg")

print("\n" + "="*50)
