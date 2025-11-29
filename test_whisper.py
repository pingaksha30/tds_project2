"""
Quick test to verify local Whisper is working
"""
import sys

print("Testing local Whisper installation...\n")

try:
    import whisper
    print("✅ whisper package imported successfully")
    
    # Try loading a model (will download on first run)
    print("\nLoading 'base' model (this may take a moment on first run)...")
    model = whisper.load_model("base")
    print("✅ Model loaded successfully")
    
    print("\n" + "="*50)
    print("SUCCESS! Local Whisper is ready to use.")
    print("="*50)
    print("\nThe transcription function will work correctly now.")
    print("Audio files will be transcribed automatically (FREE!).")
    
except ImportError as e:
    print("❌ Error: whisper package not found")
    print("\nInstall with:")
    print("  source venv/bin/activate")
    print("  pip install openai-whisper")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nThis might be due to:")
    print("  - Missing dependencies (torch, etc.)")
    print("  - Network issues (downloading model)")
    sys.exit(1)
