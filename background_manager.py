import os
import time
import datetime
import pathlib
import threading
from google import genai
import vertexai
from vertexai.vision_models import ImageGenerationModel
from dotenv import load_dotenv

load_dotenv()

# Configuration
BG_PATH = pathlib.Path("static/cyberpunk_bg.png").resolve()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = "us-central1"

def is_background_stale(days=7):
    """Returns True if the background image is older than 'days'."""
    if not BG_PATH.exists():
        return True
    
    file_time = BG_PATH.stat().st_mtime
    age = (time.time() - file_time) / (24 * 3600)
    return age >= days

def update_background():
    """Generates a new background using Gemini + Vertex AI Imagen."""
    if not PROJECT_ID:
        print("[BackgroundManager] Error: GCP_PROJECT_ID not set")
        return False

    if not GEMINI_API_KEY:
        print("[BackgroundManager] Error: GEMINI_API_KEY not found in .env")
        return False

    print("[BackgroundManager] Starting background renewal...")
    
    tmp_path = None
    try:
        # 1. Generate a vivid prompt via Gemini
        client = genai.Client(api_key=GEMINI_API_KEY)
        system_prompt = (
            "You are a creative prompt engineer for an image generation model. "
            "Generate a highly detailed, cinematic prompt for a cyberpunk-themed city background for a Connect 4 board game. "
            "The prompt should focus on atmospheric lighting (neon cyan, magenta, lime), futuristic urban architecture, "
            "and a clean, wide-angle perspective. Output ONLY the image prompt text."
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=system_prompt
        )
        image_prompt = response.text.strip()
        print(f"[BackgroundManager] Gemini prompt: {image_prompt}")

        # 2. Generate Image via Vertex AI Imagen
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        generation_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        
        print("[BackgroundManager] Requesting image from Vertex AI Imagen...")
        images = generation_model.generate_images(prompt=image_prompt, number_of_images=1)
        
        # 3. Save the image atomically
        tmp_path = BG_PATH.with_suffix(".tmp")
        images[0].save(str(tmp_path))
        tmp_path.replace(BG_PATH)
        
        print("[BackgroundManager] Background updated successfully.")
        return True

    except Exception as e:
        print(f"[BackgroundManager] Error during background update: {e}")
        return False
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as e:
                print(f"[BackgroundManager] Failed to clean up .tmp file: {e}")

if __name__ == "__main__":
    # If run directly, force an update
    update_background()
