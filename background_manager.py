import logging
import os
import time
import pathlib
from google import genai
from google.genai import types
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(pathlib.Path(__file__).parent / ".env")

# Configuration
BG_PATH    = pathlib.Path("static/cyberpunk_bg.png").resolve()
LOCATION   = "us-central1"

def is_background_stale(days=7):
    """Returns True if the background image is older than 'days'."""
    if not BG_PATH.exists():
        return True
    
    file_time = BG_PATH.stat().st_mtime
    age = (time.time() - file_time) / (24 * 3600)
    return age >= days

def update_background():
    """Generates a new background using Gemini + Vertex AI Imagen (via unified genai SDK)."""
    PROJECT_ID    = os.environ.get("GCP_PROJECT_ID")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if not PROJECT_ID:
        logger.error("GCP_PROJECT_ID not set")
        return False

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set in environment")
        return False

    logger.info("Starting background renewal...")
    
    tmp_path = None
    try:
        # 1. Generate a vivid prompt via Gemini (using Google AI Studio backend)
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
        logger.debug("Gemini prompt: %s", image_prompt)

        # 2. Generate Image via Vertex AI Imagen (using Vertex AI backend via genai SDK)
        # Note: vertexai=True uses the Vertex AI backend for this client.
        vertex_client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
        
        logger.info("Requesting image from Vertex AI Imagen (v3)...")
        res_image = vertex_client.models.generate_images(
            model="imagen-3.0-generate-001",
            prompt=image_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9"
            )
        )
        
        if not res_image.generated_images:
            logger.warning("No images were generated (possibly blocked by safety filters).")
            return False

        # 3. Save the image atomically
        tmp_path = BG_PATH.with_suffix(".tmp")
        # response.generated_images[0].image is a PIL.Image if Pillow is installed
        res_image.generated_images[0].image.save(str(tmp_path))
        tmp_path.replace(BG_PATH)
        
        logger.info("Background updated successfully.")
        return True

    except Exception as e:
        logger.error("Error during background update: %s", e)
        return False
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as e:
                logger.warning("Failed to clean up .tmp file: %s", e)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    update_background()
