import cv2
import numpy as np
import requests
import imagehash
from PIL import Image
from io import BytesIO

# Dummy in-memory list tracking past perceptual hashes
_KNOWN_HASHES = set()

def analyze_image(url: str) -> dict:
    """
    Downloads an image from the URL, computes a perceptual hash,
    checks for image reuse against known hashes, and detects human faces.
    
    Returns:
        {
            "has_face": int (0 or 1),
            "same_image_reuse": int (0 or 1)
        }
    """
    result = {"has_face": 0, "same_image_reuse": 0}
    if not url:
        return result

    try:
        # Download the image
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return result
        
        img_data = response.content
        pil_img = Image.open(BytesIO(img_data)).convert('RGB')
        
        # 1. Perceptual Hash & Reuse Detection
        phash = imagehash.phash(pil_img)
        
        # Check if hash is close to any known hashes (hamming dist <= threshold)
        threshold = 5
        is_reused = False
        for known in _KNOWN_HASHES:
            if phash - known <= threshold:
                is_reused = True
                break
                
        if is_reused:
            result["same_image_reuse"] = 1
        else:
            _KNOWN_HASHES.add(phash)
            
        # 2. Face Detection
        cv_img = np.array(pil_img)
        # Convert RGB to BGR for OpenCV (just in case, though gray conversion doesn't mind)
        cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Load pre-trained Haar cascade for frontal face
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            result["has_face"] = 1
            
    except Exception as e:
        print(f"[IMAGE MODULE] Error processing {url}: {str(e)}")
        
    return result
