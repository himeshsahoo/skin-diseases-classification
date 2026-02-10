import requests
import numpy as np
from PIL import Image
import io

# Create a dummy image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr.seek(0)

url = 'http://127.0.0.1:5000/'
files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    
    if "error" in response.text:
        start = response.text.find('<div class="error">')
        if start != -1:
            end = response.text.find('</div>', start)
            print("Error Found on Page:")
            print(response.text[start:end+6])
        else:
            print("Error word found but not in expected div.")
            print(response.text[:500])
    else:
        print("Success!")
        print(response.text[:200])
except Exception as e:
    print(f"Error: {e}")
