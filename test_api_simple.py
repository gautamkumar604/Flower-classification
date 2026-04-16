import requests

# Test health endpoint
try:
    response = requests.get('http://localhost:5000/health')
    print('Health check:', response.json())
except Exception as e:
    print('Health check failed:', e)

# Test with a simple image (we'll create one)
from PIL import Image
import io

# Create a red image (should predict rose)
red_img = Image.new('RGB', (100, 100), (255, 0, 0))
img_bytes = io.BytesIO()
red_img.save(img_bytes, format='PNG')
img_bytes.seek(0)

# Test prediction
try:
    files = {'file': ('test.png', img_bytes, 'image/png')}
    response = requests.post('http://localhost:5000/predict', files=files)
    print('Red image prediction:', response.json())
except Exception as e:
    print('Prediction failed:', e)

# Create a yellow image (should predict sunflower or dandelion)
yellow_img = Image.new('RGB', (100, 100), (255, 255, 0))
img_bytes2 = io.BytesIO()
yellow_img.save(img_bytes2, format='PNG')
img_bytes2.seek(0)

# Test prediction
try:
    files = {'file': ('test2.png', img_bytes2, 'image/png')}
    response = requests.post('http://localhost:5000/predict', files=files)
    print('Yellow image prediction:', response.json())
except Exception as e:
    print('Prediction failed:', e)