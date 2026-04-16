from PIL import Image
import requests
import io

print("Testing flower classification API...")

# Test health endpoint
try:
    r = requests.get('http://localhost:5000/health')
    print('Health check:', r.json())
except Exception as e:
    print('Health check failed:', e)

# Test with different colors
colors = [(255, 0, 0), (255, 255, 0), (255, 0, 255), (255, 255, 255), (0, 255, 0)]
names = ['red', 'yellow', 'magenta', 'white', 'green']

print("\nTesting predictions with different colors:")
for color, name in zip(colors, names):
    img = Image.new('RGB', (100, 100), color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    try:
        files = {'file': ('test.png', buf, 'image/png')}
        r = requests.post('http://localhost:5000/predict', files=files)
        result = r.json()
        print(f'{name} image -> {result["class"]} (confidence: {result["confidence"]})')
    except Exception as e:
        print(f'{name} failed: {e}')

print("\nTest complete!")