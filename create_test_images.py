from PIL import Image
import os

# Create directories
os.makedirs('dataset/flowers/daisy', exist_ok=True)
os.makedirs('dataset/flowers/rose', exist_ok=True)
os.makedirs('dataset/flowers/sunflower', exist_ok=True)
os.makedirs('dataset/flowers/tulip', exist_ok=True)
os.makedirs('dataset/flowers/dandelion', exist_ok=True)

# Create simple colored images for each class
colors = {
    'daisy': (255, 255, 255),      # White
    'rose': (255, 0, 0),           # Red
    'sunflower': (255, 165, 0),    # Orange
    'tulip': (255, 0, 255),        # Magenta
    'dandelion': (255, 255, 0)     # Yellow
}

for flower, color in colors.items():
    for i in range(10):  # 10 images per class
        img = Image.new('RGB', (224, 224), color)
        img.save(f'dataset/flowers/{flower}/{flower}_{i:03d}.jpg')

print('Created test images for all flower classes')