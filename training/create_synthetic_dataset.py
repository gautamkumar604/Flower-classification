"""
Generate synthetic flower dataset for testing
Creates simple colored images to train a basic classification model
"""

import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path

def create_synthetic_dataset():
    """
    Create synthetic flower images for each class
    """
    # Create dataset directory
    dataset_dir = Path('../dataset/flowers')
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Flower classes and their colors (RGB format for PIL)
    flower_data = {
        'daisy': {'color': (255, 255, 255), 'shape': 'circle'},      # White
        'dandelion': {'color': (255, 255, 0), 'shape': 'circle'},    # Yellow
        'rose': {'color': (255, 0, 0), 'shape': 'circle'},           # Red
        'sunflower': {'color': (255, 165, 0), 'shape': 'circle'},    # Orange
        'tulip': {'color': (255, 0, 255), 'shape': 'circle'}         # Magenta
    }

    img_size = (224, 224)

    print("Generating synthetic flower dataset...")

    for flower_name, data in flower_data.items():
        flower_dir = dataset_dir / flower_name
        flower_dir.mkdir(exist_ok=True)

        print(f"Creating {flower_name} images...")

        # Create 50 images per class (reduced for faster generation)
        for i in range(50):
            # Create base image with random background
            bg_color = tuple(np.random.randint(200, 256, 3))
            img = Image.new('RGB', img_size, bg_color)
            draw = ImageDraw.Draw(img)

            # Add flower shape
            center = (img_size[0] // 2, img_size[1] // 2)
            radius = np.random.randint(30, 60)

            if data['shape'] == 'circle':
                # Draw main flower center
                draw.ellipse(
                    [(center[0] - radius, center[1] - radius),
                     (center[0] + radius, center[1] + radius)],
                    fill=data['color']
                )

                # Add some petals
                for _ in range(np.random.randint(3, 8)):
                    angle = np.random.uniform(0, 2 * np.pi)
                    petal_center = (
                        int(center[0] + radius * 0.7 * np.cos(angle)),
                        int(center[1] + radius * 0.7 * np.sin(angle))
                    )
                    petal_radius = np.random.randint(15, 25)
                    draw.ellipse(
                        [(petal_center[0] - petal_radius, petal_center[1] - petal_radius),
                         (petal_center[0] + petal_radius, petal_center[1] + petal_radius)],
                        fill=data['color']
                    )

            # Add some noise/variation
            img_array = np.array(img)
            noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array + noise, 0, 255)
            img = Image.fromarray(img_array.astype('uint8'))

            # Save image
            filename = f"{flower_name}_{i:03d}.jpg"
            filepath = flower_dir / filename
            img.save(filepath)

    print("Synthetic dataset created successfully!")
    print(f"Location: {dataset_dir.absolute()}")

    # Count images per class
    for flower_name in flower_data.keys():
        flower_dir = dataset_dir / flower_name
        count = len(list(flower_dir.glob('*.jpg')))
        print(f"{flower_name}: {count} images")

if __name__ == '__main__':
    create_synthetic_dataset()