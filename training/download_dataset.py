"""
Dataset Download Helper Script
Downloads and organizes the TensorFlow Flowers dataset for training
"""

import tensorflow as tf
import pathlib
import shutil
import os
import requests
import tarfile
import tempfile

def download_and_organize_dataset():
    """
    Downloads the TensorFlow flowers dataset and organizes it
    """
    print("=" * 60)
    print("Flower Dataset Download & Setup")
    print("=" * 60)
    
    # Download dataset
    print("\n📥 Downloading dataset from TensorFlow...")
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    
    try:
        # Create temp directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            tar_path = os.path.join(temp_dir, 'flower_photos.tgz')
            
            # Download with requests for better control
            print("Downloading file...")
            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
            
            print("\nExtracting archive...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(temp_dir)
            
            data_dir = pathlib.Path(temp_dir) / 'flower_photos'
            print(f"✓ Dataset downloaded to: {data_dir}")
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return
    
    # Create dataset directory
    target_dir = pathlib.Path('../dataset')
    target_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Organizing dataset to: {target_dir}")
    
    # Flower classes we need
    flower_classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    target_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    # Copy and organize files
    for source_class, target_class in zip(flower_classes, target_classes):
        source_path = data_dir / source_class
        target_path = target_dir / target_class
        
        # Create target directory
        target_path.mkdir(exist_ok=True)
        
        if source_path.exists():
            # Count images
            images = list(source_path.glob('*.jpg'))
            print(f"  📸 {target_class}: {len(images)} images")
            
            # Copy images
            for img in images:
                if not (target_path / img.name).exists():
                    shutil.copy(img, target_path / img.name)
        else:
            print(f"  ⚠️  Warning: {source_class} directory not found")
    
    print("\n✅ Dataset setup complete!")
    print(f"\nDataset location: {target_dir.absolute()}")
    print("\nYou can now run train_model.py to train the model.")
    print("=" * 60)


if __name__ == '__main__':
    download_and_organize_dataset()
