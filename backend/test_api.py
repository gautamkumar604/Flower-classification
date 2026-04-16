"""
API Testing Script
Tests the Flask backend endpoints
"""

import requests
import os
import sys

API_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("\n🏥 Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Status: {data['status']}")
            print(f"✓ Model Loaded: {data['model_loaded']}")
            return True
        else:
            print(f"✗ Error: Status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to server. Is the Flask app running?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_prediction(image_path):
    """Test the prediction endpoint with an image"""
    print(f"\n🔍 Testing /predict endpoint with image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"✗ Error: Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Prediction successful!")
            print(f"  Class: {data['class']}")
            print(f"  Confidence: {data['confidence']:.2%}")
            return True
        else:
            print(f"✗ Error: Status code {response.status_code}")
            print(f"  Response: {response.json()}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_invalid_file():
    """Test error handling with invalid file"""
    print("\n🚫 Testing error handling (no file)...")
    try:
        response = requests.post(f"{API_URL}/predict")
        if response.status_code == 400:
            print(f"✓ Correctly rejected request without file")
            print(f"  Error message: {response.json()['error']}")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Flask API Test Suite")
    print("=" * 60)
    
    # Test 1: Health check
    health_ok = test_health()
    
    if not health_ok:
        print("\n❌ Health check failed. Make sure the Flask server is running:")
        print("   cd backend && python app.py")
        sys.exit(1)
    
    # Test 2: Error handling
    test_invalid_file()
    
    # Test 3: Prediction with image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_prediction(image_path)
    else:
        print("\n💡 To test prediction, run:")
        print("   python test_api.py /path/to/flower/image.jpg")
    
    print("\n" + "=" * 60)
    print("✅ Test Suite Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
