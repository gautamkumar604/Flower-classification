# 🌸 Flower Classification App

A full-stack AI application that identifies flower types from uploaded images using deep learning.

## 🎯 Features

- **Image Upload**: Drag-and-drop or click to upload flower images
- **AI Prediction**: Uses TensorFlow/Keras with MobileNetV2 transfer learning
- **Real-time Results**: Displays flower name and confidence percentage
- **Clean UI**: Beautiful, responsive interface built with React and Tailwind CSS
- **5 Flower Types**: Daisy, Dandelion, Rose, Sunflower, and Tulip

## 🛠️ Tech Stack

### Frontend
- **Framework**: React (Next.js)
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Build Tool**: Vite

### Backend
- **Framework**: Python Flask
- **AI Model**: TensorFlow/Keras
- **Architecture**: MobileNetV2 (Transfer Learning)
- **Image Processing**: Pillow (PIL)

## 📁 Project Structure

```
flower-classification-app/
├── backend/                    # Flask API server
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt       # Python dependencies
│   └── flower_model.h5        # Trained model (generated after training)
│
├── training/                   # Model training scripts
│   ├── train_model.py         # Training script
│   └── requirements.txt       # Training dependencies
│
├── dataset/                    # Training dataset (you need to provide this)
│   ├── daisy/
│   ├── dandelion/
│   ├── rose/
│   ├── sunflower/
│   └── tulip/
│
└── frontend/                   # React application (current directory)
    ├── src/
    │   └── app/
    │       ├── App.tsx
    │       └── components/
    │           └── FlowerClassifier.tsx
    └── package.json
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v18 or higher)
- **Python** (3.8 or higher)
- **pip** (Python package manager)

### Step 1: Prepare the Dataset

Download a flower dataset or create your own. Organize it in the following structure:

```
dataset/
├── daisy/          # Images of daisies
├── dandelion/      # Images of dandelions
├── rose/           # Images of roses
├── sunflower/      # Images of sunflowers
└── tulip/          # Images of tulips
```

**Recommended Dataset**: [Flower Photos Dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) or search for "flowers recognition" on Kaggle.

Each folder should contain at least 50-100 images of that flower type for good results.

### Step 2: Train the Model

1. Navigate to the training directory:
```bash
cd training
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python train_model.py
```

This will:
- Load and preprocess your dataset
- Train a MobileNetV2-based model
- Save the trained model as `flower_model.h5` in the `backend/` directory
- Take approximately 10-30 minutes depending on your hardware

**Training Output**: You should see accuracy improve over epochs. Final validation accuracy should be >85% for good results.

### Step 3: Run the Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies (if not already installed):
```bash
pip install -r requirements.txt
```

3. Start the Flask server:
```bash
python app.py
```

The backend will start on `http://localhost:5000`

You should see:
```
✓ Model loaded successfully from flower_model.h5
* Running on http://0.0.0.0:5000
```

### Step 4: Run the Frontend

1. Open a new terminal and navigate to the project root

2. Install Node.js dependencies (if not already installed):
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will start on `http://localhost:5173` (or another port if 5173 is busy)

### Step 5: Use the Application

1. Open your browser and go to `http://localhost:5173`
2. Click the upload area or drag and drop a flower image
3. Click "Predict Flower"
4. View the results with flower name and confidence percentage!

## 📡 API Documentation

### Endpoint: `POST /predict`

Predicts the flower type from an uploaded image.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file, JPG or PNG)

**Response** (Success - 200):
```json
{
  "class": "rose",
  "confidence": 0.95
}
```

**Response** (Error - 400/500):
```json
{
  "error": "Error message"
}
```

**Example using cURL**:
```bash
curl -X POST \
  http://localhost:5000/predict \
  -F "file=@/path/to/flower.jpg"
```

**Example using JavaScript**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
console.log(result); // { class: "rose", confidence: 0.95 }
```

### Endpoint: `GET /health`

Health check endpoint to verify the API is running.

**Response** (200):
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🔒 Security Features

- ✅ File type validation (JPG/PNG only)
- ✅ File size limit (5MB maximum)
- ✅ Secure filename handling
- ✅ CORS enabled for frontend communication
- ✅ Debug mode disabled
- ✅ Error handling and validation

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Image Preview**: See your uploaded image before prediction
- **Loading States**: Visual feedback during prediction
- **Error Messages**: Clear error notifications
- **Confidence Display**: Visual progress bar and percentage
- **Emoji Icons**: Flower-specific emojis for each class
- **Gradient Design**: Modern purple-pink gradient theme

## 🧪 Testing the Model

To test if your model is working correctly:

1. Make sure the backend is running
2. Try uploading different flower images
3. Check if predictions are reasonable (confidence > 70% usually means good prediction)

**Tips for better predictions**:
- Use clear, well-lit images
- Ensure the flower is the main subject
- Avoid images with multiple flower types
- Use images similar to your training data

## 📊 Model Details

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Training Strategy**: Transfer Learning
- **Frozen Layers**: MobileNetV2 base layers
- **Custom Layers**: GlobalAveragePooling2D → Dense(128) → Dense(5)
- **Activation**: Softmax (for multi-class classification)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy

## 🐛 Troubleshooting

### Backend Issues

**Problem**: `Model not loaded` error
- **Solution**: Make sure you've trained the model first using `train_model.py`

**Problem**: `ModuleNotFoundError`
- **Solution**: Install dependencies: `pip install -r backend/requirements.txt`

**Problem**: Port 5000 already in use
- **Solution**: Change the port in `app.py`: `app.run(port=5001)`

### Frontend Issues

**Problem**: `Failed to connect to server`
- **Solution**: Make sure the Flask backend is running on `http://localhost:5000`

**Problem**: CORS errors
- **Solution**: The backend has CORS enabled. Make sure flask-cors is installed.

### Training Issues

**Problem**: `Dataset directory not found`
- **Solution**: Create a `dataset/` directory with flower subfolders

**Problem**: Low accuracy (<70%)
- **Solution**: 
  - Add more training images (100+ per class recommended)
  - Train for more epochs (15-20)
  - Check dataset quality

## 📈 Improving Model Performance

1. **More Data**: Add more images to each class (500+ images per class is ideal)
2. **Data Augmentation**: Already included (rotation, flip, zoom)
3. **Fine-tuning**: Unfreeze some base model layers and train with lower learning rate
4. **Hyperparameter Tuning**: Experiment with learning rate, batch size, epochs
5. **Different Architecture**: Try other models like ResNet50, EfficientNet

## 🔄 Adding New Flower Types

To add new flower classes:

1. Add a new folder to `dataset/` with the flower name
2. Update `FLOWER_CLASSES` in `training/train_model.py`
3. Update `FLOWER_CLASSES` in `backend/app.py`
4. Update the supported flowers list in `FlowerClassifier.tsx`
5. Retrain the model
6. Update `NUM_CLASSES` to match the number of classes

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this project and make improvements!

## 📧 Support

If you encounter any issues, please check the troubleshooting section above.

---

**Made with ❤️ using React, TensorFlow, and Flask**

🌸 Happy Flower Classifying! 🌸
