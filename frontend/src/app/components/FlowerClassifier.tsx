import { useState, useRef } from 'react';
import { Upload, Flower2, Loader2 } from 'lucide-react';

interface PredictionResult {
  class: string;
  confidence: number;
}

export function FlowerClassifier() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.match(/image\/(jpeg|jpg|png)/)) {
      setError('Please select a JPG or PNG image');
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setError('File size must be less than 5MB');
      return;
    }

    setError('');
    setSelectedFile(file);
    setPrediction(null);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Dynamically construct API URL based on current host
      // Hey...
      // const apiUrl = `http://localhost:5000/predict`;
      const apiUrl = `https://flower-classification-vc7c.onrender.com/predict`;
      console.log(`${window.location}`)

      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const result: PredictionResult = await response.json();
      setPrediction(result);
    } catch (err) {
      const apiUrl = `http://${window.location.hostname}:5000`;
      setError(err instanceof Error ? err.message : `Failed to connect to server. Make sure the Flask backend is running on ${apiUrl}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setPrediction(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getFlowerEmoji = (flowerClass: string) => {
    const emojis: Record<string, string> = {
      daisy: '🌼',
      dandelion: '🌼',
      rose: '🌹',
      sunflower: '🌻',
      tulip: '🌷',
    };
    return emojis[flowerClass.toLowerCase()] || '🌸';
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-pink-500 to-purple-600 rounded-full mb-4">
            <Flower2 className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Flower Classification App
          </h1>
          <p className="text-gray-600">
            Upload a flower image to identify its type
          </p>
        </div>

        {/* Upload Section */}
        <div className="mb-6">
          <label
            htmlFor="file-upload"
            className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-gray-300 rounded-xl cursor-pointer hover:border-purple-500 hover:bg-purple-50 transition-all duration-200"
          >
            {previewUrl ? (
              <div className="relative w-full h-full p-4">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-full h-full object-contain rounded-lg"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <Upload className="w-12 h-12 text-gray-400 mb-4" />
                <p className="mb-2 text-sm text-gray-700">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-gray-500">JPG or PNG (MAX. 5MB)</p>
              </div>
            )}
            <input
              ref={fileInputRef}
              id="file-upload"
              type="file"
              className="hidden"
              accept="image/jpeg,image/jpg,image/png"
              onChange={handleFileSelect}
            />
          </label>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3 mb-6">
          <button
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-pink-500 to-purple-600 text-white font-semibold rounded-xl hover:from-pink-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Flower2 className="w-5 h-5" />
                Predict Flower
              </>
            )}
          </button>
          {selectedFile && (
            <button
              onClick={handleReset}
              className="px-6 py-3 bg-gray-100 text-gray-700 font-semibold rounded-xl hover:bg-gray-200 transition-all duration-200"
            >
              Reset
            </button>
          )}
        </div>

        {/* Prediction Result */}
        {prediction && (
          <div className="p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
            <div className="text-center">
              <div className="text-6xl mb-4">
                {getFlowerEmoji(prediction.class)}
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2 capitalize">
                {prediction.class}
              </h3>
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full">
                <span className="text-sm text-gray-600">Confidence:</span>
                <span className="text-lg font-bold text-purple-600">
                  {(prediction.confidence * 100).toFixed(2)}%
                </span>
              </div>
            </div>

            {/* Confidence Bar */}
            <div className="mt-4">
              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-pink-500 to-purple-600 rounded-full transition-all duration-500"
                  style={{ width: `${prediction.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Supported Flowers */}
        <div className="mt-8 p-4 bg-gray-50 rounded-xl">
          <p className="text-sm text-gray-600 text-center mb-3">Supported Flowers:</p>
          <div className="flex flex-wrap justify-center gap-2">
            {['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip'].map((flower) => (
              <span
                key={flower}
                className="px-3 py-1 bg-white text-gray-700 text-sm rounded-full border border-gray-200"
              >
                {flower}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
