import { FlowerClassifier } from '../components/FlowerClassifier';

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-100 via-pink-100 to-blue-100 py-12 px-4">
      <title className="text-4xl font-bold text-center text-purple-700 mb-8">
        Flower Classification App
      </title>
      {/* Main Component */}
      <FlowerClassifier />
      
    </div>
  );
}
