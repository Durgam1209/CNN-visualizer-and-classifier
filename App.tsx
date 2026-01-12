
import React, { useState, useEffect, useCallback } from 'react';
import { 
  ProjectStep, 
  DatasetType, 
  CNNLayer, 
  TrainingMetric, 
  PredictionResult 
} from './types';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  Database, 
  Layers, 
  Activity, 
  CheckCircle2, 
  ChevronRight, 
  ChevronLeft, 
  Upload, 
  Cpu, 
  Info,
  Play,
  RefreshCw,
  Image as ImageIcon,
  Zap
} from 'lucide-react';
import { analyzeClassification, generateFinalReport } from './geminiService';

// --- Constants & Defaults ---
const DATASETS = [
  { id: 'cats-vs-dogs', name: 'Cats vs Dogs', description: 'Binary classification of pets', icon: '🐶' },
  { id: 'fruits', name: 'Fruits Dataset', description: 'Classification of various fruits', icon: '🍎' },
  { id: 'flowers', name: 'Flowers Dataset', description: 'Recognizing flower species', icon: '🌻' },
  { id: 'fashion-mnist', name: 'Fashion-MNIST', description: 'Zalando\'s article images', icon: '👕' },
];

const INITIAL_LAYERS: CNNLayer[] = [
  { id: '1', type: 'Conv2D', params: { filters: 32, kernel: '3x3', activation: 'relu' } },
  { id: '2', type: 'MaxPooling2D', params: { size: '2x2' } },
  { id: '3', type: 'Conv2D', params: { filters: 64, kernel: '3x3', activation: 'relu' } },
  { id: '4', type: 'MaxPooling2D', params: { size: '2x2' } },
  { id: '5', type: 'Flatten', params: {} },
  { id: '6', type: 'Dense', params: { units: 64, activation: 'relu' } },
  { id: '7', type: 'Dropout', params: { rate: 0.5 } },
  { id: '8', type: 'Dense', params: { units: 2, activation: 'softmax' } },
];

// --- Components ---

const StepIndicator: React.FC<{ current: ProjectStep }> = ({ current }) => {
  const steps = Object.values(ProjectStep);
  return (
    <div className="flex items-center justify-between mb-8 px-4 py-3 bg-slate-800/50 rounded-xl border border-slate-700 overflow-x-auto">
      {steps.map((step, idx) => {
        const isActive = step === current;
        const isCompleted = steps.indexOf(current) > idx;
        return (
          <div key={step} className="flex items-center min-w-fit">
            <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 transition-colors ${
              isActive ? 'border-blue-500 bg-blue-500/20 text-blue-400' : 
              isCompleted ? 'border-emerald-500 bg-emerald-500 text-white' : 
              'border-slate-600 text-slate-500'
            }`}>
              {isCompleted ? <CheckCircle2 size={16} /> : idx + 1}
            </div>
            <span className={`ml-2 text-xs font-semibold uppercase tracking-wider hidden md:block ${
              isActive ? 'text-blue-400' : isCompleted ? 'text-emerald-400' : 'text-slate-500'
            }`}>
              {step}
            </span>
            {idx < steps.length - 1 && (
              <div className="w-8 h-[1px] bg-slate-700 mx-4" />
            )}
          </div>
        );
      })}
    </div>
  );
};

const Header: React.FC = () => (
  <header className="mb-8">
    <div className="flex items-center gap-3 mb-2">
      <div className="p-2 bg-blue-600 rounded-lg">
        <Cpu className="text-white" size={24} />
      </div>
      <h1 className="text-3xl font-bold tracking-tight">CNN Classifier Pro</h1>
    </div>
    <p className="text-slate-400 max-w-2xl">
      Master computer vision by building, training, and testing a Convolutional Neural Network step-by-step.
    </p>
  </header>
);

const App: React.FC = () => {
  const [step, setStep] = useState<ProjectStep>(ProjectStep.INTRO);
  const [dataset, setDataset] = useState<DatasetType>('cats-vs-dogs');
  const [layers, setLayers] = useState<CNNLayer[]>(INITIAL_LAYERS);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingData, setTrainingData] = useState<TrainingMetric[]>([]);
  const [testImage, setTestImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [summaryText, setSummaryText] = useState('');
  const [finalReport, setFinalReport] = useState('');

  const nextStep = () => {
    const steps = Object.values(ProjectStep);
    const currentIndex = steps.indexOf(step);
    if (currentIndex < steps.length - 1) {
      setStep(steps[currentIndex + 1]);
    }
  };

  const prevStep = () => {
    const steps = Object.values(ProjectStep);
    const currentIndex = steps.indexOf(step);
    if (currentIndex > 0) {
      setStep(steps[currentIndex - 1]);
    }
  };

  // --- Handlers ---

  const simulateTraining = () => {
    setIsTraining(true);
    setTrainingData([]);
    let currentEpoch = 1;
    const totalEpochs = 15;
    
    const interval = setInterval(() => {
      setTrainingData(prev => {
        const lastAcc = prev.length > 0 ? prev[prev.length - 1].accuracy : 0.45;
        const lastValAcc = prev.length > 0 ? prev[prev.length - 1].val_accuracy : 0.42;
        const lastLoss = prev.length > 0 ? prev[prev.length - 1].loss : 1.2;
        
        const newAcc = Math.min(0.98, lastAcc + Math.random() * 0.08);
        const newValAcc = Math.min(0.95, lastValAcc + Math.random() * 0.06);
        const newLoss = Math.max(0.1, lastLoss - Math.random() * 0.15);
        
        return [...prev, {
          epoch: currentEpoch,
          accuracy: newAcc,
          val_accuracy: newValAcc,
          loss: newLoss,
          val_loss: newLoss * 1.1
        }];
      });

      if (currentEpoch >= totalEpochs) {
        clearInterval(interval);
        setIsTraining(false);
      }
      currentEpoch++;
    }, 400);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setTestImage(reader.result as string);
        setPrediction(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!testImage) return;
    setIsAnalyzing(true);
    try {
      const archSummary = layers.map(l => l.type).join(' -> ');
      const result = await analyzeClassification(testImage, dataset, archSummary);
      setPrediction(result);
    } catch (err) {
      alert("Error analyzing image. Please check your API key.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleGenerateReport = async () => {
    setIsAnalyzing(true);
    try {
      const report = await generateFinalReport(summaryText);
      setFinalReport(report);
      nextStep();
    } catch (err) {
      alert("Error generating report.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // --- Render Steps ---

  const renderIntro = () => (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="grid md:grid-cols-2 gap-8 items-center">
        <div className="space-y-4">
          <h2 className="text-4xl font-extrabold text-blue-400">Deep Learning for Everyone</h2>
          <p className="text-lg text-slate-300 leading-relaxed">
            Convolutional Neural Networks (CNNs) are the backbone of modern computer vision. In this project, you'll learn how they mimic the human visual cortex to recognize patterns.
          </p>
          <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700">
            <h3 className="font-bold text-white mb-4 flex items-center gap-2">
              <Info size={18} className="text-blue-400" />
              What you will learn:
            </h3>
            <ul className="space-y-3 text-slate-400 text-sm">
              <li className="flex gap-3 items-start">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5" />
                <span>Dataset preparation & Image Augmentation</span>
              </li>
              <li className="flex gap-3 items-start">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5" />
                <span>Building multi-layer CNN architectures</span>
              </li>
              <li className="flex gap-3 items-start">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5" />
                <span>Monitoring training accuracy and loss curves</span>
              </li>
              <li className="flex gap-3 items-start">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5" />
                <span>Interpreting model predictions</span>
              </li>
            </ul>
          </div>
          <button 
            onClick={nextStep}
            className="flex items-center gap-2 px-8 py-4 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-bold transition-all transform hover:scale-105"
          >
            Start Project <ChevronRight size={20} />
          </button>
        </div>
        <div className="relative aspect-square bg-gradient-to-br from-blue-600/20 to-purple-600/20 rounded-3xl overflow-hidden border border-slate-700 flex items-center justify-center p-8">
           <div className="grid grid-cols-4 gap-4 opacity-50">
             {Array.from({ length: 16 }).map((_, i) => (
               <div key={i} className="w-12 h-12 bg-white/10 rounded-lg animate-pulse" style={{ animationDelay: `${i * 100}ms` }} />
             ))}
           </div>
           <div className="absolute inset-0 flex flex-col items-center justify-center">
             <Layers size={80} className="text-blue-500 mb-4 animate-bounce" />
             <div className="px-4 py-2 bg-slate-900/80 backdrop-blur rounded-lg border border-slate-700 mono text-sm text-blue-400">
               conv2d_layer.activation('relu')
             </div>
           </div>
        </div>
      </div>
    </div>
  );

  const renderDataset = () => (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Step 1: Choose Your Dataset</h2>
        <div className="text-slate-400 text-sm">Select a domain to focus your model on.</div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {DATASETS.map((ds) => (
          <button
            key={ds.id}
            onClick={() => setDataset(ds.id as DatasetType)}
            className={`p-6 rounded-2xl border-2 transition-all text-left ${
              dataset === ds.id 
                ? 'border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/10' 
                : 'border-slate-800 bg-slate-800/40 hover:border-slate-700'
            }`}
          >
            <div className="text-4xl mb-3">{ds.icon}</div>
            <h3 className="font-bold text-lg mb-1">{ds.name}</h3>
            <p className="text-sm text-slate-400">{ds.description}</p>
          </button>
        ))}
      </div>
      <div className="flex justify-end gap-3 pt-6 border-t border-slate-800">
        <button onClick={prevStep} className="px-6 py-2 text-slate-400 hover:text-white transition-colors">Back</button>
        <button onClick={nextStep} className="px-8 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold">Continue to Preprocessing</button>
      </div>
    </div>
  );

  const renderPreprocessing = () => (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Step 2: Preprocessing & Augmentation</h2>
        <div className="px-3 py-1 bg-amber-500/20 text-amber-500 rounded-full text-xs font-bold uppercase">Crucial for generalization</div>
      </div>
      
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 grid grid-cols-2 gap-4">
           {[
             { name: 'Original', img: 'https://picsum.photos/seed/cnn1/400/400', desc: 'Raw input data (224x224)' },
             { name: 'Resized & Grayscale', img: 'https://picsum.photos/seed/cnn1/400/400?grayscale', desc: 'Standardized 64x64 input' },
             { name: 'Horizontal Flip', img: 'https://picsum.photos/seed/cnn1/400/400', class: 'scale-x-[-1]', desc: 'Augmentation for orientation' },
             { name: 'Random Rotation', img: 'https://picsum.photos/seed/cnn1/400/400', class: 'rotate-12', desc: 'Augmentation for pose' },
           ].map((item, i) => (
             <div key={i} className="bg-slate-800/40 rounded-2xl border border-slate-700 overflow-hidden">
               <div className={`h-48 bg-slate-900 flex items-center justify-center overflow-hidden`}>
                 <img src={item.img} alt={item.name} className={`h-full w-full object-cover opacity-80 ${item.class || ''}`} />
               </div>
               <div className="p-4">
                 <div className="font-bold text-sm mb-1">{item.name}</div>
                 <div className="text-xs text-slate-500">{item.desc}</div>
               </div>
             </div>
           ))}
        </div>
        
        <div className="space-y-4">
          <div className="bg-slate-800/80 p-6 rounded-2xl border border-slate-700">
             <h3 className="font-bold mb-4 flex items-center gap-2">
               <Database size={18} className="text-blue-400" />
               Preprocessing Pipeline
             </h3>
             <div className="space-y-3">
               <div className="flex items-center gap-3 p-3 bg-slate-900/50 rounded-lg">
                 <div className="w-8 h-8 bg-blue-500/20 text-blue-400 rounded-md flex items-center justify-center font-bold text-xs">1</div>
                 <span className="text-sm">Resizing (e.g., 224x224)</span>
               </div>
               <div className="flex items-center gap-3 p-3 bg-slate-900/50 rounded-lg">
                 <div className="w-8 h-8 bg-blue-500/20 text-blue-400 rounded-md flex items-center justify-center font-bold text-xs">2</div>
                 <span className="text-sm">Normalization (0 - 1 range)</span>
               </div>
               <div className="flex items-center gap-3 p-3 bg-slate-900/50 rounded-lg">
                 <div className="w-8 h-8 bg-blue-500/20 text-blue-400 rounded-md flex items-center justify-center font-bold text-xs">3</div>
                 <span className="text-sm">Shuffle Training Data</span>
               </div>
             </div>
             <div className="mt-6 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl">
               <p className="text-xs text-emerald-400">
                 <b>Tip:</b> Augmentation creates artificial variations of training images, preventing the model from "memorizing" specific pixel positions.
               </p>
             </div>
          </div>
        </div>
      </div>

      <div className="flex justify-end gap-3 pt-6 border-t border-slate-800">
        <button onClick={prevStep} className="px-6 py-2 text-slate-400 hover:text-white transition-colors">Back</button>
        <button onClick={nextStep} className="px-8 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold">Proceed to Architecture</button>
      </div>
    </div>
  );

  const renderArchitecture = () => (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Step 3: Define CNN Architecture</h2>
        <div className="text-slate-400 text-sm">Visualizing the flow of tensors through your model.</div>
      </div>

      <div className="grid lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3 bg-slate-900 rounded-3xl border border-slate-800 p-8 flex flex-col items-center justify-center gap-4 relative overflow-hidden">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-900/10 via-transparent to-transparent opacity-50" />
          
          <div className="flex flex-col items-center gap-4 relative z-10 w-full max-w-md">
            {layers.map((layer, i) => (
              <React.Fragment key={layer.id}>
                <div className={`w-full p-4 rounded-xl border flex items-center justify-between transition-all ${
                  layer.type === 'Conv2D' ? 'bg-blue-600/10 border-blue-500/40 shadow-blue-500/5' :
                  layer.type === 'MaxPooling2D' ? 'bg-indigo-600/10 border-indigo-500/40 shadow-indigo-500/5' :
                  layer.type === 'Dense' ? 'bg-purple-600/10 border-purple-500/40 shadow-purple-500/5' :
                  'bg-slate-800/40 border-slate-700'
                }`}>
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center font-mono text-xs text-slate-300">
                      {layer.type.slice(0, 4)}
                    </div>
                    <div>
                      <h4 className="font-bold text-sm text-white">{layer.type}</h4>
                      <p className="text-[10px] uppercase tracking-widest text-slate-500">
                        {Object.entries(layer.params).map(([k, v]) => `${k}: ${v}`).join(' | ')}
                      </p>
                    </div>
                  </div>
                  <div className="text-xs font-mono text-slate-600">
                    #{i + 1}
                  </div>
                </div>
                {i < layers.length - 1 && (
                  <div className="flex flex-col items-center gap-1 opacity-40">
                    <div className="w-[2px] h-4 bg-gradient-to-b from-blue-500 to-transparent" />
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-slate-800/80 p-6 rounded-2xl border border-slate-700">
            <h3 className="font-bold text-sm mb-4 text-white uppercase tracking-wider">Layer Guide</h3>
            <div className="space-y-3">
              <div className="p-3 bg-blue-500/10 border-l-4 border-blue-500 rounded-r-lg">
                <span className="block font-bold text-xs text-blue-400">Conv2D</span>
                <p className="text-[11px] text-slate-400">Learns feature maps via filters. Detects edges, shapes.</p>
              </div>
              <div className="p-3 bg-indigo-500/10 border-l-4 border-indigo-500 rounded-r-lg">
                <span className="block font-bold text-xs text-indigo-400">Pooling</span>
                <p className="text-[11px] text-slate-400">Downsamples spatial dimensions. Redundancy reduction.</p>
              </div>
              <div className="p-3 bg-purple-500/10 border-l-4 border-purple-500 rounded-r-lg">
                <span className="block font-bold text-xs text-purple-400">Dense</span>
                <p className="text-[11px] text-slate-400">Fully connected layer for final classification logic.</p>
              </div>
            </div>
            <div className="mt-6">
              <button 
                className="w-full py-3 bg-slate-700 hover:bg-slate-600 text-white text-xs font-bold rounded-xl transition-colors flex items-center justify-center gap-2"
                onClick={() => setLayers([...INITIAL_LAYERS])}
              >
                <RefreshCw size={14} /> Reset Architecture
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-end gap-3 pt-6 border-t border-slate-800">
        <button onClick={prevStep} className="px-6 py-2 text-slate-400 hover:text-white transition-colors">Back</button>
        <button onClick={nextStep} className="px-8 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold">Move to Training</button>
      </div>
    </div>
  );

  const renderTraining = () => (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Step 4: Model Training</h2>
          <p className="text-slate-400 text-sm">Fine-tuning weights via backpropagation.</p>
        </div>
        {!isTraining && trainingData.length === 0 && (
          <button 
            onClick={simulateTraining}
            className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-bold flex items-center gap-2 shadow-lg shadow-emerald-500/20 transition-all"
          >
            <Play size={18} /> Start Training Process
          </button>
        )}
        {isTraining && (
          <div className="flex items-center gap-3 px-4 py-2 bg-blue-500/10 border border-blue-500/20 rounded-xl text-blue-400">
            <Activity className="animate-pulse" size={18} />
            <span className="font-mono text-sm">Epoch {trainingData.length > 0 ? trainingData[trainingData.length-1].epoch : 1}/15...</span>
          </div>
        )}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="bg-slate-800/40 p-6 rounded-3xl border border-slate-700 h-[400px]">
          <h3 className="font-bold text-sm mb-6 text-slate-400 uppercase tracking-widest">Accuracy Curve</h3>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={10} label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
              <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={10} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                itemStyle={{ fontSize: '12px' }}
              />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '10px', paddingBottom: '20px' }} />
              <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={3} dot={false} animationDuration={300} name="Train Acc" />
              <Line type="monotone" dataKey="val_accuracy" stroke="#10b981" strokeWidth={3} dot={false} animationDuration={300} name="Val Acc" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div className="bg-slate-800/40 p-6 rounded-3xl border border-slate-700 h-[400px]">
          <h3 className="font-bold text-sm mb-6 text-slate-400 uppercase tracking-widest">Loss Curve</h3>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={10} />
              <YAxis stroke="#94a3b8" fontSize={10} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                itemStyle={{ fontSize: '12px' }}
              />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '10px', paddingBottom: '20px' }} />
              <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={3} dot={false} animationDuration={300} name="Train Loss" />
              <Line type="monotone" dataKey="val_loss" stroke="#f59e0b" strokeWidth={3} dot={false} animationDuration={300} name="Val Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="flex justify-end gap-3 pt-6 border-t border-slate-800">
        <button onClick={prevStep} className="px-6 py-2 text-slate-400 hover:text-white transition-colors">Back</button>
        <button 
          onClick={nextStep} 
          disabled={isTraining || trainingData.length === 0}
          className={`px-8 py-2 rounded-lg font-bold transition-all ${
            !isTraining && trainingData.length > 0 
              ? 'bg-blue-600 hover:bg-blue-500 text-white' 
              : 'bg-slate-800 text-slate-500 cursor-not-allowed'
          }`}
        >
          Evaluate Model
        </button>
      </div>
    </div>
  );

  const renderEvaluation = () => (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Step 5: Testing & Prediction</h2>
        <div className="text-slate-400 text-sm">Run inference on new, unseen data.</div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="space-y-6">
          <div className="bg-slate-800/40 p-8 rounded-3xl border-2 border-dashed border-slate-700 flex flex-col items-center justify-center min-h-[300px] relative group hover:border-blue-500/50 transition-colors">
            {testImage ? (
              <div className="relative w-full h-full flex items-center justify-center">
                <img src={testImage} alt="Test" className="max-h-[250px] rounded-xl object-contain shadow-2xl" />
                <button 
                  onClick={() => setTestImage(null)} 
                  className="absolute top-0 right-0 p-2 bg-red-500 text-white rounded-full translate-x-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <RefreshCw size={14} />
                </button>
              </div>
            ) : (
              <label className="cursor-pointer flex flex-col items-center gap-4">
                <div className="w-16 h-16 bg-slate-700 rounded-full flex items-center justify-center text-slate-400">
                  <Upload size={28} />
                </div>
                <div className="text-center">
                  <span className="font-bold text-white">Click to upload test image</span>
                  <p className="text-xs text-slate-500 mt-1">PNG, JPG or JPEG (Max 5MB)</p>
                </div>
                <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
              </label>
            )}
          </div>
          
          <button
            onClick={handlePredict}
            disabled={!testImage || isAnalyzing}
            className={`w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${
              testImage && !isAnalyzing 
                ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20' 
                : 'bg-slate-800 text-slate-500 cursor-not-allowed'
            }`}
          >
            {isAnalyzing ? (
              <>
                <Zap size={20} className="animate-spin text-amber-400" /> Analyzing via CNN feature maps...
              </>
            ) : (
              <>
                <Zap size={20} /> Run Prediction
              </>
            )}
          </button>
        </div>

        <div className="space-y-4">
          {prediction ? (
            <div className="bg-slate-800/80 p-8 rounded-3xl border border-slate-700 animate-in zoom-in-95 duration-300">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Model Output</h3>
                  <div className="text-3xl font-extrabold text-emerald-400">{prediction.label}</div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Confidence</div>
                  <div className="text-2xl font-mono text-white">{(prediction.confidence * 100).toFixed(1)}%</div>
                </div>
              </div>
              
              <div className="p-4 bg-slate-900 rounded-2xl mb-6">
                <p className="text-sm text-slate-300 leading-relaxed italic">
                  "{prediction.explanation}"
                </p>
              </div>
              
              <div className="space-y-3">
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Detected Visual Features</h4>
                <div className="flex flex-wrap gap-2">
                  {prediction.features.map((feat, i) => (
                    <span key={i} className="px-3 py-1 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-full text-xs font-medium">
                      {feat}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-800/30 p-8 rounded-3xl border border-slate-800 h-full flex flex-col items-center justify-center text-center">
              <ImageIcon size={48} className="text-slate-700 mb-4" />
              <p className="text-slate-500 max-w-xs">Upload an image and run prediction to see the model's logic and extracted features.</p>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-end gap-3 pt-6 border-t border-slate-800">
        <button onClick={prevStep} className="px-6 py-2 text-slate-400 hover:text-white transition-colors">Back</button>
        <button onClick={nextStep} className="px-8 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold">Write Summary</button>
      </div>
    </div>
  );

  const renderSummary = () => (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Final Project Deliverables</h2>
        <div className="text-slate-400 text-sm">Wrap up your findings.</div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="space-y-4">
          <div className="bg-slate-800/40 p-6 rounded-2xl border border-slate-700">
            <h3 className="font-bold mb-3 flex items-center gap-2">
              <Info size={18} className="text-blue-400" />
              Learning Reflection
            </h3>
            <p className="text-sm text-slate-400 mb-4">
              Describe what you learned about CNNs, pooling layers, and model training.
            </p>
            <textarea
              value={summaryText}
              onChange={(e) => setSummaryText(e.target.value)}
              placeholder="e.g., I learned that adding more dropout layers helped reduce overfitting, and spatial pooling is essential for reducing parameter count..."
              className="w-full h-40 bg-slate-900 border border-slate-700 rounded-xl p-4 text-sm text-white focus:outline-none focus:border-blue-500 transition-colors"
            />
          </div>
          <button 
            onClick={handleGenerateReport}
            disabled={!summaryText || isAnalyzing}
            className="w-full py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-bold transition-all disabled:opacity-50"
          >
            {isAnalyzing ? 'Analyzing Reflection...' : 'Generate AI Learning Report'}
          </button>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 overflow-y-auto max-h-[500px]">
          <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-4">
            <CheckCircle2 className="text-emerald-500" size={24} />
            <h3 className="text-xl font-bold">Learning Outcomes</h3>
          </div>
          {finalReport ? (
            <div className="prose prose-invert text-slate-300 whitespace-pre-wrap text-sm leading-relaxed">
              {finalReport}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center py-20 text-center">
               <Cpu size={40} className="text-slate-800 mb-4" />
               <p className="text-slate-600 text-sm">Your AI-generated summary report will appear here once you submit your reflection.</p>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-end gap-3 pt-6 border-t border-slate-800">
        <button onClick={prevStep} className="px-6 py-2 text-slate-400 hover:text-white transition-colors">Back</button>
        <button onClick={() => window.location.reload()} className="px-8 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold">Restart Project</button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-6xl mx-auto">
      <Header />
      <StepIndicator current={step} />
      
      <main className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-3xl p-6 md:p-10 shadow-2xl relative overflow-hidden">
        {/* Decorative elements */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-600/5 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-purple-600/5 blur-[120px] rounded-full translate-y-1/2 -translate-x-1/2" />
        
        {step === ProjectStep.INTRO && renderIntro()}
        {step === ProjectStep.DATASET && renderDataset()}
        {step === ProjectStep.PREPROCESSING && renderPreprocessing()}
        {step === ProjectStep.ARCHITECTURE && renderArchitecture()}
        {step === ProjectStep.TRAINING && renderTraining()}
        {step === ProjectStep.EVALUATION && renderEvaluation()}
        {step === ProjectStep.SUMMARY && renderSummary()}
      </main>

      <footer className="mt-8 text-center text-slate-500 text-xs">
        <p>© 2024 CNN Interactive Project • Built with Gemini 3 for Advanced Reasoning</p>
      </footer>
    </div>
  );
};

export default App;
