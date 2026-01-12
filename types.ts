
export enum ProjectStep {
  INTRO = 'INTRO',
  DATASET = 'DATASET',
  PREPROCESSING = 'PREPROCESSING',
  ARCHITECTURE = 'ARCHITECTURE',
  TRAINING = 'TRAINING',
  EVALUATION = 'EVALUATION',
  SUMMARY = 'SUMMARY'
}

export type DatasetType = 'cats-vs-dogs' | 'fruits' | 'flowers' | 'fashion-mnist';

export interface CNNLayer {
  id: string;
  type: 'Conv2D' | 'MaxPooling2D' | 'Flatten' | 'Dense' | 'Dropout';
  params: Record<string, any>;
}

export interface TrainingMetric {
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

export interface PredictionResult {
  label: string;
  confidence: number;
  explanation: string;
  features: string[];
}
