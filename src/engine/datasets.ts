/**
 * Dataset presets and synthetic data generation for MiniDeepEngine
 */

import { Matrix } from './matrix';

export interface Dataset {
  id: string;
  name: string;
  nameAr: string;
  type: 'classification' | 'regression';
  X: Matrix;
  y: Matrix;
  description: string;
  descriptionAr: string;
  domain: { xMin: number; xMax: number; yMin: number; yMax: number };
}

export function generateXOR(): Dataset {
  return {
    id: 'xor',
    name: 'XOR Problem',
    nameAr: 'معضلة XOR (غير الخطية)',
    type: 'classification',
    X: [
      [0.0, 0.0],
      [0.0, 1.0],
      [1.0, 0.0],
      [1.0, 1.0],
    ],
    y: [
      [0.0],
      [1.0],
      [1.0],
      [0.0],
    ],
    description: 'The canonical non-linear classification problem that requires a hidden layer to solve.',
    descriptionAr: 'المشكلة الكلاسيكية غير الخطية التي أثبتت حتمية وجود طبقة مخفية لتعلم العلاقات المعقدة.',
    domain: { xMin: -0.5, xMax: 1.5, yMin: -0.5, yMax: 1.5 },
  };
}

export function generateCircle(nSamples = 100, noise = 0.05): Dataset {
  const X: Matrix = [];
  const y: Matrix = [];

  for (let i = 0; i < nSamples; i++) {
    const isInner = i % 2 === 0;
    const r = isInner ? (0.3 + (Math.random() - 0.5) * noise) : (0.8 + (Math.random() - 0.5) * noise);
    const theta = Math.random() * 2 * Math.PI;
    const x1 = r * Math.cos(theta);
    const x2 = r * Math.sin(theta);

    X.push([Number(x1.toFixed(3)), Number(x2.toFixed(3))]);
    y.push([isInner ? 1.0 : 0.0]);
  }

  return {
    id: 'circle',
    name: 'Concentric Circles',
    nameAr: 'الدوائر المتداخلة (Concentric Circles)',
    type: 'classification',
    X,
    y,
    description: 'Circular boundary requiring non-linear activations (like ReLU or Tanh).',
    descriptionAr: 'تصنيف دائري يختبر قدرة الشبكة على رسم حدود قرار منحنية ومغلقة.',
    domain: { xMin: -1.2, xMax: 1.2, yMin: -1.2, yMax: 1.2 },
  };
}

export function generateMoons(nSamples = 100, noise = 0.08): Dataset {
  const X: Matrix = [];
  const y: Matrix = [];

  for (let i = 0; i < nSamples; i++) {
    const isUpper = i % 2 === 0;
    const angle = Math.random() * Math.PI;

    if (isUpper) {
      const x1 = Math.cos(angle) + (Math.random() - 0.5) * noise;
      const x2 = Math.sin(angle) + (Math.random() - 0.5) * noise;
      X.push([Number(x1.toFixed(3)), Number(x2.toFixed(3))]);
      y.push([1.0]);
    } else {
      const x1 = 1 - Math.cos(angle) + (Math.random() - 0.5) * noise;
      const x2 = 0.5 - Math.sin(angle) + (Math.random() - 0.5) * noise;
      X.push([Number(x1.toFixed(3)), Number(x2.toFixed(3))]);
      y.push([0.0]);
    }
  }

  return {
    id: 'moons',
    name: 'Two Moons',
    nameAr: 'شكل الهلالين (Two Moons)',
    type: 'classification',
    X,
    y,
    description: 'Interlocking crescent moon shapes.',
    descriptionAr: 'نمط هلالي مزدوج متداخل يتطلب استيعاب الانحناءات الهندسية بدقة.',
    domain: { xMin: -1.5, xMax: 2.5, yMin: -1.0, yMax: 1.5 },
  };
}

export function generatePromptRegression(): Dataset {
  return {
    id: 'prompt_sample',
    name: 'Prompt Numeric Dataset',
    nameAr: 'بيانات المقال الرقمية (Linear Regression)',
    type: 'regression',
    X: [
      [1.0, 2.0],
      [2.0, 1.0],
      [3.0, 3.0],
      [4.0, 2.0],
    ],
    y: [
      [5.0],
      [4.0],
      [8.0],
      [7.0],
    ],
    description: 'Exact numeric dataset from the user prompt: relation between 2 inputs and target scalar.',
    descriptionAr: 'البيانات الرقمية التجريبية الدقيقة الواردة في منشور المستخدم لتدريب الانحدار الخطي.',
    domain: { xMin: 0.0, xMax: 5.0, yMin: 0.0, yMax: 4.0 },
  };
}

export function generateSineWave(nSamples = 50): Dataset {
  const X: Matrix = [];
  const y: Matrix = [];

  for (let i = 0; i < nSamples; i++) {
    const xVal = -Math.PI + (2 * Math.PI * i) / (nSamples - 1);
    const noise = (Math.random() - 0.5) * 0.1;
    const yVal = Math.sin(xVal) + noise;

    X.push([Number(xVal.toFixed(3)), 0]);
    y.push([Number(yVal.toFixed(3))]);
  }

  return {
    id: 'sine',
    name: 'Sine Curve Fitting',
    nameAr: 'انحدار الدالة الجيبية (Sine Curve)',
    type: 'regression',
    X,
    y,
    description: 'Non-linear function approximation learning the sine wave.',
    descriptionAr: 'تقريب الدوال غير الخطية: تدريب الشبكة العصبية على ملاءمة منحنى دالة الجيب sin(x).',
    domain: { xMin: -3.5, xMax: 3.5, yMin: -1.5, yMax: 1.5 },
  };
}

export function generateRealEstatePricing(): Dataset {
  return {
    id: 'real_estate_pricing',
    name: 'Real Estate & Apartment Pricing',
    nameAr: 'تسعير العقارات والشقق (Real Estate Regression)',
    type: 'regression',
    X: [
      [60, 2],
      [80, 3],
      [100, 3],
      [120, 4],
      [150, 5],
      [90, 2],
    ],
    y: [
      [12],
      [16],
      [20],
      [25],
      [32],
      [18],
    ],
    description: 'Predicting real estate prices based on apartment area (m²) and number of rooms using MLP Regression & Adam Optimizer.',
    descriptionAr: 'تطبيق عملي: التنبؤ بأسعار الشقق والعقارات بناءً على المساحة (م²) وعدد الغرف باستخدام شبكة عصبية متعددة الطبقات وخوارزمية Adam.',
    domain: { xMin: 50, xMax: 160, yMin: 1, yMax: 6 },
  };
}

export function generateBehaviorClassification(): Dataset {
  return {
    id: 'behavior_classification',
    name: 'Customer Behavior & Binary Classification',
    nameAr: 'تصنيف السلوك والعملاء (Binary Classification)',
    type: 'classification',
    X: [
      [0.1, 0.2],
      [0.2, 0.1],
      [0.8, 0.9],
      [0.9, 0.8],
      [0.1, 0.9],
      [0.9, 0.2],
    ],
    y: [
      [0],
      [0],
      [1],
      [1],
      [0],
      [1],
    ],
    description: 'Binary classification of user behavior with Sigmoid activation and Binary Cross-Entropy loss.',
    descriptionAr: 'تطبيق عملي: تصنيف سلوك العملاء وفصل الفئات الثنائية باستخدام دالة Sigmoid ودالة خسارة Binary Cross-Entropy.',
    domain: { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 },
  };
}

export const DATASET_PRESETS: Dataset[] = [
  generateXOR(),
  generateBehaviorClassification(),
  generateRealEstatePricing(),
  generateCircle(),
  generateMoons(),
  generatePromptRegression(),
  generateSineWave(),
];
