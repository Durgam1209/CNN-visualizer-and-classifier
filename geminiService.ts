
import { GoogleGenAI, Type } from "@google/genai";

const getAI = () => new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

export const analyzeClassification = async (
  imageData: string,
  dataset: string,
  architecture: string
) => {
  const ai = getAI();
  const prompt = `
    You are acting as a Convolutional Neural Network (CNN) model that has just been trained on the "${dataset}" dataset.
    The architecture used was: ${architecture}.
    
    Analyze the provided image and return a JSON object describing:
    1. The predicted label (be specific to the dataset).
    2. Confidence level (0 to 1).
    3. An explanation of what "features" the CNN layers likely detected (e.g., specific textures, shapes, or patterns characteristic of that class).
    4. 3-4 key visual features detected by the hypothetical feature maps.

    Provide the output strictly in JSON format.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: [
        {
          parts: [
            { text: prompt },
            { inlineData: { mimeType: 'image/jpeg', data: imageData.split(',')[1] } }
          ]
        }
      ],
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            label: { type: Type.STRING },
            confidence: { type: Type.NUMBER },
            explanation: { type: Type.STRING },
            features: {
              type: Type.ARRAY,
              items: { type: Type.STRING }
            }
          },
          required: ['label', 'confidence', 'explanation', 'features']
        }
      }
    });

    return JSON.parse(response.text);
  } catch (error) {
    console.error("Gemini Classification Error:", error);
    throw error;
  }
};

export const generateFinalReport = async (summary: string) => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Write a professional 5-7 line summary report on learning outcomes for a student who completed a CNN Image Classification project with these reflections: ${summary}. Focus on concepts like feature extraction, model optimization, and data augmentation.`
  });
  return response.text;
};
