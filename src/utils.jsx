import * as tf from "@tensorflow/tfjs";

// --- Text Use Case Helpers ---

export function preprocessText(text) {
  const cleaned = text.toLowerCase().replace(/[^a-z\s]/g, "");
  const words = cleaned.split(/\s+/).filter(Boolean);
  const vocab = [...new Set(words)];
  const wordToIndex = new Map(vocab.map((word, i) => [word, i]));
  const indexToWord = new Map(vocab.map((word, i) => [i, word]));
  return { words, vocab, wordToIndex, indexToWord };
}

export function createTextSequences(words, wordToIndex, sequenceLength) {
  const sequences = [];
  const nextWords = [];
  for (let i = 0; i < words.length - sequenceLength; i++) {
    sequences.push(words.slice(i, i + sequenceLength));
    nextWords.push(words[i + sequenceLength]);
  }
  const X = sequences.map((seq) => seq.map((word) => wordToIndex.get(word)));
  const y = nextWords.map((word) => wordToIndex.get(word));
  return { X, y };
}

// --- Time Series Use Case Helpers ---

export function parseCSV(csvText) {
  const lines = csvText.split("\n").filter((line) => line.trim() !== "");
  if (lines.length < 2) return { headers: [], data: [] };
  const headers = lines[0].split(",").map((h) => h.trim());
  const data = lines.slice(1).map((line) => {
    const values = line.split(",").map((v) => v.trim());
    const row = {};
    headers.forEach((header, i) => {
      row[header] = values[i];
    });
    return row;
  });
  return { headers, data };
}

export function normalizeData(data) {
  const tensor = tf.tensor1d(data);
  const min = tensor.min();
  const max = tensor.max();
  const normalized = tensor.sub(min).div(max.sub(min));
  return {
    normalizedData: normalized.arraySync(),
    min: min.arraySync(),
    max: max.arraySync(),
  };
}

export function createTimeSeriesSequences(data, sequenceLength) {
  const X = [];
  const y = [];
  for (let i = 0; i < data.length - sequenceLength; i++) {
    X.push(data.slice(i, i + sequenceLength));
    y.push(data[i + sequenceLength]);
  }
  return { X, y };
}

// --- Evaluation Metric Helpers ---

export function calculateRMSE(yTrue, yPred) {
  return tf.tidy(() => {
    const error = yTrue.sub(yPred);
    const squaredError = error.square();
    const meanSquaredError = squaredError.mean();
    const rmse = meanSquaredError.sqrt();
    return rmse.dataSync()[0];
  });
}

export function calculateR2(yTrue, yPred) {
  return tf.tidy(() => {
    const meanYTrue = yTrue.mean();
    const totalSumOfSquares = yTrue.sub(meanYTrue).square().sum();
    const residualSumOfSquares = yTrue.sub(yPred).square().sum();
    const r2 = tf.scalar(1).sub(residualSumOfSquares.div(totalSumOfSquares));
    return r2.dataSync()[0];
  });
}