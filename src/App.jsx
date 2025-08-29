/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-unused-vars */

import React, { useState, useCallback, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { FileUp } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

import {
  preprocessText,
  createTextSequences,
  parseCSV,
  normalizeData,
  createTimeSeriesSequences,
  calculateRMSE,
  calculateR2,
} from "./utils";

import Footer from "./components/Footer";

// --- React Components ---
const IconZap = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="h-5 w-5 mr-2"
  >
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
  </svg>
);
const IconCpu = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="h-5 w-5 mr-2"
  >
    <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
    <rect x="9" y="9" width="6" height="6"></rect>
    <line x1="9" y1="1" x2="9" y2="4"></line>
    <line x1="15" y1="1" x2="15" y2="4"></line>
    <line x1="9" y1="20" x2="9" y2="23"></line>
    <line x1="15" y1="20" x2="15" y2="23"></line>
    <line x1="20" y1="9" x2="23" y2="9"></line>
    <line x1="20" y1="14" x2="23" y2="14"></line>
    <line x1="1" y1="9" x2="4" y2="9"></line>
    <line x1="1" y1="14" x2="4" y2="14"></line>
  </svg>
);
const IconFileText = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="h-5 w-5 mr-2"
  >
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
    <line x1="16" y1="13" x2="8" y2="13"></line>
    <line x1="16" y1="17" x2="8" y2="17"></line>
    <polyline points="10 9 9 9 8 9"></polyline>
  </svg>
);
const IconTrendingUp = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="h-5 w-5 mr-2"
  >
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
    <polyline points="17 6 23 6 23 12"></polyline>
  </svg>
);

const MessageBox = ({ message, type }) => {
  if (!message) return null;
  const baseStyle = "p-4 rounded-lg my-4 text-sm";
  const typeStyles = {
    info: "bg-blue-100 border border-blue-400 text-blue-700",
    error: "bg-red-100 border border-red-400 text-red-700",
    success: "bg-green-100 border border-green-400 text-green-700",
  };
  return (
    <div className={`${baseStyle} ${typeStyles[type] || typeStyles.info}`}>
      {message}
    </div>
  );
};

const HyperparameterControls = ({ params, setParams, isTraining, useCase }) => {
  const {
    epochs,
    learningRate,
    lstmUnits,
    sequenceLength,
    sequenceLengthPercent,
  } = params;
  const handleParamChange = (param, value) => {
    setParams((prev) => ({ ...prev, [param]: Number(value) }));
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <label
          htmlFor="epochs"
          className="block text-sm font-medium text-gray-700"
        >
          Epochs: {epochs}
        </label>
        <input
          id="epochs"
          type="range"
          min="1"
          max="200"
          value={epochs}
          onChange={(e) => handleParamChange("epochs", e.target.value)}
          disabled={isTraining}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>
      <div>
        <label
          htmlFor="learningRate"
          className="block text-sm font-medium text-gray-700"
        >
          Learning Rate: {learningRate}
        </label>
        <input
          id="learningRate"
          type="range"
          min="0.001"
          max="0.1"
          step="0.001"
          value={learningRate}
          onChange={(e) => handleParamChange("learningRate", e.target.value)}
          disabled={isTraining}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>
      <div>
        <label
          htmlFor="lstmUnits"
          className="block text-sm font-medium text-gray-700"
        >
          LSTM Units: {lstmUnits}
        </label>
        <input
          id="lstmUnits"
          type="range"
          min="8"
          max="128"
          step="8"
          value={lstmUnits}
          onChange={(e) => handleParamChange("lstmUnits", e.target.value)}
          disabled={isTraining}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>
      {useCase === "time-series" ? (
        <div>
          <label
            htmlFor="sequenceLengthPercent"
            className="block text-sm font-medium text-gray-700"
          >
            Sequence Length: {sequenceLengthPercent}% of Training Data
          </label>
          <input
            id="sequenceLengthPercent"
            type="range"
            min="1"
            max="25"
            step="1"
            value={sequenceLengthPercent}
            onChange={(e) =>
              handleParamChange("sequenceLengthPercent", e.target.value)
            }
            disabled={isTraining}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-gray-500 mt-1">
            Percentage of training data to use as the lookback window.
          </p>
        </div>
      ) : (
        <div>
          <label
            htmlFor="sequenceLength"
            className="block text-sm font-medium text-gray-700"
          >
            Sequence Length: {sequenceLength} words
          </label>
          <input
            id="sequenceLength"
            type="range"
            min="2"
            max="50"
            value={sequenceLength}
            onChange={(e) =>
              handleParamChange("sequenceLength", e.target.value)
            }
            disabled={isTraining}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-gray-500 mt-1">
            Number of words to consider for predicting the next one.
          </p>
        </div>
      )}
    </div>
  );
};

export default function App() {
  const [useCase, setUseCase] = useState("next-word");

  // Shared state
  const [params, setParams] = useState({
    epochs: 50,
    learningRate: 0.01,
    lstmUnits: 32,
    sequenceLength: 10, // For text
    sequenceLengthPercent: 5, // For time series
  });
  const [model, setModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingLog, setTrainingLog] = useState([]);
  const [message, setMessage] = useState({
    text: 'Select a use case, provide data, and click "Train Model".',
    type: "info",
  });
  const [prediction, setPrediction] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const modelMetadata = useRef({});

  // Text-specific state
  const [textTrainingData, setTextTrainingData] = useState(
    "The quick brown fox jumps over the lazy dog. A stitch in time saves nine. An apple a day keeps the doctor away. The early bird catches the worm."
  );
  const [seedText, setSeedText] = useState("The quick brown fox");

  // Time-series-specific state
  const [csvData, setCsvData] = useState({ headers: [], data: [] });
  const [fileName, setFileName] = useState("");
  const [selectedColumn, setSelectedColumn] = useState("");
  const [trainTestSplit, setTrainTestSplit] = useState(80);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [testChartData, setTestChartData] = useState([]);

  const resetStateForUseCase = (newUseCase) => {
    setUseCase(newUseCase);
    setModel(null);
    setTrainingLog([]);
    setPrediction("");
    setMessage({
      text: "Provide data for the new use case and train the model.",
      type: "info",
    });
    setCsvData({ headers: [], data: [] });
    setFileName("");
    setSelectedColumn("");
    setEvaluationResults(null);
    setTestChartData([]);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (event) => {
        const { headers, data } = parseCSV(event.target.result);
        if (headers.length > 0) {
          setCsvData({ headers, data });
          setSelectedColumn(headers[0]);
          setMessage({
            text: `CSV loaded. Select the column to predict and train the model.`,
            type: "info",
          });
        } else {
          setMessage({
            text: "Could not parse CSV file. Check format.",
            type: "error",
          });
        }
      };
      reader.readAsText(file);
    }
  };

  const handleTrain = useCallback(async () => {
    setIsTraining(true);
    setTrainingLog([]);
    setPrediction("");
    setModel(null);
    setMessage({ text: "Starting training...", type: "info" });
    await new Promise((resolve) => setTimeout(resolve, 10));

    try {
      if (useCase === "next-word") {
        await trainNextWordModel();
      } else if (useCase === "time-series") {
        await trainTimeSeriesModel();
      }
    } catch (error) {
      console.error("Training Error:", error);
      setMessage({
        text: `Error during training: ${error.message}`,
        type: "error",
      });
    } finally {
      setIsTraining(false);
    }
  }, [
    useCase,
    params,
    textTrainingData,
    csvData,
    selectedColumn,
    trainTestSplit,
  ]);

  const trainNextWordModel = async () => {
    const { sequenceLength } = params;
    if (textTrainingData.trim().split(/\s+/).length < sequenceLength + 1) {
      throw new Error(
        `Training data must contain at least ${sequenceLength + 1} words.`
      );
    }
    const { words, vocab, wordToIndex, indexToWord } =
      preprocessText(textTrainingData);
    modelMetadata.current = {
      wordToIndex,
      indexToWord,
      vocabSize: vocab.length,
      sequenceLength, // Store for prediction
      useCase: "next-word",
    };

    setMessage({
      text: `Data preprocessed. Vocab size: ${vocab.length}. Creating sequences...`,
      type: "info",
    });
    const { X, y } = createTextSequences(words, wordToIndex, sequenceLength);

    const X_tensor = tf.tensor2d(X, [X.length, sequenceLength]);
    const y_tensor = tf.oneHot(tf.tensor1d(y, "int32"), vocab.length);

    const newModel = tf.sequential();
    newModel.add(
      tf.layers.embedding({
        inputDim: vocab.length,
        outputDim: 8,
        inputLength: sequenceLength,
      })
    );
    newModel.add(
      tf.layers.lstm({ units: params.lstmUnits, returnSequences: false })
    );
    newModel.add(
      tf.layers.dense({ units: vocab.length, activation: "softmax" })
    );

    newModel.compile({
      optimizer: tf.train.adam(params.learningRate),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    setMessage({
      text: `Model compiled. Starting training for ${params.epochs} epochs...`,
      type: "info",
    });
    await newModel.fit(X_tensor, y_tensor, {
      epochs: params.epochs,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          setTrainingLog((prev) => [
            ...prev,
            {
              epoch: epoch + 1,
              loss: logs.loss.toFixed(4),
              acc: logs.acc.toFixed(4),
            },
          ]);
        },
      },
    });

    setModel(newModel);
    setMessage({
      text: "Training complete! You can now make predictions.",
      type: "success",
    });
  };

  const trainTimeSeriesModel = async () => {
    if (csvData.data.length === 0 || !selectedColumn) {
      throw new Error("Please upload a CSV file and select a column.");
    }
    setEvaluationResults(null);
    setTestChartData([]);

    const series = csvData.data
      .map((row) => parseFloat(row[selectedColumn]))
      .filter((v) => !isNaN(v));

    const { normalizedData, min, max } = normalizeData(series);

    const splitIndex = Math.floor(
      normalizedData.length * (trainTestSplit / 100)
    );
    const trainData = normalizedData.slice(0, splitIndex);
    const testData = normalizedData.slice(splitIndex); // Note: we'll adjust this for sequencing

    // --- CRITICAL: Calculate sequence length AFTER splitting data ---
    const sequenceLengthInSteps = Math.max(
      2,
      Math.floor(trainData.length * (params.sequenceLengthPercent / 100))
    );
    setMessage({
      text: `Using sequence length of ${sequenceLengthInSteps} steps.`,
      type: "info",
    });
    await new Promise((resolve) => setTimeout(resolve, 10));

    if (series.length < sequenceLengthInSteps + 1) {
      throw new Error(
        `The selected column must have at least ${
          sequenceLengthInSteps + 1
        } numeric values to create sequences.`
      );
    }

    const testDataWithOverlap = normalizedData.slice(
      splitIndex - sequenceLengthInSteps
    );

    const { X: X_train, y: y_train } = createTimeSeriesSequences(
      trainData,
      sequenceLengthInSteps
    );
    const { X: X_test, y: y_test } = createTimeSeriesSequences(
      testDataWithOverlap,
      sequenceLengthInSteps
    );

    if (X_train.length === 0 || X_test.length === 0) {
      throw new Error(
        "Not enough data to create train/test splits. Try adjusting the train/test split or sequence length percentage."
      );
    }

    const X_train_tensor = tf
      .tensor2d(X_train)
      .reshape([X_train.length, sequenceLengthInSteps, 1]);
    const y_train_tensor = tf.tensor1d(y_train);
    const X_test_tensor = tf
      .tensor2d(X_test)
      .reshape([X_test.length, sequenceLengthInSteps, 1]);
    const y_test_tensor = tf.tensor1d(y_test);

    const newModel = tf.sequential();
    newModel.add(
      tf.layers.lstm({
        units: params.lstmUnits,
        inputShape: [sequenceLengthInSteps, 1],
      })
    );
    newModel.add(tf.layers.dense({ units: 1 }));

    newModel.compile({
      optimizer: tf.train.adam(params.learningRate),
      loss: "meanSquaredError",
    });

    setMessage({
      text: `Model compiled. Training on ${X_train.length} samples...`,
      type: "info",
    });
    await newModel.fit(X_train_tensor, y_train_tensor, {
      epochs: params.epochs,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          setTrainingLog((prev) => [
            ...prev,
            { epoch: epoch + 1, loss: logs.loss.toFixed(6) },
          ]);
        },
      },
    });

    setMessage({
      text: "Training complete. Evaluating on test data...",
      type: "info",
    });
    await new Promise((resolve) => setTimeout(resolve, 10));

    const normalizedPredictions = newModel.predict(X_test_tensor);

    const denormalize = (tensor) => tensor.mul(max - min).add(min);
    const y_test_denorm = denormalize(y_test_tensor);
    const predictions_denorm = denormalize(normalizedPredictions);

    const y_test_array = await y_test_denorm.array();
    const predictions_array = await predictions_denorm.array();

    const rmse = calculateRMSE(y_test_denorm, predictions_denorm);
    const r2 = calculateR2(y_test_denorm, predictions_denorm);

    setEvaluationResults({ rmse, r2 });

    const chartData = y_test_array.map((actual, i) => ({
      index: i,
      actual: actual,
      predicted: predictions_array[i],
    }));
    setTestChartData(chartData);

    modelMetadata.current = {
      min,
      max,
      useCase: "time-series",
      sequenceLength: sequenceLengthInSteps, // Store the calculated steps
      lastSequence: normalizedData.slice(-sequenceLengthInSteps),
    };

    setModel(newModel);
    setMessage({
      text: "Evaluation complete! See test results below. You can also predict the next value.",
      type: "success",
    });
  };

  const handlePredict = useCallback(async () => {
    if (!model) {
      setMessage({ text: "Model not trained yet.", type: "error" });
      return;
    }
    setIsPredicting(true);
    setPrediction("...");
    await new Promise((resolve) => setTimeout(resolve, 10));

    try {
      if (modelMetadata.current.useCase === "next-word") {
        await predictNextWord();
      } else if (modelMetadata.current.useCase === "time-series") {
        await predictNextValue();
      }
    } catch (error) {
      console.error("Prediction Error:", error);
      setMessage({ text: `Prediction Error: ${error.message}`, type: "error" });
      setPrediction("");
    } finally {
      setIsPredicting(false);
    }
  }, [model, seedText]);

  const predictNextWord = async () => {
    const { wordToIndex, indexToWord, sequenceLength } = modelMetadata.current;
    let inputWords = seedText
      .toLowerCase()
      .replace(/[^a-z\s]/g, "")
      .split(/\s+/)
      .filter(Boolean);
    if (inputWords.length < sequenceLength) {
      throw new Error(`Seed text must have at least ${sequenceLength} words.`);
    }
    inputWords = inputWords.slice(-sequenceLength);
    const inputIndices = inputWords.map((word) => wordToIndex.get(word));
    if (inputIndices.some((idx) => idx === undefined)) {
      throw new Error(
        "Seed text contains words not in the training vocabulary."
      );
    }
    const inputTensor = tf.tensor2d([inputIndices], [1, sequenceLength]);
    const predictionTensor = model.predict(inputTensor);
    const predictedIndex = await predictionTensor.argMax(-1).data();
    setPrediction(indexToWord.get(predictedIndex[0]));
  };

  const predictNextValue = async () => {
    const { min, max, lastSequence, sequenceLength } = modelMetadata.current;
    const inputTensor = tf
      .tensor2d([lastSequence])
      .reshape([1, sequenceLength, 1]);
    const predictionTensor = model.predict(inputTensor);
    const normalizedPred = await predictionTensor.data();

    const denormalizedPred = tf
      .tensor1d(normalizedPred)
      .mul(tf.scalar(max - min))
      .add(tf.scalar(min));
    const finalValue = await denormalizedPred.data();

    setPrediction(finalValue[0].toFixed(4));
  };

  return (
    <div className="bg-gray-50 min-h-screen font-sans text-gray-800 flex flex-col">
      <div className="flex-grow">
        <div className="container mx-auto p-4 md:p-8">
          <header className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900">
              LSTM Playground
            </h1>
            <p className="text-lg text-gray-600 mt-2">
              Experiment with LSTMs directly in your window.
            </p>
          </header>

          <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
            {/* All app content goes here */}
            <div className="mb-6 border-b pb-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-3">
                0. Select Use Case
              </h2>
              <div className="flex space-x-2 rounded-lg bg-gray-200 p-1">
                <button
                  onClick={() => resetStateForUseCase("next-word")}
                  className={`w-full py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                    useCase === "next-word"
                      ? "bg-white text-blue-600 shadow"
                      : "text-gray-600 hover:bg-gray-300"
                  }`}
                >
                  Next Word Prediction
                </button>
                <button
                  onClick={() => resetStateForUseCase("time-series")}
                  className={`w-full py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                    useCase === "time-series"
                      ? "bg-white text-blue-600 shadow"
                      : "text-gray-600 hover:bg-gray-300"
                  }`}
                >
                  Time Series Forecasting
                </button>
              </div>
            </div>

            {useCase === "next-word" && (
              <div className="mb-6">
                <h2 className="text-2xl font-semibold flex items-center text-gray-800">
                  <IconFileText />
                  1. Training Data
                </h2>
                <p className="text-sm text-gray-500 mb-3">
                  Provide the text the model will learn from.
                </p>
                <textarea
                  value={textTrainingData}
                  onChange={(e) => setTextTrainingData(e.target.value)}
                  disabled={isTraining}
                  className="w-full h-40 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                />
              </div>
            )}

            {useCase === "time-series" && (
              <div className="mb-6">
                <h2 className="text-2xl font-semibold flex items-center text-gray-800 space-x-2">
                  <IconTrendingUp />
                  <span>1. Training Data</span>
                </h2>
                <p className="text-sm text-gray-500 mb-3">
                  Upload a CSV file with a header row. Select the column with
                  numerical data to predict.
                </p>

                <div
                  onDrop={(e) => {
                    e.preventDefault();
                    const file = e.dataTransfer.files[0];
                    if (file && file.type === "text/csv") {
                      handleFileChange({ target: { files: [file] } });
                    }
                  }}
                  onDragOver={(e) => e.preventDefault()}
                  className={`w-full h-40 flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl transition ${
                    isTraining
                      ? "cursor-not-allowed bg-gray-100"
                      : "cursor-pointer bg-white hover:bg-gray-50"
                  } border-gray-300 mb-4 text-center`}
                >
                  <FileUp className="h-8 w-8 text-gray-400 mb-2" />
                  <p className="text-gray-600">
                    {fileName ? (
                      <>
                        <strong>{fileName}</strong>
                        <br />
                        Drag to replace or use the button below
                      </>
                    ) : (
                      <>
                        <strong>Drag & drop</strong> your CSV file here or click
                        below to upload
                      </>
                    )}
                  </p>
                </div>

                <div className="flex items-center space-x-4">
                  <label className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-2 px-4 border border-gray-300 rounded-lg cursor-pointer">
                    <span>{fileName ? "Change File" : "Upload CSV"}</span>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileChange}
                      disabled={isTraining}
                      className="hidden"
                    />
                  </label>
                  {fileName && (
                    <span className="text-gray-600">{fileName}</span>
                  )}
                </div>

                {csvData.headers.length > 0 && (
                  <div className="mt-4">
                    <label
                      htmlFor="column-select"
                      className="block text-sm font-medium text-gray-700"
                    >
                      Column to Predict:
                    </label>
                    <select
                      id="column-select"
                      value={selectedColumn}
                      onChange={(e) => setSelectedColumn(e.target.value)}
                      disabled={isTraining}
                      className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                    >
                      {csvData.headers.map((h) => (
                        <option key={h} value={h}>
                          {h}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {csvData.headers.length > 0 && (
                  <div className="mt-6">
                    <label
                      htmlFor="trainTestSplit"
                      className="block text-sm font-medium text-gray-700"
                    >
                      Train/Test Split:{" "}
                      <span className="font-bold">{trainTestSplit}% Train</span>{" "}
                      / {100 - trainTestSplit}% Test
                    </label>
                    <input
                      id="trainTestSplit"
                      type="range"
                      min="10"
                      max="90"
                      step="5"
                      value={trainTestSplit}
                      onChange={(e) =>
                        setTrainTestSplit(Number(e.target.value))
                      }
                      disabled={isTraining}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer mt-1"
                    />
                  </div>
                )}
              </div>
            )}

            <div className="mb-8">
              <h2 className="text-2xl font-semibold flex items-center text-gray-800">
                <IconCpu />
                2. Hyperparameters
              </h2>
              <HyperparameterControls
                params={params}
                setParams={setParams}
                isTraining={isTraining}
                useCase={useCase}
              />
            </div>

            <div className="text-center mb-8">
              <button
                onClick={handleTrain}
                disabled={isTraining}
                className="bg-blue-600 text-white font-bold py-3 px-8 rounded-lg shadow-md hover:bg-blue-700 disabled:bg-gray-400 flex items-center justify-center mx-auto"
              >
                {isTraining ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Training...
                  </>
                ) : (
                  "Train Model"
                )}
              </button>
            </div>

            <MessageBox message={message.text} type={message.type} />

            {trainingLog.length > 0 && (
              <div className="mb-8">
                <h3 className="text-xl font-semibold text-gray-700 mb-4">
                  Training Progress
                </h3>
                <div className="h-64 w-full bg-gray-50 p-2 rounded-lg border">
                  <ResponsiveContainer>
                    <LineChart
                      data={trainingLog}
                      margin={{ top: 5, right: 20, left: -10, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis yAxisId="left" domain={["auto", "auto"]} />
                      {useCase === "next-word" && (
                        <YAxis
                          yAxisId="right"
                          orientation="right"
                          domain={[0, 1]}
                        />
                      )}
                      <Tooltip />
                      <Legend />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="loss"
                        stroke="#ef4444"
                        strokeWidth={2}
                        name="Loss"
                      />
                      {useCase === "next-word" && (
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="acc"
                          stroke="#22c55e"
                          strokeWidth={2}
                          name="Accuracy"
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {evaluationResults && testChartData.length > 0 && (
              <div className="mb-8">
                <h3 className="text-xl font-semibold text-gray-700 mb-4">
                  Test Set Evaluation
                </h3>
                <div className=" mb-4 text-center">
                  <div className="p-4 bg-gray-100 rounded-lg">
                    <p className="text-sm text-gray-600">RMSE</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {evaluationResults.rmse.toFixed(4)}
                    </p>
                  </div>
                  {/* <div className="p-4 bg-gray-100 rounded-lg">
                    <p className="text-sm text-gray-600">R-Squared (R²)</p>
                    <p className="text-2xl font-bold text-green-600">
                      {evaluationResults.r2.toFixed(4)}
                    </p>
                  </div> */}
                </div>

                <div className="h-80 w-full bg-gray-50 p-2 rounded-lg border">
                  <ResponsiveContainer>
                    <LineChart
                      data={testChartData}
                      margin={{ top: 5, right: 20, left: -10, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="index"
                        label={{
                          value: "Test Data Point Index",
                          position: "insideBottom",
                          offset: -5,
                        }}
                      />
                      <YAxis domain={["auto", "auto"]} />
                      <Tooltip
                        formatter={(value) =>
                          typeof value === "number" ? value.toFixed(2) : value
                        }
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#ef4444"
                        strokeWidth={2}
                        name="Actual Values"
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        name="Predicted Values"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {model && !isTraining && (
              <div className="mt-8 pt-6 border-t border-gray-200">
                <h2 className="text-2xl font-semibold flex items-center text-gray-800">
                  <IconZap />
                  3. Prediction
                </h2>
                {useCase === "next-word" && (
                  <>
                    <p className="text-sm text-gray-500 mb-3">
                      Provide a starting sequence to predict the next word.
                    </p>
                    <div className="flex flex-col sm:flex-row items-center gap-4">
                      <input
                        type="text"
                        value={seedText}
                        onChange={(e) => setSeedText(e.target.value)}
                        className="flex-grow w-full p-3 border border-gray-300 rounded-lg"
                        placeholder="Enter seed text..."
                      />
                      <button
                        onClick={handlePredict}
                        disabled={isPredicting}
                        className="bg-green-600 text-white font-bold py-3 px-6 rounded-lg shadow-md hover:bg-green-700 disabled:bg-gray-400 w-full sm:w-auto"
                      >
                        {isPredicting ? "Predicting..." : "Predict Next Word"}
                      </button>
                    </div>
                  </>
                )}
                {useCase === "time-series" && (
                  <div className="text-center">
                    <p className="text-sm text-gray-500 mb-3">
                      The model will predict the next value in the sequence
                      based on the uploaded data.
                    </p>
                    <button
                      onClick={handlePredict}
                      disabled={isPredicting}
                      className="bg-green-600 text-white font-bold py-3 px-6 rounded-lg shadow-md hover:bg-green-700 disabled:bg-gray-400"
                    >
                      {isPredicting ? "Predicting..." : "Predict Next Value"}
                    </button>
                  </div>
                )}
                {prediction && (
                  <div className="mt-4 p-4 bg-gray-100 rounded-lg text-center">
                    <p className="text-gray-600">
                      Predicted Next{" "}
                      {useCase === "next-word" ? "Word" : "Value"}:
                    </p>
                    <p className="text-2xl font-bold text-indigo-600">
                      {prediction}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}
