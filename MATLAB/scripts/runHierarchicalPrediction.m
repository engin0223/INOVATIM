% runHierarchicalPrediction.m
clear; clc;
BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL';
addpath(fullfile(BASEPATH, 'functions'));
addpath(fullfile(BASEPATH, 'functions', 'models'));
addpath(fullfile(BASEPATH, 'functions', 'data'));

datasetFile = "beat_dataset_filtered.mat";
load("Model_netArr_v0.4.mat","net"); netArr = net;
load("Model_netBin_v0.3.mat","net"); netBin = net; clear net;

[X, Y] = loadDataset(datasetFile);

% Normalize
[X, mu, sigma] = normalizeData(X);

% --- Call hierarchicalPredict function
[predLabels, binScores, arrScores] = hierarchicalPredict(X, Y, netBin, netArr);

trueLabels = string(cellstr(Y));

% --- Evaluate
[metricsTable, accuracy] = evaluateConfusionMetrics(trueLabels, predLabels);
fprintf("Overall Accuracy: %.2f%%\n", accuracy*100);
disp(metricsTable);

% --- Confusion chart
figure; confusionchart(trueLabels, predLabels); title('Hierarchical Prediction Confusion Matrix');
