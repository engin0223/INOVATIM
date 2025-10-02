% runSingleModelPrediction.m
% Use a single multi-class model (netArr trained on N + arr classes)

clear; clc;
BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL';
addpath(fullfile(BASEPATH, 'functions'));
addpath(fullfile(BASEPATH, 'functions', 'models'));
addpath(fullfile(BASEPATH, 'functions', 'data'));

datasetFile = "mitbih_beat_dataset.mat";
load("Model_netArr_v0.2.mat","netArr");  % ensure this model includes 'N' in classes

[X, Y] = loadDataset(datasetFile);

[X, mu, sigma] = normalizeData(X);

classNames = categories(categorical(cellstr(Y)));
N_all = size(X,1);

scoresAll = predictBatched(netArr, X, 1024);
[~, predIdx] = max(scoresAll, [], 2);
predLabels = classNames(predIdx)'; % as column

trueLabels = cellstr(Y);

[metricsTable, accuracy] = evaluateConfusionMetrics(trueLabels, predLabels);
fprintf("Overall Accuracy (Single Model): %.2f%%\n", accuracy*100);
disp(metricsTable);

figure; confusionchart(trueLabels, predLabels); title('Confusion Matrix (Single Model)');
