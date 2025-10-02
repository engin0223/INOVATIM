% runBinaryClassification.m
% Driver script to run binary classification training (Normal vs Arr)

clear; clc;
BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL';
addpath(fullfile(BASEPATH, 'functions'));
addpath(fullfile(BASEPATH, 'functions', 'models'));
addpath(fullfile(BASEPATH, 'functions', 'data'));

%--- Config
datasetFile = "beat_dataset_filtered.mat";
rngSeed = 42;
miniBatchSize = 32;

%--- Load & normalize
[X, Y] = loadDataset(datasetFile);
[X, mu, sigma] = normalizeData(X);

%--- Prepare binary labels and split
Y_binary = double(Y ~= 'N');
YCat = categorical(Y_binary);

[trainIdx, valIdx] = stratifiedSplit(YCat, 0.25, rngSeed);

%--- Prepare data for trainnet: [time x features x batch]
X_batch = permute(X,[2 3 1]); % [seqLen x features x N]
XTrain = gpuArray(single(X_batch(:,:,trainIdx)));
XVal   = gpuArray(single(X_batch(:,:,valIdx)));

YTrain_bin = YCat(trainIdx);
YVal_bin   = YCat(valIdx);

%--- Training options
opts = struct();
opts.MaxEpochs = 1000;
opts.InitialLearnRate = 1e-4;
opts.MiniBatchSize = miniBatchSize;
opts.L2Regularization = 1e-3;
opts.Shuffle = "every-epoch";
opts.Plots = "training-progress";
opts.ValidationFrequency = max(1, floor(numel(trainIdx)/miniBatchSize));
opts.GradientThreshold = 1e-1;
opts.Metrics = 'fscore';

lrScheduler = customExpDecayLearnRate();

%--- Train
inputDim = size(X,3);
netBin = trainBinaryClassifier(XTrain, YTrain_bin, XVal, YVal_bin, inputDim, opts, lrScheduler);

fprintf("Arrhythmia binary classification done.\n");

%% --- Save model (uses your existing helper)
idx = saveWithComplexIndex('Model_netBin_', netBin);
