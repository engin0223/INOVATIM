% runArrhythmiaClassification.m
% Driver script to train arrhythmia subtype classifier (arrhythmia-only or including N)

clear; clc;
BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL';
addpath(fullfile(BASEPATH, 'functions'));
addpath(fullfile(BASEPATH, 'functions', 'models'));
addpath(fullfile(BASEPATH, 'functions', 'data'));

datasetFile = "beat_dataset_filtered.mat";
rngSeed = 123;
miniBatchSize = 32*4;

%--- Load & normalize
[X, Y] = loadDataset(datasetFile);
[X, mu, sigma] = normalizeData(X);

%--- Create cell sequences (time x features) for arrhythmia-only
N = size(X,1);
X_cell = cell(N,1);
for i = 1:N
    X_cell{i} = squeeze(X(i,:,:))'; % [seqLen x features]
end

%--- Select arrhythmia indices (non-'N')
arrIdx = find(~(Y == 'N'));
XArr = X_cell(arrIdx);
YArr = Y(arrIdx);

YArrCat = categorical(cellstr(YArr));
[trainArrIdx, valArrIdx] = stratifiedSplit(YArrCat, 0.2, rngSeed);

XTrainArr = XArr(trainArrIdx);
XValArr   = XArr(valArrIdx);
YTrainArr = YArrCat(trainArrIdx);
YValArr   = YArrCat(valArrIdx);

%--- Class weighting (inverse frequency)
classCounts = countcats(categorical(cellstr(Y)));
classNames = categories(categorical(cellstr(Y)));
classWeights = sum(classCounts) ./ (numel(classNames) * classCounts);
classWeights = classWeights / sum(classWeights) * numel(classNames); % normalized

% Map weights to arrhythmia classes used in training
classesArr = categories(YArrCat);
weightsArr = zeros(numel(classesArr),1);
for i=1:numel(classesArr)
    idx = find(cell2mat(classNames) == classesArr{i});
    if isempty(idx)
        weightsArr(i) = 1;
    else
        weightsArr(i) = classWeights(idx);
    end
end

%--- Convert cell arrays XTrainArr/XValArr into arrays [time x features x batch]
% trainnet in your environment may accept cell sequences; we'll convert to array here
convertCellToArray = @(C) cat(3, C{:}); % yields [time x features x batch]
XTrainArr_array = single(convertCellToArray(XTrainArr));
XValArr_array   = single(convertCellToArray(XValArr));

% Move to GPU if available
XTrainArr_array = gpuArray(XTrainArr_array);
XValArr_array = gpuArray(XValArr_array);

%--- Training options
opts = struct();
opts.MaxEpochs = 1000;
opts.InitialLearnRate = 1e-3;
opts.MiniBatchSize = miniBatchSize;
opts.L2Regularization = 1e-3;
opts.Shuffle = "every-epoch";
opts.Plots = "training-progress";
opts.Metrics = ["fscore", "accuracy"];
opts.ValidationFrequency = max(1, floor(size(XTrainArr_array,3)/opts.MiniBatchSize));
opts.GradientThreshold = 1e-1;

lrScheduler = customExpDecayLearnRate();

%--- Train
inputDim = size(X,3);
numArrClasses = numel(classesArr);
netArr = trainArrhythmiaClassifier(XTrainArr_array, YTrainArr, XValArr_array, YValArr, ...
    inputDim, numArrClasses, weightsArr, opts, lrScheduler);

fprintf("Arrhythmia subtype classification done.\n");

%% --- Save model
idx = saveWithComplexIndex('Model_netArr_', netArr);
