%% Step 1: Binary Classification (Using trainNet)
clear; clc;

% Load dataset
load("mitbih_beat_dataset.mat")  % X: [N x SeqLen x Features], Y: char labels
[N, seqLen, inputDim] = size(X);

% Z-score normalization per feature
mu = squeeze(mean(mean(X,1,'omitnan'),2,'omitnan'));
sigma = squeeze(std(reshape(X,[],inputDim),0,1,'omitnan'))';
sigma(sigma==0) = 1;

for f = 1:inputDim
    X(:,:,f) = (X(:,:,f) - mu(f)) ./ sigma(f);
end

% Convert to cell arrays
X_cell = cell(N,1);
for i = 1:N
    X_cell{i} = squeeze(X(i,:,:));
end

% Train/Validation split
rng(42);
idx = randperm(N);
numVal = max(round(0.2*N),1);
valIdx = idx(1:numVal);
trainIdx = idx(numVal+1:end);

XTrain = X_cell(trainIdx);
XVal   = X_cell(valIdx);

% Binary labels
Y_binary = double(Y ~= 'N');  % 0 = Normal, 1 = Arrhythmia
YTrain_bin = categorical(Y_binary(trainIdx));
YVal_bin   = categorical(Y_binary(valIdx));

% --------------------------
% Binary classifier layers
% --------------------------
hiddenChannels = 64; dropoutRate = 0.4;
layersBin = [
    sequenceInputLayer(inputDim)
    convolution1dLayer(7, hiddenChannels,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(dropoutRate)
    
    convolution1dLayer(5, hiddenChannels,'Padding','same','DilationFactor',2)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(dropoutRate)
    
    globalAveragePooling1dLayer
    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(dropoutRate)
    fullyConnectedLayer(2)
    softmaxLayer
];

% Convert to dlnetwork
dlnetBin = dlnetwork(layerGraph(layersBin));

% Training options
optionsBin = trainingOptions("adam", ...
    MaxEpochs=200, ...
    MiniBatchSize=256, ...
    InitialLearnRate=1e-3, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal,YVal_bin}, ...
    Plots="training-progress");

% Train network using trainNet
%netBin = trainnet(dlnetBin, XTrain, YTrain_bin, optionsBin);

fprintf("Binary classification done.\n");

%% Step 2: Arrhythmia-only Network
arrIdx = find(Y ~= 'N');         % indices of arrhythmic beats

load("mitbih_beat_dataset.mat")

[N, seqLen, inputDim] = size(X);

% Z-score normalization per feature
mu = squeeze(mean(mean(X,1,'omitnan'),2,'omitnan'));
sigma = squeeze(std(reshape(X,[],inputDim),0,1,'omitnan'))';
sigma(sigma==0) = 1;

for f = 1:inputDim
    X(:,:,f) = (X(:,:,f) - mu(f)) ./ sigma(f);
end

% Convert to cell arrays
X_cell = cell(N,1);
for i = 1:N
    X_cell{i} = squeeze(X(i,:,:))';
end

XArr = X_cell(arrIdx);            % arrhythmia samples
YArr = Y(arrIdx);                 % arrhythmia labels

% Convert to categorical for training
YArrCat = categorical(cellstr(YArr));

% Split into train/val
NArr = numel(XArr);
rng(123);
idxArr = randperm(NArr);
numValArr = max(round(0.2*NArr),1);
valArrIdx = idxArr(1:numValArr);
trainArrIdx = idxArr(numValArr+1:end);

XTrainArr = XArr(trainArrIdx);
XValArr   = XArr(valArrIdx);

YTrainArr = YArrCat(trainArrIdx);
YValArr   = YArrCat(valArrIdx);

% --------------------------
% Arrhythmia classifier layers
% --------------------------
latentDim = 32; hiddenChannels = 128; dropoutRate = 0.4;
layersArr = [
    sequenceInputLayer(inputDim)
    convolution1dLayer(9, hiddenChannels,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(dropoutRate)
    
    convolution1dLayer(7, hiddenChannels,'Padding','same','DilationFactor',2)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(dropoutRate)

    convolution1dLayer(5, hiddenChannels,'Padding','same','DilationFactor',2)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(dropoutRate)
    
    globalAveragePooling1dLayer
    fullyConnectedLayer(latentDim)
    reluLayer
    dropoutLayer(dropoutRate)
    fullyConnectedLayer(numel(unique(YArr)))  % number of arrhythmia classes
    softmaxLayer
];

dlnetArr = dlnetwork(layerGraph(layersArr));

scheduler = piecewiseLearnRate("DropFactor", 0.1^(0.05), "FrequencyUnit", "epoch", "Period", 10);

lrSchedule = customExpDecayLearnRate();


optionsArr = trainingOptions("adam", ...
    MaxEpochs=1000, ...
    InitialLearnRate=1e-12, ...
    Shuffle="every-epoch", ...
    LearnRateSchedule=lrSchedule, ...
    Metrics=["fscore", "accuracy"], ...
    MiniBatchSize=32, ...
    ValidationData={XValArr,YValArr}, ...
    Plots="training-progress");

% Train arrhythmia classifier
%netArr = trainnet(XTrainArr, YTrainArr, dlnetArr, "crossentropy", optionsArr);

fprintf("Arrhythmia subtype classification done.\n");

%% Save the Networks

save("Model_v0.2.mat", "netArr", "netBin");

save("Model_Arr_v0.1.mat", "netArr");

%% Hierarchical Prediction (Mixed Networks)
clear; clc;

load("mitbih_beat_dataset.mat")

load("Model_Binary_v0.1.mat")
load("Model_Arr_v0.1.mat")

classNamesBin = ["N","Arr"];                              
classNamesArr = categories(categorical(cellstr(Y)));    
classNamesArr = classNamesArr(classNamesArr ~= "N");     

N_all = size(X,1);
seqLen = size(X,2);
inputDim = size(X,3);

% Binary classifier predictions
X_allCell = squeeze(num2cell(X,[2 3]));   % SeriesNetwork still needs cell array
binScores = predict(netBin, X_allCell);              
[~, binPredIdx] = max(binScores, [], 2);
binPred = classNamesBin(binPredIdx);
isArrhythmia = binPred == "Arr";

predLabels = strings(N_all,1);
predLabels(~isArrhythmia) = "N";

clearvars X_allCell

% Arrhythmia subtype predictions using GPU full array
arrIndices = find(isArrhythmia);
if ~isempty(arrIndices)
    % Prepare data as [features x time x batch] and convert to single + GPU
    X_arr = permute(X(arrIndices,:,:), [2 3 1]);  % [features x time x batch]
    X_arr = gpuArray(single(X_arr));
    
    % Batch prediction
    scoresArr = predict(netArr, X_arr);         % [numClasses x batch]
    scoresArr = gather(scoresArr);
    [~, idx] = max(scoresArr, [], 2);
    predLabels(arrIndices) = classNamesArr(idx)';
end

trueLabels = string(cellstr(Y));

[confMat, order] = confusionmat(trueLabels, predLabels);
confChart = confusionchart(trueLabels, predLabels);
confChart.Title = 'Confusion Matrix';

numClasses = numel(order);
precision = zeros(numClasses,1); recall = zeros(numClasses,1); f1 = zeros(numClasses,1);
for i = 1:numClasses
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;
    precision(i) = TP / (TP + FP + eps);
    recall(i)    = TP / (TP + FN + eps);
    f1(i)        = 2 * (precision(i)*recall(i)) / (precision(i)+recall(i)+eps);
end

accuracy = sum(predLabels==trueLabels)/N_all;


metricsTable = table(order, precision, recall, f1, 'VariableNames', {'Class','Precision','Recall','F1'});
fprintf("Overall Accuracy: %.2f%%\n", accuracy*100);
disp(metricsTable);

% Arrhythmia-only metrics
arrClasses = order(order ~= "N");          % arrhythmia class labels
arrIndicesAll = find(order ~= "N");        % indices of arrhythmia classes in confMat
confMatArr = confMat(arrIndicesAll, arrIndicesAll);
numArrClasses = numel(arrClasses);

% Initialize arrays
precisionArr = zeros(numArrClasses,1); 
recallArr    = zeros(numArrClasses,1); 
f1Arr        = zeros(numArrClasses,1);

% Compute metrics per arrhythmia class
for i = 1:numArrClasses
    TP = confMatArr(i,i);
    FP = sum(confMatArr(:,i)) - TP;
    FN = sum(confMatArr(i,:)) - TP;
    precisionArr(i) = TP / (TP + FP + eps);
    recallArr(i)    = TP / (TP + FN + eps);
    f1Arr(i)        = 2 * (precisionArr(i)*recallArr(i)) / (precisionArr(i)+recallArr(i)+eps);
end

% Compute number of samples per arrhythmia class (exclude row k if needed)
k = 2;  % example row to exclude (optional)
rowsToCount = setdiff(1:size(confMat,1), k);
arrN = sum(confMat(rowsToCount, arrIndicesAll), 1)';  % column vector

% Sort by number of samples
[~, sortIdx] = sort(arrN, 'descend');

% Create table including NumSamples
metricsTableArr = table(...
    arrClasses(sortIdx), ...
    precisionArr(sortIdx), ...
    recallArr(sortIdx), ...
    f1Arr(sortIdx), ...
    arrN(sortIdx), ...
    'VariableNames', {'Class','Precision','Recall','F1','NumSamples'});

% Compute overall accuracy for arrhythmia-only samples
accuracyArrOnly = sum(predLabels(arrIndices) == trueLabels(arrIndices)) / length(arrIndices);
fprintf("Overall Accuracy (Arrhythmia only): %.2f%%\n", accuracyArrOnly*100);

% Display table in a GUI window
f = figure('Name','Arrhythmia Metrics','NumberTitle','off','Position',[100 100 600 400]);
uitable(f, 'Data', metricsTableArr{:,:}, ...
           'ColumnName', metricsTableArr.Properties.VariableNames, ...
           'Units','Normalized','Position',[0 0 1 1]);
