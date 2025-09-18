%% Step 1: Binary Classification
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

% Binary classifier layers
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
    classificationLayer
];

optionsBin = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'MiniBatchSize',256, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal,YVal_bin}, ...
    'Plots','training-progress');

% Train binary classifier
netBin = trainNetwork(XTrain,YTrain_bin,layersBin,optionsBin);

fprintf("Binary classification done.\n");

% Step 2: Arrhythmia-only Network
arrIdx = find(Y ~= 'N');         % indices of arrhythmic beats
XArr = X_cell(arrIdx);            % arrhythmia samples
YArr = Y(arrIdx);                 % arrhythmia labels (multi-class)

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

% Arrhythmia classifier / embedding network layers
latentDim = 16;
layersArr = [
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
    fullyConnectedLayer(latentDim)
    reluLayer
    dropoutLayer(dropoutRate)
    fullyConnectedLayer(numel(unique(YArr)))  % number of arrhythmia classes
    softmaxLayer
    classificationLayer
];

optionsArr = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'MiniBatchSize',128, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValArr,YValArr}, ...
    'Plots','training-progress');

% Train arrhythmia classifier (completely separate network)
netArr = trainNetwork(XTrainArr,YTrainArr,layersArr,optionsArr);

fprintf("Arrhythmia subtype classification done.\n");


%% Hierarchical Prediction for All Beats
% X_all : {N x 1} cell array, each cell [Features x SeqLen]
% Y_all : true labels (char), length N
% netBin : binary classifier (Normal vs Arrhythmia)
% netArr : arrhythmia classifier (subtypes)


% Suppose original data X_all_mat is [N x SeqLen x Features]
N = size(X,1);
X_all = cell(N,1);

for i = 1:N
    X_all{i} = squeeze(X(i,:,:));  % [Features x SeqLen]
end

N_all = size(X, 1);
predLabels = strings(N_all,1);   % store predictions

% --------------------------
% Binary classifier predictions
% --------------------------
binPred = classify(netBin, X_all); % categorical 0/1
isArrhythmia = binPred == categorical(1);

% --------------------------
% Predict arrhythmia subtypes only for arrhythmic beats
% --------------------------
arrIndices = find(isArrhythmia);
if ~isempty(arrIndices)
    XArr = X_all(arrIndices);  % arrhythmic beats
    arrPred = classify(netArr, XArr); % categorical labels
    predLabels(arrIndices) = string(cellstr(arrPred));
end

% Normal beats
normalIndices = find(~isArrhythmia);
predLabels(normalIndices) = "N";

% Convert true labels to string for comparison
trueLabels = string(Y);

% --------------------------
% Confusion matrix
% --------------------------
[confMat, order] = confusionmat(trueLabels, predLabels);
confChart = confusionchart(trueLabels, predLabels);
confChart.Title = 'Confusion Matrix';

% --------------------------
% Metrics: Accuracy, Precision, Recall, F1-score
% --------------------------
numClasses = numel(order);
precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1 = zeros(numClasses,1);

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


% --------------------------
% Metrics excluding 'N' class
% --------------------------
arrClasses = order(order ~= "N");                   % arrhythmic classes only
confMatArr = confMat(order ~= "N", order ~= "N");   % submatrix excluding 'N'

numArrClasses = numel(arrClasses);
precisionArr = zeros(numArrClasses,1);
recallArr = zeros(numArrClasses,1);
f1Arr = zeros(numArrClasses,1);

for i = 1:numArrClasses
    TP = confMatArr(i,i);
    FP = sum(confMatArr(:,i)) - TP;
    FN = sum(confMatArr(i,:)) - TP;

    precisionArr(i) = TP / (TP + FP + eps);
    recallArr(i)    = TP / (TP + FN + eps);
    f1Arr(i)        = 2 * (precisionArr(i)*recallArr(i)) / (precisionArr(i)+recallArr(i)+eps);
end

metricsTableArr = table(arrClasses, precisionArr, recallArr, f1Arr, ...
                        'VariableNames', {'Class','Precision','Recall','F1'});

accuracyArrOnly = sum(predLabels(arrIndices) == trueLabels(arrIndices)) / length(arrIndices);

fprintf("Overall Accuracy (Arrhythmia only): %.2f%%\n", accuracyArrOnly*100);
disp(metricsTableArr);

