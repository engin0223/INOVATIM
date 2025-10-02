function loss = weightedCrossEntropy(predictions, targets, classWeights)
% weightedCrossEntropy - Weighted cross-entropy loss for minibatch
% Predictions: [numClasses x batch] probabilities
% Targets: one-hot [numClasses x batch]
% classWeights: vector length numClasses

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));

    weights = classWeights(:);
    wTargets = targets .* weights;
    wTargets = wTargets ./ sum(wTargets,1);
    loss = -sum(wTargets .* log(predictions+eps), 1);
    loss = mean(loss);
end
