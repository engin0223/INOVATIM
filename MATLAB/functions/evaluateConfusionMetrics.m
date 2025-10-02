function [metricsTable, accuracy] = evaluateConfusionMetrics(trueLabels, predLabels)
% evaluateConfusionMetrics  Compute precision/recall/f1 per class and overall accuracy
%
%   [metricsTable, accuracy] = evaluateConfusionMetrics(trueLabels, predLabels)
%   trueLabels/predLabels should be cellstr or string arrays of same length.
    
    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));
    
    trueLabels = cellstr(trueLabels);
    predLabels = cellstr(predLabels);

    [confMat, order] = confusionmat(trueLabels, predLabels);
    numClasses = numel(order);

    precision = zeros(numClasses,1); recall = zeros(numClasses,1); f1 = zeros(numClasses,1);
    for i = 1:numClasses
        TP = confMat(i,i);
        FP = sum(confMat(i,:)) - TP;
        FN = sum(confMat(:,i)) - TP;
        precision(i) = TP / (TP + FP + eps);
        recall(i)    = TP / (TP + FN + eps);
        f1(i)        = 2 * (precision(i)*recall(i)) / (precision(i)+recall(i)+eps);
    end

    classCounts = sum(confMat,2);

    metricsTable = table(order, classCounts, precision, recall, f1, ...
        'VariableNames', {'Class','Count','Precision','Recall','F1'});

    % Sort table by classCounts descending
    metricsTable = sortrows(metricsTable, 'Count', 'descend');

    accuracy = sum(strcmp(trueLabels, predLabels)) / numel(trueLabels);
end
