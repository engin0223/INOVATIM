function [predLabels, binScores, arrScores] = hierarchicalPredict(X, Y, netBin, netArr)
% hierarchicalPredict - Perform hierarchical ECG beat classification
%
%   Inputs:
%       X      - [N x SeqLen x Features] dataset (normalized)
%       netBin - trained binary classifier (Normal vs Arrhythmia)
%       netArr - trained arrhythmia classifier (arrhythmia subtypes)
%
%   Outputs:
%       predLabels - string array [N x 1] predicted labels (N or arr subtype)
%       binScores  - [N x 2] scores from binary network
%       arrScores  - [numArr x numArrClasses] scores from arrhythmia network

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));

    N_all = size(X,1);

    % --- Binary predictions (batched)
    binScores = predictBatched(netBin, X, 1024); % [N x 2]
    [~, binPredIdx] = max(binScores, [], 2);
    classNamesBin = ["N","Arr"];
    binPred = classNamesBin(binPredIdx);
    isArrhythmia = binPred == "Arr";

    predLabels = strings(N_all,1);
    predLabels(~isArrhythmia) = "N";

    % --- Arrhythmia subtype predictions
    arrScores = [];
    arrIndices = find(isArrhythmia);
    if ~isempty(arrIndices)
        % Subset of dataset
        Xsub = X(arrIndices,:,:);
        arrScores = predictBatched(netArr, Xsub, 1024); % [numArr x numArrClasses]

        % Get class names from arrNet
        arrClasses = ['!'	'"'	'+'	'/'	'A'	'E'	'F'	'J'	'L'	'Q'	'R'	'S'	'V'	'['	']'	'a'	'e'	'f'	'j'	'x'	'|'	'~']';
        
        [~, idx] = max(arrScores, [], 2);
        predLabels(arrIndices) = string(arrClasses(idx))';
    end
end
