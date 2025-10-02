function [trainIdx, valIdx] = stratifiedSplit(Ylabels, valFraction, seed)
% stratifiedSplit  Stratified split indices for labels
%
%   [trainIdx, valIdx] = stratifiedSplit(Ylabels, valFraction, seed)
%   Ylabels: categorical or convertible to categorical (Nx1)
%   valFraction: e.g. 0.25 for 75/25 train/val
%   seed: rng seed for reproducibility

    if nargin < 3, seed = 0; end
    if nargin < 2, valFraction = 0.25; end

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));

    Ycat = categorical(Ylabels);
    rng(seed);
    classes = categories(Ycat);

    trainIdx = [];
    valIdx = [];

    for c = 1:numel(classes)
        idx = find(Ycat == classes{c});
        idx = idx(randperm(numel(idx)));
        nVal = round(valFraction * numel(idx));
        nTrain = numel(idx) - nVal;

        if nTrain > 0
            trainIdx = [trainIdx; idx(1:nTrain)];
        end
        if nVal > 0
            valIdx = [valIdx; idx(nTrain+1:end)];
        end
    end

    % Shuffle final indices
    trainIdx = trainIdx(randperm(numel(trainIdx)));
    valIdx = valIdx(randperm(numel(valIdx)));
end
