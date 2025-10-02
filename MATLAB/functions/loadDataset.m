function [X, Y] = loadDataset(filename)
% loadDataset  Load dataset from a .mat file containing variables X and Y
%
%   [X, Y] = loadDataset(filename)
%
%   X -> [N x SeqLen x Features]
%   Y -> labels (char array, cellstr, or categorical)
%
%   Example:
%       [X,Y] = loadDataset('mitbih_beat_dataset.mat');

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));
    fullFile = fullfile(BASEPATH, 'data', filename);

    if ~isfile(fullFile)
        error('loadDataset:FileNotFound', 'File "%s" not found.', filename);
    end

    data = load(fullFile);
    if isfield(data, 'X')
        X = data.X;
    else
        error('loadDataset:MissingX', 'File does not contain variable X.');
    end

    if isfield(data, 'Y')
        Y = data.Y;
    else
        error('loadDataset:MissingY', 'File does not contain variable Y.');
    end
end
