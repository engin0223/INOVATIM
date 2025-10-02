function Xbatch = prepareSequenceBatch(X, indices)
% prepareSequenceBatch  Prepare [features x time x batch] input for predict/train
%
%   Xbatch = prepareSequenceBatch(X, indices)
%   X is [N x SeqLen x Features]
%   Xbatch is [SeqLen x Features x batch] (MATLAB's sequenceInput expects time x features x batch)
%
%   Note: this function does NOT move to GPU. Caller should convert to gpuArray if desired.

    if isempty(indices)
        Xbatch = [];
        return;
    end

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));

    seqLen = size(X,2);
    inputDim = size(X,3);
    batchSize = numel(indices);

    Xbatch = zeros(seqLen, inputDim, batchSize, 'single');
    for b = 1:batchSize
        xi = squeeze(X(indices(b),:,:)); % [seqLen x features]
        if size(xi,2) ~= inputDim
            xi = xi'; % try transposition if shapes mismatch
        end
        Xbatch(:,:,b) = single(xi);
    end
end
