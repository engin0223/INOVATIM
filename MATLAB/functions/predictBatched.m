function scores = predictBatched(net, X, batchSize)
% predictBatched  Predict all samples in X in batches using network net
%
%   scores = predictBatched(net, X, batchSize)
%   X is [N x SeqLen x Features]
%   scores is [N x numClasses]
%
    if nargin < 3, batchSize = 1024; end
    N = size(X,1);
    % get number of classes using predict on a small batch
    sampleBatch = prepareSequenceBatch(X, 1:min(4,N));
    sampleBatch = gpuArray(single(sampleBatch));
    tmp = predict(net, sampleBatch);
    numClasses = size(tmp,2);

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));

    scores = zeros(N, numClasses, 'single');

    for i = 1:batchSize:N
        idx = i:min(i+batchSize-1, N);
        Xbatch = prepareSequenceBatch(X, idx); % [time x features x b]
        Xbatch = gpuArray(single(Xbatch));
        out = predict(net, Xbatch); % [numClasses x b]
        out = gather(out);
        scores(idx,:) = out;
    end
end
