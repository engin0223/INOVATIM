function [X_norm, mu, sigma] = normalizeData(X)
% normalizeData  Z-score normalize each feature channel of X
%
%   [X_norm, mu, sigma] = normalizeData(X)
%   X: [N x SeqLen x Features]
%   mu: [Features x 1]
%   sigma: [Features x 1] (zeros replaced by 1 to avoid div by zero)

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));

    [N, seqLen, inputDim] = size(X);
    mu    = squeeze(mean(mean(X,1,'omitnan'),2,'omitnan'));
    sigma = squeeze(std(reshape(X,[],inputDim),0,1,'omitnan'))';
    sigma(sigma == 0) = 1;

    X_norm = X;
    for f = 1:inputDim
        X_norm(:,:,f) = (X(:,:,f) - mu(f)) ./ sigma(f);
    end
end
