function layers = buildArrhythmiaClassifier(inputDim, numClasses, latentDim, hiddenChannels, dropoutRate)
% buildArrhythmiaClassifier  Build CNN layers for arrhythmia multi-class classifier
%
%   layers = buildArrhythmiaClassifier(inputDim, latentDim, hiddenChannels, dropoutRate, numClasses)
    
    arguments
        inputDim (1,1) double {mustBePositive}
        numClasses (1,1) double {mustBePositive}
        latentDim (1,1) double = 32
        hiddenChannels (1,1) double = 128
        dropoutRate (1,1) double = 0.25
    end

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));
    

    layers = [
        sequenceInputLayer(inputDim,"Name","seq_in")
        convolution1dLayer(9, hiddenChannels,'Padding','same',"Name","conv1")
        batchNormalizationLayer("Name","bn1")
        reluLayer("Name","relu1")
        dropoutLayer(dropoutRate,"Name","drop1")

        convolution1dLayer(7, hiddenChannels,'Padding','same','DilationFactor',2,"Name","conv2")
        batchNormalizationLayer("Name","bn2")
        reluLayer("Name","relu2")
        dropoutLayer(dropoutRate,"Name","drop2")

        convolution1dLayer(5, hiddenChannels,'Padding','same','DilationFactor',2,"Name","conv3")
        batchNormalizationLayer("Name","bn3")
        reluLayer("Name","relu3")
        dropoutLayer(dropoutRate,"Name","drop3")

        globalAveragePooling1dLayer("Name","gap")

        fullyConnectedLayer(hiddenChannels,"Name","fc1")
        batchNormalizationLayer("Name","bn4")
        reluLayer("Name","relu4")
        dropoutLayer(dropoutRate,"Name","drop4")

        fullyConnectedLayer(latentDim,"Name","fc_latent")
        reluLayer("Name","relu_latent")
        dropoutLayer(dropoutRate,"Name","drop_latent")

        fullyConnectedLayer(numClasses,"Name","fc_out")
        softmaxLayer("Name","softmax")
    ];
end
