function netArr = trainArrhythmiaClassifier(XTrain, YTrain, XVal, YVal, ...
    inputDim, numClasses, classWeights, options, lrScheduler)
% trainArrhythmiaClassifier  Train multi-class arrhythmia classifier with optional weighted loss
%
%   netArr = trainArrhythmiaClassifier(...)

    arguments
        XTrain
        YTrain
        XVal
        YVal
        inputDim (1,1) double {mustBePositive}
        numClasses (1,1) double {mustBePositive}
        classWeights double = []
        options = struct()
        lrScheduler = []
    end

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));

    if isempty(fieldnames(options))
        options.MaxEpochs = 200;
        options.InitialLearnRate = 1e-5;
        options.MiniBatchSize = 64;
        options.L2Regularization = 1e-4;
        options.Plots = "training-progress";
        options.Shuffle = "every-epoch";
        options.Metrics = ["fscore", "accuracy"];
    end

    latentDim = 32; hiddenChannels = 128; dropoutRate = 0.6;
    layers = buildArrhythmiaClassifier(inputDim, numClasses, latentDim, hiddenChannels, dropoutRate);

    % Check if scheduler is provided
    if isempty(lrScheduler)
        topts = trainingOptions("adam", ...
            MaxEpochs=options.MaxEpochs, ...
            InitialLearnRate=options.InitialLearnRate, ...
            MiniBatchSize=options.MiniBatchSize, ...
            L2Regularization=options.L2Regularization, ...
            Shuffle=options.Shuffle, ...
            Plots=options.Plots, ...
            Metrics=options.Metrics, ...
            ValidationData={XVal,YVal}, ...
            ValidationFrequency=options.ValidationFrequency, ...
            GradientThreshold=options.GradientThreshold);
    else
        topts = trainingOptions("adam", ...
            MaxEpochs=options.MaxEpochs, ...
            InitialLearnRate=options.InitialLearnRate, ...
            MiniBatchSize=options.MiniBatchSize, ...
            L2Regularization=options.L2Regularization, ...
            Shuffle=options.Shuffle, ...
            Plots=options.Plots, ...
            Metrics=options.Metrics, ...
            ValidationData={XVal,YVal}, ...
            ValidationFrequency=options.ValidationFrequency, ...
            GradientThreshold=options.GradientThreshold, ...
            LearnRateSchedule="piecewise", ...          % or "none" if using custom
            LearnRateDropFactor=lrScheduler.DropFactor, ... % from your scheduler struct
            LearnRateDropPeriod=lrScheduler.Period);
    end


    if isempty(classWeights)
        netArr = trainnet(XTrain, YTrain, layers, "crossentropy", topts);
    else
        % trainnet supports custom loss handle in your original code; supply wrapper
        lossHandle = @(YPred,YTrue) weightedCrossEntropy(YPred, YTrue, classWeights);
        netArr = trainnet(XTrain, YTrain, layers, lossHandle, topts);
    end
end
