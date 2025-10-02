function results = analyzeEvents(D, netArr, netBin, varargin)
%ANALYZEEVENTS Perform event inference and statistics using arrhythmia + binary networks
%
% Inputs:
%   D       - Input data [N x seqLen x features]
%   netArr  - Trained arrhythmia classification network
%   netBin  - Trained binary classification network
%
% Output (struct):
%   results.E_percentEvents   - Expected proportion per class
%   results.Std_percentEvents - Standard deviation of proportions
%   results.CI_lower          - Lower bound of CI
%   results.CI_upper          - Upper bound of CI
%   results.scores_all        - All predicted class scores
%   results.numClasses        - Number of classes
%   results.classNames        - Class names
%   results.diagnosis         - Struct with detected arrhythmia conditions

    % Parameters
    [numSamples, seqLen, inputDim] = size(D);
    if ~isempty(varargin)
        batchSize = varargin{1};
        if length(varargin) > 1
            c_perc = varargin{2};
        else
            c_perc = 0.95;
        end
    else
        batchSize = 1024;  % adjust based on GPU memory
        c_perc = 0.95;
    end
    numBatches = ceil(numSamples / batchSize);

    % Z-score normalization
    mu    = mean(reshape(D, [], inputDim), 1);
    sigma = std(reshape(D, [], inputDim), 0, 1);
    
    for f = 1:inputDim
        D(:,:,f) = (D(:,:,f) - mu(f)) ./ sigma(f);
    end

    % Class info
    classNamesArr = {...
    '!', '"', '+', '/', 'A', 'E', 'F', 'J', 'L', 'N', ...
    'Q', 'R', 'S', 'V', '[', ']', 'a', 'e', 'f', 'j', ...
    'x', '|', '~' ...
    };


    numClasses = numel(classNamesArr);

    % Init accumulators
    E_numEvents = 0;
    scores_all = [];

    % Batch loop
    for b = 1:numBatches
        batchIdx = (b-1)*batchSize + 1 : min(b*batchSize, numSamples);
        dataBatch = squeeze(D(batchIdx,:,:));  % [batch x seqLen x feat]
        dataBatch = gpuArray(permute(dataBatch, [2 3 1])); % [seqLen x feat x batch]

        % Predict
        scoresBin = gather(predict(netBin, dataBatch));
        scoresArr = gather(predict(netArr, dataBatch));

        % Combine scores
        scoresBatch = [scoresArr(:, 1:9) scoresBin(:, 1) scoresArr(:, 10:end)];
        scores_all = [scores_all; scoresBatch];

        % Accumulate expected counts
        E_numEvents = E_numEvents + sum(scoresBatch, 1);

        fprintf("Processed: Batch %d/%d\n", b, numBatches)
    end

    % Variance & stats
    N = size(scores_all, 1);
    Var_numEvents = (1/(N-1)) * sum((scores_all - E_numEvents).^2, 1);
    Std_percentEvents = sqrt(Var_numEvents / N);
    E_percentEvents = E_numEvents / N;

    % Confidence interval
    c = -erfinv(-c_perc) ./ Std_percentEvents;
    CI_lower = (E_numEvents - c) / N;
    CI_upper = (E_numEvents + c) / N;

    % Convert scores → predicted labels
    [~, predIdx] = max(scores_all, [], 2);

    % ----- Clinical rule-based detection -----
    diagnosis = struct();

    % PVC burden (% V beats)
    V_idx = find(strcmp(classNamesArr, "V"));
    if ~isempty(V_idx)
        PVC_percent = mean(predIdx == V_idx);
        diagnosis.FrequentPVCs = PVC_percent > 0.10; % >10% of beats
        diagnosis.PVC_percent = PVC_percent;
    end

    % Left / Right Bundle Branch Block
    L_idx = find(strcmp(classNamesArr, "L"));
    R_idx = find(strcmp(classNamesArr, "R"));
    diagnosis.LBBB = any(predIdx == L_idx);
    diagnosis.RBBB = any(predIdx == R_idx);

    % Paced rhythm
    P_idx = find(strcmp(classNamesArr, "/"));
    if ~isempty(P_idx)
        Paced_percent = mean(predIdx == P_idx);
        diagnosis.PacedRhythm = Paced_percent > 0.80; % >80% paced beats
        diagnosis.Paced_percent = Paced_percent;
    end

    % Tachycardia/Bradycardia using HR if available
    if size(D,3) > 1 % assuming HR feature is present
        HR = squeeze(D(:,1,end)); % replace with actual HR column
        diagnosis.SinusTachy = mean(HR > 100) > 0.05;  % >5% of time HR>100
        diagnosis.SinusBrady = mean(HR < 50) > 0.05;   % >5% of time HR<50
    end

    % VF/VT episodes (look for [ ... ] segments)
    VF_start = find(strcmp(classNamesArr, "["));
    VF_end   = find(strcmp(classNamesArr, "]"));
    if ~isempty(VF_start) && ~isempty(VF_end)
        diagnosis.VF_detected = any(predIdx == VF_start | predIdx == VF_end);
    else
        diagnosis.VF_detected = false;
    end

    % Convert scores → predicted labels
    [~, predIdx] = max(scores_all, [], 2);

    % Map indices to class names
    annotations = classNamesArr(predIdx);

    % Collect results
    results.E_percentEvents   = E_percentEvents;
    results.Std_percentEvents = Std_percentEvents;
    results.CI_lower          = CI_lower;
    results.CI_upper          = CI_upper;
    results.scores_all        = scores_all;
    results.numClasses        = numClasses;
    results.classNames        = classNamesArr;
    results.annotations       = annotations;   % NEW: predicted labels
    results.diagnosis         = diagnosis;
end
