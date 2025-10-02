% runEventAnalysis.m
% Run analyzeEvents() (user-supplied), produce event stats & plots
%clear; clc;
BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL';
addpath(fullfile(BASEPATH, 'functions'));
addpath(fullfile(BASEPATH, 'functions', 'models'));
addpath(fullfile(BASEPATH, 'functions', 'data'));

%datasetFile = "beat_dataset_filtered.mat";
load("Model_netArr_v0.4.mat","net"); netArr = net;
load("Model_netBin_v0.3.mat","net"); netBin = net; clear net;

%[X, ~] = loadDataset(datasetFile);

% Normalize
[X, mu, sigma] = normalizeData(X);

% analyzeEvents is assumed to exist (user-provided)
results = analyzeEvents(X, netArr, netBin);

E_percentEvents = results.E_percentEvents;
Std_percentEvents = results.Std_percentEvents;
CI_lower = results.CI_lower;
CI_upper = results.CI_upper;
scores_all = results.scores_all;
numClasses = results.numClasses;
classNames = results.classNames;

% Build table
E_percentEventsTable = table(E_percentEvents', Std_percentEvents', CI_lower', CI_upper', ...
    'RowNames', classNames, ...
    'VariableNames', {'PercentEvents', 'Std', 'CI_Lower', 'CI_Upper'});

disp(E_percentEventsTable);

% Plot
plotEventStats(classNames, E_percentEvents, CI_lower, CI_upper);

%% Optionally save analyzed dataset

% Assuming 'results.annotations' contains the class labels corresponding to X
Y = results.annotations;  % Assign class labels

% Define the mapping as a containers.Map
symbolMap = containers.Map( ...
    {'!', '"', '+', '/', 'A', 'E', 'F', 'J', 'L', 'N', 'Q', 'R', 'S', 'V', '[', ']', 'a', 'e', 'f', 'j', 'x', '|', '~'}, ...
    {'Ventricular flutter wave', 'Unknown / not listed', 'Unknown / not listed', 'Paced beat', ...
     'Atrial premature beat', 'Ventricular escape beat', 'Fusion of ventricular and normal beat', ...
     'Nodal (junctional) premature beat', 'Left bundle branch block beat', 'Normal beat', ...
     'Unclassifiable beat', 'Right bundle branch block beat', 'Supraventricular premature beat', ...
     'Premature ventricular contraction', 'Start of ventricular flutter/fibrillation', ...
     'End of ventricular flutter/fibrillation', 'Aberrated atrial premature beat', ...
     'Atrial escape beat', 'Fusion of paced and normal beat', 'Nodal (junctional) escape beat', ...
     'Non-conducted P-wave (blocked APB)', 'Isolated QRS-like artifact', 'Unknown / not listed'});

% Create a new cell array with symbol + explanation
Y_explained = cell(size(Y));
for k = 1:length(Y)
    sym = Y{k};
    if isKey(symbolMap, sym)
        Y_explained{k} = [sym ' - ' symbolMap(sym)];
    else
        Y_explained{k} = [sym ' - Unknown'];
    end
end

% Save X and Y to a .mat file
save(fullfile(BASEPATH, 'functions/data/', 'ecg_smartwatch_labeled_dataset.mat'), 'X', 'Y_explained');
