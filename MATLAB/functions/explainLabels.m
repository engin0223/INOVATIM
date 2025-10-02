function Y_explained = explainLabels(Y)
% explainLabels  Map single-character labels to human-readable descriptions
%
%   Y_explained = explainLabels(Y)
%   Y can be cellstr, char array, categorical; returns cell array of "sym - text"

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));

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

    Ycell = cellstr(Y);
    Y_explained = cell(size(Ycell));
    for k = 1:numel(Ycell)
        sym = Ycell{k};
        if isKey(symbolMap, sym)
            Y_explained{k} = [sym ' - ' symbolMap(sym)];
        else
            Y_explained{k} = [sym ' - Unknown'];
        end
    end
end
