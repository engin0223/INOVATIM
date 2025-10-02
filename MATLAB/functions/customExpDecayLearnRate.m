function scheduler = customExpDecayLearnRate()
% customExpDecayLearnRate  Return a simple piecewiseLearnRate suitable for trainingOptions
% This wraps the original line used in your code and returns a piecewiseLearnRate object.
%
%   scheduler = customExpDecayLearnRate()
%
%   The default factor/period are conservative; change inside if needed.
    
    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));
    scheduler = piecewiseLearnRate("DropFactor", 0.1^(0.1), "FrequencyUnit", "epoch", "Period", 10);
end
