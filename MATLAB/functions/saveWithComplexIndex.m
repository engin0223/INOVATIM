function idx = saveWithComplexIndex(baseName, net, folder)
% saveWithComplexIndex - Saves a variable with an incremented complex index
%
% Syntax:
%   idx = saveWithComplexIndex(baseName, saveVar)
%   idx = saveWithComplexIndex(baseName, saveVar, folder)
%
% Inputs:
%   baseName - base filename as string, e.g., 'Model_Arr_'
%   saveVar  - variable to save (can be any MATLAB variable)
%   folder   - (optional) folder to save the file (default: specified folder)
%
% Output:
%   idx      - complex index used for the saved file

if nargin < 3
    folder = 'C:\Users\hp\Documents\MATLAB\INOVATIM_FINAL\functions\models'; % default folder
end

ext = '.mat';

% Get list of existing files
files = dir(fullfile(folder, [baseName, '*.mat']));

% Determine new index
if isempty(files)
    idx = 0 + 1i; 
else
    idxList = [];
    for k = 1:numel(files)
        name = files(k).name;
        expr = regexp(name, [baseName '(.*?)' ext], 'tokens', 'once');
        if ~isempty(expr)
            try
                idxList(end+1) = str2double(extractBetween(expr{1},2,strfind(expr{1},'.')-1)) + ...
                                 1i*str2double(extractAfter(expr{1},'.')); % convert '3+4i' → complex
            catch
                % skip malformed names
            end
        end
    end
    
    if isempty(idxList)
        idx = 0 + 1i;
    else
        lastIdx = idxList(end);
        % Increment rule: increase real part until 10, then increase imaginary
        if imag(lastIdx) < 10
            idx = 0 + (imag(lastIdx)+1)*1i;
        else
            idx = lastIdx + 1;
        end
    end
end

% Construct filename
fileName = fullfile(folder, sprintf('%sv%d.%d%s', baseName, real(idx), imag(idx), ext));

% Save variable and index
save(fileName, 'net', 'idx');

fprintf('Saved as %s with index %s\n', fileName, num2str(idx));

end
