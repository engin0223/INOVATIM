function plotEventStats(classNames, E_percentEvents, CI_lower, CI_upper)
% plotEventStats  Bar + errorbar plot for event percentages
%
%   plotEventStats(classNames, E_percentEvents, CI_lower, CI_upper)
%
    if nargin < 4
        CI_upper = E_percentEvents;
        CI_lower = zeros(size(E_percentEvents));
    end

    BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL/functions';
    addpath(fullfile(BASEPATH, 'models'));
    addpath(fullfile(BASEPATH, 'data'));

    percentEvents = E_percentEvents(:);
    ciLower = CI_lower(:);
    ciUpper = CI_upper(:);

    errLower = percentEvents - ciLower;
    errUpper = ciUpper - percentEvents;

    x = 1:numel(percentEvents);

    figure;
    errorbar(x, percentEvents*100, errLower*100, errUpper*100, 'o', ...
        'MarkerSize', 6, 'MarkerFaceColor', 'b', 'LineWidth', 1.5);
    grid on;
    xlabel('Event Class');
    ylabel('Percent Events (%)');
    title('Percent Events with 95% Confidence Intervals');
    xticks(x);
    xticklabels(classNames);
    xtickangle(45);

    figure;
    bar(x, percentEvents*100);
    hold on;
    errorbar(x, percentEvents*100, errLower*100, errUpper*100, 'k.', 'LineWidth', 1.5);
    grid on;
    xlabel('Event Class');
    ylabel('Percent Events (%)');
    title('Percent Events with 95% Confidence Intervals');
    xticks(x);
    xticklabels(classNames);
    xtickangle(45);
    hold off;
end
