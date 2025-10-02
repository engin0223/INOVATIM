% Load dataset
clear; clc;
BASEPATH = 'C:/Users/hp/Documents/MATLAB/INOVATIM_FINAL';
addpath(fullfile(BASEPATH, 'functions'));
addpath(fullfile(BASEPATH, 'functions', 'models'));
addpath(fullfile(BASEPATH, 'functions', 'data'));

load('ecg_smartwatch_labeled_dataset.mat'); % X and Y
numRows = size(X,1);

% Create figure
fig = figure('Name','Row Plotter','NumberTitle','off');

% Set units to normalized for dynamic resizing
ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.1 0.2 0.8 0.7]);

rowLabel = uicontrol('Style','text', 'Units','normalized', ...
    'Position',[0.02 0.02 0.12 0.05], 'String','Select Row:');

rowSelector = uicontrol('Style','edit', 'Units','normalized', ...
    'Position',[0.15 0.02 0.12 0.06], 'String','1');

plotBtn = uicontrol('Style','pushbutton','Units','normalized', ...
    'Position',[0.30 0.02 0.12 0.06], 'String','Plot', ...
    'Callback', @(src,event) plotRow(rowSelector, X, Y_explained, ax));

leftBtn = uicontrol('Style','pushbutton','Units','normalized', ...
    'Position',[0.45 0.02 0.06 0.06], 'String','<', ...
    'Callback', @(src,event) navigateRow(-1, rowSelector, X, Y_explained, ax));

rightBtn = uicontrol('Style','pushbutton','Units','normalized', ...
    'Position',[0.52 0.02 0.06 0.06], 'String','>', ...
    'Callback', @(src,event) navigateRow(1, rowSelector, X, Y_explained, ax));

% Plot function
function plotRow(rowSelector, X, Y, ax)
    r = str2double(get(rowSelector,'String'));
    if isnan(r) || r < 1 || r > size(X,1)
        errordlg('Invalid row number','Error'); return;
    end
    plot(ax, X(r,:));
    xlabel(ax,'Column Index');
    ylabel(ax,'Value');
    title(ax,['Row ' num2str(r) ' of X | Label: ' Y{r}]);
end

% Navigation function
function navigateRow(direction, rowSelector, X, Y, ax)
    r = str2double(get(rowSelector,'String')) + direction;
    r = max(1, min(size(X,1), r));
    set(rowSelector,'String',num2str(r));
    plotRow(rowSelector, X, Y, ax);
end
