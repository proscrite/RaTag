clc; clear; close all;

% Load the data
MainFolder = 'G:\AXO_DATA\miniLOTEL\';
RUNNumber = 'RUN5';
PNumber = 'P1';
data = load([MainFolder, RUNNumber, '\', RUNNumber ,'_P1_MeanWFM.mat']);
% data = load('/home/yair/Downloads/MeanWFM.mat');
waveform = data.FinalMeanWFMs;
dt = 0.2;  % time step
sample = 0:dt:(length(waveform)-1)*dt;

log_waveform = log(-1*waveform);
log_waveform(log_waveform <= 0) = NaN;

valid_idx = ~isnan(log_waveform);
y = log_waveform(valid_idx);
x = sample(valid_idx);

cftool(x, y);