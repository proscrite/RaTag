clc; clear all; close all;

HD11 = 'F:\AXO_Data';
HD12 = 'E:\AXO_DATA\';
RUNNumber = 'RUN69';
PNumber = 'P2';

mainSaveFolder = 'C:\Users\Admin\Desktop\Matlab_variables\meanWFMs\';

files = dir([HD12, '\', RUNNumber, '\', PNumber, '\*.wfm']);
[~, index] = sort({files.date});
files = files(index);
meanWFM = [];

for k = 1:500
    disp(['Processing file: ', num2str(k), ' of: ', num2str(length(files))]);
    [sig_PMT1, t_PMT1, ~, ~, ~] = wfm2readALLframesScaled([files(k).folder, '\', files(k).name], 1);
    meanWFM = [meanWFM; mean(sig_PMT1, 1)];
end

meanWFMs = mean(meanWFM, 1);
save([files(k).folder, '\Analysis\meanWFMs', RUNNumber, PNumber, '.mat'], "meanWFMs");
save([mainSaveFolder, 'meanWFMs', RUNNumber, PNumber, '.mat'], "meanWFMs");