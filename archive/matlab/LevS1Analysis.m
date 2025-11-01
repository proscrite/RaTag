clc; close all; clear all;

MainFolder = '\\132.72.12.214\d\AXO_DATA\miniLOTEL\';
RUNnumber = 'RUN1';
subRUNnumber = 'P4';
commonSaveFolder = [MainFolder, RUNnumber, '\', subRUNnumber, '\Analysis'];
folder = [MainFolder, RUNnumber, '\', subRUNnumber,'\Analysis\result'];

A = dir(fullfile(folder, '*.mat'));
fileDates = datetime({A.date}, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
[~, index] = sort(fileDates);
A = A(index);

AllTables = [];

for i = 1:10:length(A)
    filePath = fullfile(folder, A(i).name);
    fprintf('Loading file: %s\n', filePath);
    
    S = load(filePath);
    if isfield(S, 'Area_s1_frame')
        AllTables = [AllTables; S.Area_s1_frame];
    else
        warning('No "results" field in %s', A(i).name);
    end
end

disp('Final combined table size:');
disp(size(AllTables));

save([commonSaveFolder,  '\DriftScan', RUNnumber, '_', subRUNnumber, '.mat'], 'AllTables');

%%

P1 = load('\\132.72.12.214\d\AXO_DATA\miniLOTEL\RUN1\P1\Analysis\DriftScanRUN1_P1.mat');
P2 = load('\\132.72.12.214\d\AXO_DATA\miniLOTEL\RUN1\P2\Analysis\DriftScanRUN1_P2.mat');
P3 = load('\\132.72.12.214\d\AXO_DATA\miniLOTEL\RUN1\P3\Analysis\DriftScanRUN1_P3.mat');
P4 = load('\\132.72.12.214\d\AXO_DATA\miniLOTEL\RUN1\P4\Analysis\DriftScanRUN1_P4.mat');

P = [P1.AllTables; P2.AllTables; P3.AllTables; P4.AllTables];

figure();
subplot(1,2,1)
plot(movmean(P, 500)); hold on;
xline([length(P1.AllTables), length(P1.AllTables)+length(P2.AllTables),...
length(P1.AllTables)+length(P2.AllTables)+length(P3.AllTables)]);
xlabel('Sample [#]');
ylabel('S1e [mVnS]');
grid();
title('S1');
subplot(1,2,2)
histogram(P4.AllTables, 500);
xlabel('mVnS');
ylabel('Entries');
title('S1e');

