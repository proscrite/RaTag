clc; clear; close all;

MainFolder = 'E:\AXO_DATA\';

RUNnumber = 'RUN67';
subRUNnumber = 'P22';

folder = [MainFolder, RUNnumber, '\', subRUNnumber, '\Analysis\result5Taus'];
commonSaveFolder = [MainFolder, RUNnumber, '\', subRUNnumber, '\Analysis\'];
A = dir(fullfile(folder, '*.mat'));
fileDates = datetime({A.date}, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
[~, index] = sort(fileDates);
A = A(index);

AllTables = [];

for i = 1:length(A)
    filePath = fullfile(folder, A(i).name);
    fprintf('Loading file: %s\n', filePath);
    
    S = load(filePath);
    if isfield(S, 'results')
        T = S.results.Area_s1;
        AllTables = [AllTables; T];
    else
        warning('No "results" field in %s', A(i).name);
    end
end

disp('Final combined table size:');
disp(size(AllTables));

save([commonSaveFolder,  '\DriftScan', RUNnumber, '_', subRUNnumber, '_5Taus.mat'], 'AllTables');

%%
% figure()
% subplot(2,3,1)
% i = find((AllTables.Area_ped > -8) &(AllTables.Area_ped < 8));
% hist(AllTables.Area_ped(i), 200);
% xlabel('mVnS');
% title('Pedestal');
% 
% subplot(2,3,2)
% i = find((AllTables.Area_ser > 5) &(AllTables.Area_ser < 80));
% hist(AllTables.Area_ser(i), 200);
% xlabel('mVnS');
% title('SER');
% 
% subplot(2,3,3)
% AreaPedSer = [AllTables.Area_ped AllTables.Area_ser];
% i = find((AreaPedSer > -8) &(AreaPedSer < 80));
% hist(AreaPedSer(i), 200);
% xlabel('mVnS');
% title('SER+Ped.');
% 
% subplot(2,3,4)
% i = find((AllTables.Area_s1tail > 1)&(AllTables.Area_s1tail < 5000));
% hist(AllTables.Area_s1tail(i), 200);
% xlabel('mVnS');
% title('S1 tail');
% 
% subplot(2,3,5)
% i = find((AllTables.Area_s1 > 3) &(AllTables.Area_s1 < 10E3));
% hist(AllTables.Area_s1(i), 200);
% xlabel('mVnS');
% title('S1');
% 
% subplot(2,3,6)
% AreaS1Peaks = AllTables.Area_s1 + AllTables.Area_s1tail;
% i = find((AreaS1Peaks > 1) &(AreaS1Peaks < 10E3));
% hist(AreaS1Peaks(i), 200);
% xlabel('mVnS');
% title('Full S1');