function DriftFieldLior(Path, FiguresOn, FileName, RUNnumber)
addpath(Path);
warning('off', 'all');
cd(Path);

% [~, currentFolderName] = fileparts(pwd);
% TausVal = readtable(['../', RUNnumber, 'TauValues.txt']);
% TausVal.Properties.VariableNames = {'PNum', 'Tau'};
% 
% idx = TausVal.PNum == string(currentFolderName);      % Logical index where first column is "S1"
% tau = TausVal.Tau(idx);
tau = 96.6739;


AnalysisPath = fullfile(Path, 'Analysis');
if ~isfolder(AnalysisPath)
    mkdir(AnalysisPath);
end
SaveName = FileName;
SavePATH = fullfile(AnalysisPath, 'resultNTaus');
mkdir(SavePATH);

A = dir([FileName, '*.wfm']);
[~, index] = sort({A.date});
A = A(index);

params = struct('smoothwin', 0, 'std2avg',   3,   'MaxS2dist',  2000, 'Npeak',   30, 'groupthr', 0,...
    'groupthr2', 0, 's1wthrLow', 200, 's1wthrHigh', 2e3,   's2wthr', 14e3);

serparams = struct('SERthr', 1.5, 'SERMinPeakDistance', 0, 'SERMinPeakProminenceSER', 1, 'smoothwin', 0);

Area_s1_file   = [];
Area_S1tailpeaks_file = [];
Area_ser_file  = [];
Area_ped_file  = [];
Datefile = [];
Timefile = [];
WFile   =[];

for k = 1:length(A)
    % k = 1119;
    disp(['Processing file: ', num2str(k), ' of: ', num2str(length(A))]);
    % Split the string into date and time
    splitData = split(A(k).date, ' ');
    datePart = splitData{1}; % '09-Mar-2025'
    timePart = splitData{2}; % '04:42:12'

    A1 = fullfile(Path, [FileName, num2str(k), 'Wfm_Ch2.wfm']);
    [sig_PMT1, t_PMT1, ~, ~, ~] = wfm2readALLframesScaled(A1, 1);

    tbin = abs(diff(t_PMT1(1:2)));
    fullscaleWin = int32(length(t_PMT1)*tbin); %% in nsec unit

    win1usec   = round(1000 / tbin);
    win2usec   = round(2000 / tbin);
    win5usec   = round(5000 / tbin);
    win10usec  = round(10000 / tbin);
    win20usec  = round(20000 / tbin);
    win30usec  = round(30000 / tbin);
    win5nsec   = round(5 /tbin);
    win10nsec  = round(10 / tbin);
    win30nsec  = round(30 / tbin);
    win100nsec = round(100 / tbin);
    win200nsec = round(200 / tbin);
    win300nsec = round(300 / tbin);
    
    tauChannel = int32(round(tau/tbin, 1));
    y_sig1 = sig_PMT1;
    results = struct('date', [], 'time', [],...
                     'Area_5tau_s1', [], 'Area_6tau_s1', [],...
                     'Area_7tau_s1', [], 'Area_8tau_s1', [],...
                     'Area_9tau_s1', [], 'Area_10tau_s1', []);

    serparams.SERMinPeakDistance = win30nsec;

    params.smoothwin = win100nsec;
    params.groupthr = win1usec;
    params.groupthr2 = win300nsec;
    params.SERMinPeakDistance = win100nsec;
    LLD = win5usec;
    Area_s1_5_frame  = [];
    Area_s1_6_frame  = [];
    Area_s1_7_frame  = [];
    Area_s1_8_frame  = [];
    Area_s1_9_frame  = [];
    Area_s1_10_frame = [];

    Area_ped_frame = [];
    Area_ser_frame = [];
    Area_S1tailpeaks_frame = [];
    Pad_area_All = [];
    DateFrame = [];
    TimeFrame = [];
    WFrame = [];
    for frame = 1:size(y_sig1, 1)
        % frame = 97;
        disp(['frame', num2str(frame), ' of ', num2str(size(y_sig1, 1))]);
        Area_S1_5_wave = [];
        Area_S1_6_wave = [];
        Area_S1_7_wave = [];
        Area_S1_8_wave = [];
        Area_S1_9_wave = [];
        Area_S1_10_wave = [];


        y_sig1(frame, :) = y_sig1(frame, :) - mean(y_sig1(frame, 1:win1usec));
        smoothed_signal = movmean(y_sig1(frame, :), params.smoothwin);
        std_smooth_L = std(smoothed_signal(1:LLD));
        avg_smooth_L = -1*mean(abs(smoothed_signal(1:LLD)));
        S1thr = avg_smooth_L - 10 * std_smooth_L;
        thr1 = find(smoothed_signal < S1thr);
        thr2S1 = find(smoothed_signal < 0.5*min(smoothed_signal));

        if isempty(thr1) || length(thr1) < 5 || (thr1(end) >= length(t_PMT1)-win100nsec)
            continue;
        end

        groups  = groupNumbersFast(thr1, params.groupthr);
        % Lets find if there is more than 1 S1
        groups2S1  = groupNumbersFast(thr2S1, params.groupthr2);
        if length(groups2S1) > 1
            continue;
        end

        S1lim = groups{1,1};
        w1 = t_PMT1(S1lim(end)) - t_PMT1(S1lim(1));

        if(w1 <= win100nsec*tbin) || (w1 > 35*win1usec*tbin)
            continue;
        end
        S15Taulim = [groups{1,1}(1),  groups{1,1}(1)+5*tauChannel];
        S16Taulim = [groups{1,1}(1),  groups{1,1}(1)+6*tauChannel];
        S17Taulim = [groups{1,1}(1),  groups{1,1}(1)+7*tauChannel];
        S18Taulim = [groups{1,1}(1),  groups{1,1}(1)+8*tauChannel];
        S19Taulim = [groups{1,1}(1),  groups{1,1}(1)+9*tauChannel];
        S110Taulim = [groups{1,1}(1), groups{1,1}(1)+10*tauChannel];
        
        if(S15Taulim(2) >= length(t_PMT1)) || (S16Taulim(2) >= length(t_PMT1)) || (S17Taulim(2) >= length(t_PMT1)) || (S18Taulim(2) >= length(t_PMT1)) || (S19Taulim(2) >= length(t_PMT1)) || (S110Taulim(2) >= length(t_PMT1))
            Area_S1_5_wave = -1000;
            Area_S1_6_wave = -1000;
            Area_S1_7_wave = -1000;
            Area_S1_8_wave = -1000;
            Area_S1_9_wave = -1000;
            Area_S1_10_wave = -1000;
        else
        Area_S1_5_wave = abs(trapz(t_PMT1(S15Taulim(1):S15Taulim(2)), y_sig1(frame, S15Taulim(1):S15Taulim(2))));
        Area_S1_6_wave = abs(trapz(t_PMT1(S16Taulim(1):S16Taulim(2)), y_sig1(frame, S16Taulim(1):S16Taulim(2))));
        Area_S1_7_wave = abs(trapz(t_PMT1(S17Taulim(1):S17Taulim(2)), y_sig1(frame, S17Taulim(1):S17Taulim(2))));
        Area_S1_8_wave = abs(trapz(t_PMT1(S18Taulim(1):S18Taulim(2)), y_sig1(frame, S18Taulim(1):S18Taulim(2))));
        Area_S1_9_wave = abs(trapz(t_PMT1(S19Taulim(1):S19Taulim(2)), y_sig1(frame, S19Taulim(1):S19Taulim(2))));
        Area_S1_10_wave = abs(trapz(t_PMT1(S110Taulim(1):S110Taulim(2)), y_sig1(frame, S110Taulim(1):S110Taulim(2))));

        if (FiguresOn)
            f=figure();
            subplot(2,1,1)
            plot(t_PMT1, y_sig1(frame, :)); hold on;
            plot(t_PMT1(S1Taulim(1):S1Taulim(2)), y_sig1(frame, S1Taulim(1):S1Taulim(2)));
            plot(t_PMT1, smoothed_signal, LineStyle="-",Color='k',LineWidth=2);
            title(['file: ', num2str(k), ' frame: ', num2str(frame), ' Num S1: ', num2str(length(groups2S1))]);

            subplot(2,1,2)
            plot(t_PMT1(S1Taulim(1):S1Taulim(2)), y_sig1(frame, S1Taulim(1):S1Taulim(2)));
            title(['Area S1: ', num2str(Area_S1_wave)]);

            pause();
            close(f);
        end
        end

        DateFrame      = [DateFrame; datePart];
        TimeFrame      = [TimeFrame; timePart];
        Area_s1_5_frame  = [Area_s1_5_frame; Area_S1_5_wave];
        Area_s1_6_frame  = [Area_s1_6_frame; Area_S1_6_wave];
        Area_s1_7_frame  = [Area_s1_7_frame; Area_S1_7_wave];
        Area_s1_8_frame  = [Area_s1_8_frame; Area_S1_8_wave];
        Area_s1_9_frame  = [Area_s1_9_frame; Area_S1_9_wave];
        Area_s1_10_frame = [Area_s1_10_frame; Area_S1_10_wave];

    end

    results.date          = [results.date        DateFrame             ];
    results.time          = [results.time        TimeFrame             ];
    results.Area_5tau_s1  = [results.Area_5tau_s1     Area_s1_5_frame  ];
    results.Area_6tau_s1  = [results.Area_6tau_s1     Area_s1_6_frame  ];
    results.Area_7tau_s1  = [results.Area_7tau_s1     Area_s1_7_frame  ];
    results.Area_8tau_s1  = [results.Area_8tau_s1     Area_s1_8_frame  ];
    results.Area_9tau_s1  = [results.Area_9tau_s1     Area_s1_9_frame  ];
    results.Area_10tau_s1  = [results.Area_10tau_s1   Area_s1_10_frame  ];


    save(fullfile(SavePATH, [SaveName, '_fileNum_', num2str(k), '.mat']), "results", '-v7.3');
end
end


function groups = groupNumbersFast(indexOverThreshold, threshold)
indexOverThreshold = sort(indexOverThreshold);
groups = {};
idx = [true, diff(indexOverThreshold) > threshold];
splitIdx = find(idx);
for i = 1:length(splitIdx)
    if i == length(splitIdx)
        groups{end+1} = indexOverThreshold(splitIdx(i):end);
    else
        groups{end+1} = indexOverThreshold(splitIdx(i):splitIdx(i+1)-1);
    end
end
end
