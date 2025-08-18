function MeanWFMsScript_NTaus(Path, FiguresOn, FileName, RUNnumber, PNumber)
addpath(Path);
warning('off', 'all');
cd(Path);

A = dir([FileName, '*.wfm']);
[~, index] = sort({A.date});
A = A(index);
MeanWFMs = [];
for k = 1:300 %length(A)
    disp(['Processing file: ', num2str(k), ' of: ', num2str(length(A))]);

    % Split the string into date and time
    splitData = split(A(k).date, ' ');
    datePart = splitData{1}; % '09-Mar-2025'
    timePart = splitData{2}; % '04:42:12'

    A1 = fullfile(Path, [FileName, num2str(k), 'Wfm_Ch1.wfm']);
    [Sig1, Time1, ~, ~, ~] = wfm2readALLframesScaled(A1, 1);
    [~, t1usIdx] = min(abs(Time1 - 1e3)); %% find nearest value to 1usec
    [~, minus3usIdx] = min(abs(Time1 - (-1e3))); %% find nearest value to 1usec


    t_PMT1 = Time1(minus3usIdx:t1usIdx);
    y_sig1 = Sig1(:,minus3usIdx:t1usIdx);
    tbin = abs(diff(t_PMT1(1:2)));
    MeanWFM = [];

    for frame = 1:size(y_sig1, 1)
        disp(['frame', num2str(frame), ' of ', num2str(size(y_sig1, 1))]);

        y_sig1(frame, :) = y_sig1(frame, :) - mean(y_sig1(frame, 1:time2ch(500, tbin)));
        smoothed_signal = movmean(y_sig1(frame, :), time2ch(100, tbin));
        std_smooth_L = std(smoothed_signal(1:time2ch(200, tbin)));
        avg_smooth_L = -1*mean(abs(smoothed_signal(1:time2ch(200, tbin))));
        S1thr = avg_smooth_L - 10 * std_smooth_L;
        thr1 = find(smoothed_signal < S1thr);
        thr2S1 = find(smoothed_signal < 0.5*min(smoothed_signal));

        if isempty(thr1) || length(thr1) < 5 || (thr1(end) >= length(t_PMT1)-time2ch(100, tbin))
            continue;
        end

        % Lets find if there is more than 1 S1
        groups2S1  = groupNumbersFast(thr2S1, time2ch(300, tbin));
        if length(groups2S1) > 1
            continue;
        end
        MeanWFM = [MeanWFM; y_sig1(frame,:)];

        if (FiguresOn)
            f = figure();
            plot(t_PMT1, y_sig1(frame, :)); hold on;
            title(['file num: ', num2str(k), ' event num: ', num2str(frame), ' number of group: ', num2str(length(groups2S1))]);
            pause();
            close(f);
        end
    end
    MeanWFMs = [MeanWFMs; mean(MeanWFM,1)];
end

FinalMeanWFMs = mean(MeanWFMs, 1);

% Get parent folder (one level up)
Path = char(strip(Path, 'right', '\'));  % Remove any trailing backslash if present
parentFolder = fileparts(Path);

save([parentFolder, '\', RUNnumber, '_', PNumber, '_MeanWFM.mat'], 'FinalMeanWFMs');
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

function ch = time2ch(tnsec, tbin)
ch = round(tnsec / tbin);
end
