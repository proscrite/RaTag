function SER_QE_1PMT_NTaus(Path, FiguresOn, FileName, RUNnumber)

addpath(Path)
warning('off','all');
cd(Path)

A         = dir([FileName, '*.wfm']);
[~,index] = sortrows({A.date}.');
A = A(index); clear index
SaveName = FileName;

% [~, currentFolderName] = fileparts(pwd);
% TausVal = readtable(['../', RUNnumber, 'TauValues.txt']);
% TausVal.Properties.VariableNames = {'PNum', 'Tau'};
% 
% idx = TausVal.PNum == string(currentFolderName);      % Logical index where first column is "S1"
% tau = TausVal.Tau(idx);
% meanTau = load('X:\AXO_Data\dict_S1av_BLS.mat');
tau = 96.6739;

Pointer = 0;
SERthr = 0.5;
SERMinPeakDistance = 5;
SERMinPeakProminenceSER = 1;

Ntaus = [5, 6, 7, 8, 9 ,10];
total_evt = 0;
nTaus_counter = zeros(1, length(Ntaus));

for t=1:length(SERthr)
    SaveFRAMES  = [Path, '/SER_Frames/', 'ser_thr', num2str(SERthr(t)), '_', num2str(datenum(clock)), '/'];
    mkdir(SaveFRAMES);
    AreaS1SER_all = [];
    for k = 1:length(A)-1  %running over all flies  %<====================================
        % k=2;
        Pointer = Pointer +1;
        disp(['Number of evt is :', num2str(nTaus_counter ./ total_evt)]);
        SaveFrameName = [SaveName, num2str(k)];
        A1 = ([FileName, num2str(k), 'Wfm_Ch2.wfm']);
        [sig_PMT1,t_PMT1,info_sig,~,~] = wfm2readALLframesScaled([Path,'/', A1],1);
        t_sig1 = t_PMT1';

        tbin = abs(t_sig1(1)-t_sig1(2));
        tauChannel = int32(round(tau/tbin, 1));


        s1w_file       = [];
        s2w_file       = [];
        dists1s2_file  = [];
        Area_s1_file   = [];
        Area_s2_file   = [];
        PHfile         = [];
        AreaS1BKG_file = [];
        Counter = 0;
        AreaS1SER_file = [];
        Pad_area_file  = [];
        Pad_area_All = [];
        for frame =1:size(sig_PMT1, 1)  %running over all frames of the file
            % frame =606;
            total_evt = total_evt + 1;
            y_sig1    = sig_PMT1(frame,:)-mean(sig_PMT1(frame, 1:1e3));
            disp([' File:',num2str(k),' event:',num2str(frame)]);

            [~, startidx] = min(abs(t_sig1 - 0));
            Area_ser_wave = zeros(1, length(Ntaus));
            All_gaps_Tausidx = zeros(1, length(Ntaus));
            for nT=1:length(Ntaus)
                All_gaps_Tausidx(nT) = Gap_idx(startidx, Ntaus(nT), tauChannel);
            end
            for p=1:length(Ntaus)
                gap_Tausidx = Gap_idx(startidx, Ntaus(p), tauChannel);
                end_Tausid = End_idx(gap_Tausidx, time2ch(20, tbin));
                [pksser, Locsser, widthsser, promsser]= findpeaks(-y_sig1(gap_Tausidx:end_Tausid), 'MinPeakHeight', SERthr(t),...
                    'MinPeakDistance', SERMinPeakDistance, 'MinPeakProminence',SERMinPeakProminenceSER);
                if isempty(Locsser)
                    Area_ser_wave(p) = -100;
                    continue
                end
                serlim = (gap_Tausidx:end_Tausid);
                peaklim = [gap_Tausidx+Locsser(end)-time2ch(5, tbin), gap_Tausidx+Locsser(end)+time2ch(5, tbin)];
                Area_ser_wave(p) = (trapz(t_sig1(serlim), -y_sig1(serlim)));
                nTaus_counter(p) = nTaus_counter(p) + 1;
                if(FiguresOn)
                    f = figure('units','normalized','outerposition',[0 0 1 1]);
                    subplot(2,2,1);
                    plot(t_sig1, y_sig1); hold on;
                    scatter(t_sig1(gap_Tausidx+Locsser(end)), y_sig1(gap_Tausidx+Locsser(end)), 'or', 'filled');
                    xline([t_sig1(serlim(1)), t_sig1(serlim(end))]);
                    xline([t_sig1(gap_Tausidx), t_sig1(end_Tausid)]);
                    ylabel('mV');
                    xlabel('ns');
                    grid("on");
                    title([' File:',num2str(k),' event:',num2str(frame), ' NTau = ', num2str(Ntaus(p))]);
                    hold off;

                    subplot(2,2,2);
                    plot(t_sig1(startidx-time2ch(500, tbin):serlim(2)+time2ch(1000, tbin)),...
                         y_sig1((startidx-time2ch(500, tbin):serlim(2)+time2ch(1000, tbin)))); hold on;
                    scatter(t_sig1(gap_Tausidx+Locsser(end)), y_sig1(gap_Tausidx+Locsser(end)), 'or', 'filled');
                    xline([t_sig1(serlim(1)), t_sig1(serlim(end))], LineWidth=0.5);
                    xline(t_sig1([startidx, All_gaps_Tausidx]), 'r', LineWidth=0.5);
                    xline([t_sig1(peaklim(1)), t_sig1(peaklim(2))], 'b');
                    ylabel('mV');
                    xlabel('ns');
                    grid("on");
                    title(['1 tau = ', num2str(tau), 'nsec']);
                    hold off;

                    subplot(2,2,3)
                    plot(t_sig1(startidx-time2ch(500, tbin):serlim(2)+time2ch(500, tbin)), y_sig1((startidx-time2ch(500, tbin):serlim(2)+time2ch(500, tbin)))); hold on;
                    scatter(t_sig1(gap_Tausidx+Locsser(end)), y_sig1(gap_Tausidx+Locsser(end)), 'or', 'filled');
                    xline([t_sig1(serlim(1)), t_sig1(serlim(end))], LineWidth=0.5);
                    xline([t_sig1(startidx), t_sig1(gap_Tausidx)], 'r', LineWidth=0.5);%%%%%
                    xline([t_sig1(peaklim(1)), t_sig1(peaklim(2))], 'b');
                    ylabel('mV');
                    xlabel('ns');
                    grid("on");
                    title(['1 tau = ', num2str(tau), 'nsec']);
                    hold off;

                    subplot(2,2,4)
                    plot(t_sig1(serlim(1)-time2ch(5, tbin):serlim(end)+time2ch(5, tbin)), y_sig1((serlim(1)-time2ch(5, tbin):serlim(end)+time2ch(5, tbin)))); hold on;
                    scatter(t_sig1(gap_Tausidx+Locsser(end)), y_sig1(gap_Tausidx+Locsser(end)), 'or', 'filled');
                    xline([t_sig1(serlim(1)), t_sig1(serlim(end))], LineWidth=2);
                    xline([t_sig1(peaklim(1)), t_sig1(peaklim(2))], 'b');
                    ylabel('mV');
                    xlabel('ns');
                    title(['Area ', num2str(Area_ser_wave)]);
                    grid("on");
                    hold off;

                    pause();
                    close(f);
                end
            end
            if all(Area_ser_wave == -100)
                continue;
            end
            AreaS1SER_file = [AreaS1SER_file; Area_ser_wave];
        end
        save([SaveFRAMES, SaveFrameName,'.mat'],'AreaS1SER_file','-v7.3');
        AreaS1SER_all = [AreaS1SER_all; AreaS1SER_file];
        % RUNTable = table(AreaS1SER_all, 'VariableNames', {'AreaS1SER_frames'});
        % save([SaveFRAMES, SaveFrameName,'.mat'],'RUNTable','-v7.3');
    end

end

RUNTable = table(AreaS1SER_all(:,1), AreaS1SER_all(:,2), AreaS1SER_all(:,3), ...
    'VariableNames', {'AreaS1SER_5Tau', 'AreaS1SER_8Tau', 'AreaS1SER_10Tau'});
save([Path, '/SER_Frames/', 'AreaSER_NTau.mat'],'RUNTable','-v7.3');
end

function gapTausidx = Gap_idx(strtidx, tau, tauChannel)
gapTausidx = int32(strtidx + tau*tauChannel);
end

function endTausidx = End_idx(gapTausidx, window)
endTausidx = int32(gapTausidx + window);
end

function win2nsec = time2ch(N, tbin)
win2nsec = int32(N/tbin);
end