%% loading EU data files
%% Initialization
clear all;
linux = false;
subs = [253, 264, 273, 565, ...
              548, 384, 375, 583, 590, 620, 862, 916, 922, 958, 970, 1084, 1096, 1146, ...
                   115, 139, 442, 635, 818, 1073, 1077, 1125, 1150, 732, 13089, 13245]; %  

subs = [620,1125,565,958,273,442,1096,590,970,1077]; %  253, 264, 273, 590, 620, 862, 922, 1084
% subs = [253, 1096, 1125, 548, 442];

subs = [253];
if(linux)
    addpath(genpath('/home/ghnaf/EU data loading codes'));
end

% Loading properties
freq_bands=[0.1 4; 4 8; 8 13; 13 30; 30 50; 70 100];
freq_band_labels={'DeltaMag','ThetaMag','AlphaMag','BetaMag','GammaMag','HGammaMag'};
load_core = Load_Core();
load_core.window_size_sec = 2.5; % 60; % 
load_core.stride_sec = 1.5; % 60; % 
load_core.pre_szr_sec = 10;
load_core.min_sel_hours = 1;
load_core.target_Fs = 256;
load_core.howmany_to_load = nan;
load_core.num_windows = nan;
load_core.data_conversions = [myFFT()]; % normalizedFFT(), myFFT(), myCorr(), noChange()
load_core.num_classes = 2;
load_core.down_sampl_ratio = nan;
load_core.freq_bands = freq_bands;
load_core.welchs_win_len = 1;
load_core.welchs_stride = 0.75;
load_core.state_win_lengths = [180];
load_core.concat = 1;
n_pre_szr = round(num_wins_stride(load_core.pre_szr_sec, load_core.window_size_sec, load_core.stride_sec));
load_core.n_pre_szr = n_pre_szr;
load_core.train_test_ratio = 0.5;
load_core.class_gen_mode = 'pre+det'; % 'detection', 'prediction', 'pre+det'
load_core.adjacency_func= 'invcov'; %'corr', 'SRPM', 'invcov'
load_core.sel_mode = 'setting'; % 'structure-interictal', 'feature-both', 'setting', 'TVGraphLasso', 'conv_size'
load_core.feature_load_mode = 'train';  % 'train', 'test', 'train-test'
if(linux)
    load_core.file_root = '/mnt/EU';
    load_core.meta_root = '/home/ghnaf/EU data loading codes/SZR_ANT-master';
else
    load_core.file_root = 'C:/EU'; 
    load_core.meta_root = 'C:/Users/Nafiseh Ghoroghchian/Dropbox/PhD/EU data loading codes/SZR_ANT-master'; 
end
%% 
for sub_id=subs
%     try
        tic;
        if(linux)
            save_root = '/home/ghnaf';
        else
            save_root = 'C:/EU'; 
        end
        inv_dir = get_inv_dir(sub_id);
        
        
        %% Get all the files with clinical szrs and their SOZ
        load_core.sub_id = sub_id;
        load_core.cli_szr_info=get_szr_fnames(sub_id, load_core.meta_root, load_core.file_root);
        n_szrs=length(load_core.cli_szr_info);
%         continue;
        %% Get list of all files and the timing of all szrs (clinical and subclinical)
        file_info=get_fnames_and_szr_times(sub_id, load_core.meta_root);
        n_files=length(file_info);
        %% Load bad chans and ignore them
        badchan_fname=fullfile(load_core.meta_root,'EU_METADATA','BAD_CHANS',sprintf('bad_chans_%d.txt',sub_id));
        badchans=csv2Cell(badchan_fname);
        if strcmpi(badchans{1},'None'),
            badchans=[];
        end
        fprintf('# of bad chans: %d\n',length(badchans));
        %% Get channel labels
        bipolar_labels=derive_bipolar_pairs(sub_id, load_core.meta_root);
        n_chan=size(bipolar_labels,1);
        %% Get unique SOZ chans
        soz_chans_mono=[];
        for a=1:n_szrs,
            soz_chans_mono=union(soz_chans_mono,load_core.cli_szr_info(a).clinical_soz_chans);
        end
        fprintf('Monopolar SOZ chans are:\n');
        disp(soz_chans_mono); %later

        soz_chans_bi=cell(1,1);
        soz_bi_ct=0;
        for cloop=1:n_chan,
            use_chan=0;
            for a=1:2,
                if ismember(bipolar_labels{cloop,a},soz_chans_mono),
                    use_chan=1;
                end
            end
            if use_chan,
                soz_bi_ct=soz_bi_ct+1;
                soz_chans_bi{soz_bi_ct,1}=bipolar_labels{cloop,1};
                soz_chans_bi{soz_bi_ct,2}=bipolar_labels{cloop,2};
            end
        end
    %     fprintf('Bipolar SOZ chans (Good & Bad) are:\n');
        disp(soz_chans_bi); 

        [soz_chans_bi, n_soz_chan] = selection_good_channels(soz_chans_bi, badchans); %later
    %     fprintf('# of SOZ Channels (just good): %d\n',n_soz_chan);
        %% Get good chan
        [load_core.chans_bi, n_chan] = selection_good_channels(bipolar_labels, badchans);
        soz_ch_ids = channel_id(load_core.chans_bi, soz_chans_bi)-1;
        %% loop on files
        %  later? is it all files, ictals too??
        % loop_files = cat(1,cli_szr_info',file_info); % file_info; % 
        % Figure out how many non-szr samples to draw from each file
        % n_nonszr_obs=ceil(n_tpt_ct(cloop)/loop_n_files); % later 
        
%         tt_szr_boundray = floor(length(cli_szr_info)*train_test_ratio);
        [file_info, train_idx, test_idx] = split_idx(file_info, load_core);
        outdir = fullfile(save_root, strcat('EU features ','Raw_', num2str(load_core.window_size_sec), load_core.class_gen_mode, '_preszr_', num2str(load_core.pre_szr_sec)));
        
        if(contains(load_core.sel_mode,'feature-both')||contains(load_core.sel_mode,'conv_size'))
            outdir = fullfile(outdir,inv_dir);
            outfname = fullfile(outdir, sprintf('pat_FR_%d',sub_id));
            outMatFileName = strcat(outfname, '.mat');
            fileExistFlag = false;
            if(exist(outMatFileName,'file'))
                fileExistFlag = true;
            end
            if ~exist(outdir,'dir'),
                mkdir(outdir);
            end
            if fileExistFlag,
                load(outfname);
            end
            % training
            load_core.total_n_nonszr = total_szr_wins(load_core)*10;
%             max( num_wins_stride(load_core.min_sel_hours*3600, load_core.window_size_sec, load_core.stride_sec));
            load_core.n_nonszr_pfile = round(load_core.total_n_nonszr/length(file_info));
            fprintf('# n_nonszr window per file: %d \n ', load_core.n_nonszr_pfile);
            if(contains(load_core.sel_mode,'conv_size'))
                [out_core_train] = loading_file_main(sub_id, file_info(train_idx)', load_core);
                conv_sizes = out_core_test.conv_sizes;
            end
            if(~contains(load_core.sel_mode,'conv_size') ) % && contains(load_core.feature_load_mode, 'train') &&((fileExistFlag && size(X_train, 1)==1 && isnan(X_train))||(~fileExistFlag))
                [out_core_train] = loading_file_main(sub_id, file_info(train_idx)', load_core);
        %         [out_core_temp] = loading_file_main(sub_id, cli_szr_info(1:tt_szr_boundray), load_core);
        %         out_core_train = concat_out_core(out_core_train, out_core_temp);
        %         clear out_core_temp;
                X_train = out_core_train.X;
                y_train = out_core_train.y;
                sel_win_nums_train = out_core_train.sel_win_nums;
                clip_sizes_train = out_core_train.clip_sizes;
                conv_sizes = out_core_test.conv_sizes;
           
                disp('train X shape is: ')
                disp(size(X_train))
                disp('train y shape is: ')
                disp(size(y_train))
                fprintf('# of training szr labels \n', sum(y_train==1))
            elseif(~contains(load_core.sel_mode,'conv_size') && ~exist('X_train', 'var'))
                X_train = NaN;
                y_train = NaN;
                sel_win_nums_train = NaN;
                clip_sizes_train = NaN;
            end
            
            %% testing
            if(~contains(load_core.sel_mode,'conv_size') && contains(load_core.feature_load_mode, 'test')&&((fileExistFlag && size(X_test, 1)==1 && isnan(X_test))||(~fileExistFlag))) % 
                [out_core_test] = loading_file_main(sub_id, file_info(test_idx)', load_core);
                X_test = out_core_test.X;
                y_test = out_core_test.y;
                sel_win_nums_test = out_core_test.sel_win_nums;
                clip_sizes_test = out_core_test.clip_sizes;
                conv_sizes = out_core_test.conv_sizes;

                disp('test X shape is: ')
                disp(size(X_test))
                disp('test y shape is: ')
                disp(size(y_test))
                fprintf('# of testing szr labels = %d \n', sum(y_test==1))
            elseif(~contains(load_core.sel_mode,'conv_size') && ~exist('X_test', 'var'))
                X_test = NaN;
                y_test = NaN;
                sel_win_nums_test = NaN;
                clip_sizes_test = NaN;
            end
            %% save features
            load_core.soz_ch_ids = soz_ch_ids;
    %         load_core.sel_win_nums = sel_win_nums;
    %         load_core.clip_sizes = clip_sizes;
            try 
                load_core.conv_sizes = out_core_train.conv_sizes;
                conv_sizes = out_core_train.conv_sizes;
            catch
                load_core.conv_sizes = out_core_test.conv_sizes;
                conv_sizes = out_core_test.conv_sizes;
            end
            window_size_sec = load_core.window_size_sec;
            stride_sec = load_core.stride_sec;
            
            
            save(outfname,'X_train','y_train','X_test','y_test','conv_sizes','soz_ch_ids', ...
                                'sel_win_nums_train','sel_win_nums_test','clip_sizes_train',...
                                    'clip_sizes_test','n_pre_szr','stride_sec', 'window_size_sec',...
                                        'conv_sizes','-v7.3');
        end
        if(contains(load_core.sel_mode,'structure'))
            load_core.n_nonszr_pfile = NaN;
            outdir = fullfile(save_root, 'EU side info',inv_dir);
            if ~exist(outdir,'dir'),
                mkdir(outdir);
            end
            basefname = fullfile(outdir, sprintf('pat_FR_%d',sub_id));
            outfname = strcat(basefname,'_sideAdj_',load_core.adjacency_func);
%             if exist(outfname,'file')
%                 continue;
%             end
            out_core = loading_file_main(sub_id, file_info(train_idx)', load_core);
%             adj_all = out_core.A_all;
            adj_means = squeeze(mean(out_core.A_means));
            adj_vars = mean(out_core.A_vars);
%             save(strcat(outfname,'_sideAdjAll_',load_core.adjacency_func),'adj_all')
            save(outfname,'adj_means','adj_vars')
        elseif(contains(load_core.sel_mode,'setting'))
            load_core.n_nonszr_pfile = NaN;
            outdir = fullfile(outdir, 'EU settings',inv_dir);
            if ~exist(outdir,'dir'),
                mkdir(outdir);
            end
            basefname = fullfile(outdir, sprintf('pat_FR_%d',sub_id));
            outfname = strcat(basefname,'_setting');
%             if exist(outfname,'file')
%                 continue;
%             end
            train_files = file_info(train_idx)';
            test_files = file_info(test_idx)';
            save(outfname,'train_files', 'test_files','file_info', 'load_core')
        elseif(contains(load_core.sel_mode,'TVGraphLasso'))
            cd('C:/Program Files/MATLAB/R2017a/cvx')
            cvx_setup
            settings = TVGraphLassoSetting();
            settings.B = 100;
            settings.landa = 1;
            settings.mode = 'two';
            out_core = loading_file_main(sub_id, file_info(train_idx)', load_core);
            [Theta, A] = myTVGraphLasso( out_core.all_covs(1:10,:,:), settings );
            outdir = fullfile(save_root, 'TVGraphLasso',inv_dir);
            if ~exist(outdir,'dir'),
                mkdir(outdir);
            end
            outfname = fullfile(outdir, sprintf('pat_FR_%d',sub_id));
            save(outfname,'Theta', 'A', 'load_core')
        end
        fprintf('------ Patient # %d is Done in %f seconds!\n', sub_id, toc)
%         clear X_train, y_train, X_test, y_test;
%     catch
%         continue;
%     end
end

