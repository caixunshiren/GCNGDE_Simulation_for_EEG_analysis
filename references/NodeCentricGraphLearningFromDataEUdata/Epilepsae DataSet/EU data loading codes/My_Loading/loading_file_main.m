function [ out_core ] = loading_file_main(sub_id, loop_files, load_core)
ch_loop = load_core.chans_bi; % soz_chans_bi;
loop_n_files = length(loop_files); % min(10,length(loop_files)); % n_szrs;
X = cell(1,0);
y = cell(1,0);
for floop=1:loop_n_files,
    %% Read header
    tic;
    file = loop_files(floop);
%     field_names = fieldnames(file);
%     field_names_with_onset = field_names(find(contains(field_names,'onset')));
%     if(any(contains(field_names_with_onset, 'clin_szr_onsets_sec')) &&~isempty(file.clin_szr_onsets_sec) ||...
%                   any(contains(field_names_with_onset, 'clinical_onset_sec')) && ~isempty(file.clinical_onset_sec))
%         [file, load_core.cli_szr_info] = corresponding_szr_file(file, load_core.cli_szr_info);
%     end
    %         ffname = field_names(find(contains(field_names,'fname')));
    %         ffname = file.(genvarname(ffname{1}));
    
    %% test whether file_info include cli_szr_info: NO!
    field_names = fieldnames(file);
    if(any(contains(field_names,'clinical_fname')))
    %             disp('********************clinical_fname found!!!!!');
    %             return 
        full_data_fname = file.clinical_fname;
    elseif(any(contains(field_names,'fname')))
    %             continue;
        try
            full_data_fname=get_data_fullfname(sub_id,file.fname, load_core.file_root); 
        catch
            full_data_fname = file.fname;
        end
    else
        continue;
    end

    %%
    pat = bin_file(full_data_fname); 
    Fs = pat.a_samp_freq;
    load_core.Fs = Fs;
    load_core.window_size = round(load_core.window_size_sec*load_core.target_Fs);
    load_core.stride = round(load_core.stride_sec*load_core.target_Fs);
    if isempty(Fs),
        error('Could not find file: %s',file.clinical_fname);
    end
    %         fprintf('FS=%f\n',Fs);
    if isempty(Fs),
        error('Could not find file: %s',full_data_fname);
    end
    %         fprintf('# of samples=%d\n',(pat.a_n_samples));

    %% Import entire clip (typically 1 hour long)
    pat.a_channs_cell={ch_loop{:,1}}; % Channel to import
    ieeg = pat.get_bin_signals(1,pat.a_n_samples);
    if isempty(ieeg),
        error('Could not read ieeg');
    end
    pat.a_channs_cell={ch_loop{:,2}}; % Channel to import
    ieeg = ieeg - pat.get_bin_signals(1,pat.a_n_samples);

    %         ieeg_time_sec_pre_decimate=[0:(length(ieeg)-1)]/Fs; % time relative to start of file
    if(contains(load_core.sel_mode,'feature')||contains(load_core.sel_mode,'conv_size'))
        %% Generate classes
        field_names_with_onset = field_names(find(contains(field_names,'onset')));
        %         var_names_with_onset = cellfun(@(s) genvarname(s), field_names_with_onset);
        %         index = cellfun(@(var) ~isempty(file.var), var_names_with_onset) %cellfun(@(s) ~isempty(strfind(field_names, s)), {'onsets'})
        % Clinical Szrs
        if  any(contains(field_names_with_onset, 'clin_szr_onsets_sec')) &&~isempty(file.clin_szr_onsets_sec),
            [classes] = class_generation(ieeg, file, 'clin_szr_onsets_sec', 'clin_szr_offsets_sec', load_core); %later
        % Subclinical Szrs
        elseif  any(contains(field_names_with_onset, 'sub_szr_onsets_sec')) &&~isempty(file.sub_szr_onsets_sec),
            [classes] = class_generation(ieeg, file, 'sub_szr_onsets_sec', 'sub_szr_offsets_sec', load_core); 
        elseif any(contains(field_names_with_onset, 'clinical_onset_sec')) && ~isempty(file.clinical_onset_sec),
            [classes] = class_generation(ieeg, file, 'clinical_onset_sec', 'clinical_offset_sec', load_core);   
        %         elseif ~isempty(file.file_onset_sec),
        %             [szr_class] = class_generation(ieeg, file, 'file_onset_sec', 'file_offset_sec', load_core);   
        else
            classes = zeros(1,size(ieeg,2));
        end
        if(load_core.Fs>load_core.target_Fs)
            [ieeg, classes] = Down_Sample(ieeg, classes, load_core); %later
            load_core.Fs = load_core.target_Fs;
        end
        if(~contains(load_core.sel_mode,'raw'))
        
            %% sliding window
            ieeg = createRollingWindow(ieeg, load_core.window_size, load_core.stride);
            ieeg = permute(ieeg,[2,1,3]);
            classes = squeeze(createRollingWindow(classes, load_core.window_size, load_core.stride));
            classes = max(classes,[],2);
            %% Get non-szr time window ids
            if(isnan(load_core.n_nonszr_pfile))
                inn_sel_win_nums =[ 1:length(classes)]';
            else
                [ieeg, classes, inn_sel_win_nums] = select_ftrs(ieeg, classes, load_core);
            end
            %% data conversion
            [features, conv_sizes] = data_convert(ieeg, load_core);
            if(contains(load_core.sel_mode,'conv_size'))
                break;
            end
%         clear ieeg;
        end
        %% concat new windows with previous windows
        if(load_core.concat)
            if(any(size(X)==0))
                X = features;
                y = classes;
                sel_win_nums = inn_sel_win_nums-1;
                clip_sizes = [0 size(X,1)];
            else
                X = cat(1, X, features); %[X features]; %
                y = cat(1, y, classes); %[y ; classes]; %
                sel_win_nums = cat(1, sel_win_nums, inn_sel_win_nums);
                clip_sizes = cat(1, clip_sizes, [clip_sizes(size(clip_sizes,1),2) size(X,1)]);
            end
        else
            disp('non-concatenation is not implemented yet!');
            return;
        end
        delay = toc;
%         disp('X shape is: ')
%         disp(size(X))
        fprintf('# of szr labels = %d \n', sum(y==1))
%         fprintf('Feature extraction delay per window = %f \n', delay/size(features,1))
    else
        %% sliding window
        ieeg = createRollingWindow(ieeg, load_core.window_size, load_core.stride);
        ieeg = permute(ieeg,[2,1,3]);
    end
    
    
    if(contains(load_core.sel_mode,'structure'))
        adj = adjacency_calc(ieeg, load_core);
        new_A = squeeze(mean(adj));
        if(any(isinf(new_A)))
            continue;
        end
        if(floop==1)
%             A_all = all_A_calc;
            A_means = new_A;
            A_vars = sum(sum(squeeze(var(adj))));
        else
%             A_all = cat(1, A_all, all_A_calc);
            A_means = cat(3, A_means, new_A);
            squeeze(mean(A_means))
            A_vars = [A_vars, sum(sum(squeeze(var(adj))))];
        end
    elseif(contains(load_core.sel_mode,'TVGraphLasso'))
        if(floop==1)
            all_covs = cov3D(ieeg);
        else
            all_covs = cat(1, all_covs, cov3D(ieeg));
        end
    end
    fprintf('%f percent of patient %d \n', floop*100/loop_n_files, sub_id)
end
if(contains(load_core.sel_mode,'conv_size'))
    out_core.conv_sizes = conv_sizes;
elseif(contains(load_core.sel_mode,'feature'))
    out_core = Classif_OutCore();
    out_core.X = X;
    out_core.y = y;
    out_core.conv_sizes = conv_sizes;
    out_core.sel_win_nums = sel_win_nums;
    out_core.clip_sizes = clip_sizes;
elseif(contains(load_core.sel_mode,'structure'))
    out_core = Structural_OutCore();
%     out_core.A_all = A_all;
%     out_core.A_means = squeeze(mean(A_all));
%     out_core.A_vars = squeeze(var(A_all));
    out_core.A_means = permute(A_means, [3,1,2]);
    out_core.A_vars = A_vars;
elseif(contains(load_core.sel_mode,'TVGraphLasso'))
    out_core = TVGraphLasso_OutCore();
    out_core.all_covs = all_covs;
end
end

