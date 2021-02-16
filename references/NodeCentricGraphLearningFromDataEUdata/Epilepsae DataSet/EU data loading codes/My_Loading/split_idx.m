function [file_info, train_idx, test_idx] = split_idx(file_info, load_core)
% file_info = sortFilesByTime(file_info);
loop_n_files = length(file_info); % n_szrs;
cli_szr_info = load_core.cli_szr_info;
ictal_idx = [];
test_idx = nan;


for floop=1:loop_n_files,
    file = file_info(floop);
    field_names = fieldnames(file);
    field_names_with_onset = field_names(find(contains(field_names,'onset')));
    if(any(contains(field_names_with_onset, 'clin_szr_onsets_sec')) &&~isempty(file.clin_szr_onsets_sec) ||...
              any(contains(field_names_with_onset, 'clinical_onset_sec')) && ~isempty(file.clinical_onset_sec))
        [file_info(floop), cli_szr_info] = corresponding_szr_file(file, cli_szr_info);
        ictal_idx = [ictal_idx floop];
    end
end
nonictal_idx = setdiff([1:length(file_info)], ictal_idx);
if(~isnan(load_core.howmany_to_load))
    nonictal_idx = sort(nonictal_idx(datasample([1:length(nonictal_idx)], load_core.howmany_to_load,'Replace',false)));
end
if(contains(load_core.sel_mode,'feature')||contains(load_core.sel_mode,'setting'))
    num_train_files_szr = floor(length(ictal_idx)*load_core.train_test_ratio);
    num_train_files_nonszr = floor(length(nonictal_idx)*load_core.train_test_ratio);
    
    train_ictal_idx = ictal_idx(1:num_train_files_szr); % datasample(ictal_idx, num_train_files_szr,'Replace',false);
    train_nonictal_idx = nonictal_idx(1:num_train_files_nonszr); % datasample(nonictal_idx, num_train_files_nonszr,'Replace',false);
    
    train_idx = cat(2, train_ictal_idx, train_nonictal_idx); % sort
    test_nonictal_idx = setdiff(nonictal_idx, train_nonictal_idx);
    test_idx = cat(2, sort(cat(2,setdiff(ictal_idx, train_ictal_idx),test_nonictal_idx(1:5))), test_nonictal_idx(6:length(test_nonictal_idx)));
        
elseif(contains(load_core.sel_mode,'structure'))
    train_idx = nonictal_idx;
end
if(contains(load_core.sel_mode,'TVGraphLasso'))
    train_idx = nonictal_idx;
    test_idx = train_idx;
    return;
end
end

