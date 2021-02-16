function [ n_wins ] = total_szr_wins(load_core)

total_szr_len=0;
n_szrs = length(load_core.cli_szr_info);
for i=1:n_szrs
%     file = load_core.cli_szr_info(i);
%     pat = bin_file(file.clinical_fname);
    total_szr_len = total_szr_len + load_core.cli_szr_info(i).clinical_offset_sec - load_core.cli_szr_info(i).clinical_onset_sec; 
end
n_wins = round(num_wins_stride(total_szr_len, load_core.window_size_sec, load_core.stride_sec));
end

