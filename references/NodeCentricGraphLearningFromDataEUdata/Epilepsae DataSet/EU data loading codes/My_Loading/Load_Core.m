classdef Load_Core
    
    properties
        Fs;
        target_Fs;
        num_windows;
        howmany_to_load;
        data_conversions;
        num_classes;
        down_sampl_ratio;
        freq_bands;
        only_seizures;
        welchs_win_len;
        welchs_stride;
        state_win_lengths;
        concat;
        window_size;
        stride;
        n_pre_szr;
        soz_ch_ids;
        sel_win_nums;
        clip_sizes;
        conv_sizes;
        window_size_sec;
        pre_szr_sec;
        stride_sec;
        min_sel_hours;
        chans_bi;
        file_root;
        meta_root;
        cli_szr_info;
        train_test_ratio;
        sel_mode;
        sub_id;
        total_n_nonszr;
        n_nonszr_pfile;
        adjacency_func;
        class_gen_mode;
        feature_load_mode;
    end
    
    methods
    end
    
end

