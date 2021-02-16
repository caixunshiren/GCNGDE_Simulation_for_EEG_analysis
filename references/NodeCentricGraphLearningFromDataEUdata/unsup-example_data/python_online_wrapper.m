function [X, y, conv_sizes, sel_win_nums, clip_sizes] = python_online_wrapper( sub_id, file_ids, mode, outdir )
%X, y, conv_sizes, sel_win_nums, clip_sizes
linux = false;
sub_id = double(sub_id);
file_ids = double(cell2mat(file_ids));

if(linux)
    addpath(genpath('/home/ghnaf/EU data loading codes'));
else
    addpath(genpath('C:\Users\Nafiseh Ghoroghchian\Dropbox\PhD\EU data loading codes'));
end
inv_dir = get_inv_dir(sub_id);
if(linux)
    save_root = '/home/ghnaf/';
else
    save_root = 'C:\EU\';
end
outdir = fullfile(outdir,'EU settings',inv_dir); 
basefname = fullfile(outdir, sprintf('pat_FR_%d',sub_id));
outfname = strcat(basefname,'_setting');
load(outfname);
load_core.sel_mode = 'feature-both';
load_core.n_nonszr_pfile = NaN;
if(contains(mode, 'total'))
    load_core.n_nonszr_pfile = NaN;
end


if(linux)
    load_core.file_root = '/mnt/EU';
    load_core.meta_root = '/home/ghnaf/EU data loading codes/SZR_ANT-master';
else
    load_core.file_root = 'E:\EU';
    load_core.meta_root = 'C:\Users\Nafiseh Ghoroghchian\Dropbox\PhD\EU data loading codes\SZR_ANT-master';
end
% test_files = [test_files(1), test_files(length(test_files)), test_files(2), test_files(length(test_files)-1), test_files(3:length(test_files)-2)];
train_test = [train_files, test_files];
out_core = loading_file_main(sub_id, train_test(file_ids)', load_core);
X = out_core.X;
y = out_core.y;
conv_sizes = out_core.conv_sizes;
sel_win_nums = out_core.sel_win_nums;
clip_sizes = out_core.clip_sizes;
end

