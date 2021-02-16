function [ out_core ] = concat_out_core( core1, core2 )

out_core = OutCore();

out_core.X = cat(1, core1.X, core2.X);
out_core.y = cat(1, core1.y, core2.y);
out_core.conv_sizes = cat(1, core1.conv_sizes, core2.conv_sizes);
out_core.sel_win_nums = cat(1, core1.sel_win_nums, core2.sel_win_nums);
out_core.clip_sizes = cat(1, core1.clip_sizes, core2.clip_sizes);

end

