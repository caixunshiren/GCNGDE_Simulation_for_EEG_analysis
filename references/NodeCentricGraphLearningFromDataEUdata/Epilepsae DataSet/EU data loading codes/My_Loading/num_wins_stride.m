function [ n_wins ] = num_wins_stride( total_len, win_len, stride )

n_wins = floor((total_len-win_len)/stride) +1;

end

