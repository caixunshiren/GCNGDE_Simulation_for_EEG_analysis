function [ good_chans_bi, n_chan ] = selection_good_channels( chans_bi, badchans )

n_chan=size(chans_bi,1);
if(size(badchans,1)>0)
    good_chan_ids=[];
    for a=1:n_chan,
        temp_label=sprintf('%s-%s',chans_bi{a,1},chans_bi{a,2});
        if any(cellfun(@(s) ~isempty(strfind(temp_label, s)), badchans)), %findStrInCell(temp_label,badchans),
            fprintf('Channel %s is bad. Ignoring it.\n',temp_label);
        else
            good_chan_ids=[good_chan_ids a];
            %fprintf('%s-%s # of obs: %d\n',chans_bi{a,1},chans_bi{a,2},n_tpt_ct(a));
            %fprintf('%s-%s # of obs\n',chans_bi{a,1},chans_bi{a,2});
        end
    end
else
    good_chan_ids = 1:n_chan;
end
good_chans_bi=chans_bi(good_chan_ids,:);
n_chan=size(good_chans_bi,1);


end

