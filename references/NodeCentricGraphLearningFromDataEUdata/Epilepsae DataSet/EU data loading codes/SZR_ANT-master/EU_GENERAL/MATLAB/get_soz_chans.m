% This script outputs the list of bipolar SOZ channels to a text file in
% this directory:
%  ~/SZR_ANT/EU_METADATA/SOZ_CHANS/
%
% The file is to be used to figure out which channels to use for each seed
% channel for computing PLV

if ismac,
    root_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/';
else
    root_dir='/home/dgroppe/GIT/SZR_ANT/';
end

%% Choose sub to process
sub_id=1096;
sub_id=1077;
subs=[970, 253, 565, 590, 620, 958, 264, 273, 862, 1125];


for sub_id=subs,
    %% Load bad chans and ignore them
    badchan_fname=fullfile(root_dir,'EU_METADATA','BAD_CHANS',sprintf('bad_chans_%d.txt',sub_id));
    badchans=csv2Cell(badchan_fname);
    if strcmpi(badchans{1},'None'),
        badchans=[];
    end
    fprintf('# of bad chans: %d\n',length(badchans));
    
    
    % Load list of SOZ channels and the number of samples for each
    indir=fullfile(root_dir,'EU_GENERAL','EU_GENERAL_FTRS');
    infname=fullfile(indir,sprintf('%d_szr_sample_size',sub_id));
    fprintf('Loading counts of # of szr observations/electrode in %s\n',infname);
    % save(infname,'n_tpt_ct','soz_chans_bi','ftr_fs');
    load(infname);
    
    n_chan=size(soz_chans_bi,1);
    fprintf('# of SOZ Channels (good & bad): %d\n',n_chan);
    fprintf('Feature sampling rate: %f Hz\n',ftr_fs);
    good_chan_ids=[];
    for a=1:n_chan,
        temp_label=sprintf('%s-%s',soz_chans_bi{a,1},soz_chans_bi{a,2});
        if findStrInCell(temp_label,badchans),
            fprintf('SOZ channel %s is bad. Ignoring it.\n',temp_label);
        else
            good_chan_ids=[good_chan_ids a];
            fprintf('%s-%s # of obs: %d\n',soz_chans_bi{a,1},soz_chans_bi{a,2},n_tpt_ct(a));
        end
    end
    soz_chans_bi=soz_chans_bi(good_chan_ids,:);
    n_chan=size(soz_chans_bi,1);
    fprintf('# of SOZ Channels (just good): %d\n',n_chan);
    clear good_chan_ids
    
    %% Output usable SOZ channels to text file
    out_fname=fullfile(root_dir,'EU_METADATA','SOZ_CHANS',sprintf('%d_bi_soz_chans.txt',sub_id));
    fprintf('Save list of bipolar SOZ channels to %s\n',out_fname);
    
    fid=fopen(out_fname,'w');
    for a=1:n_chan,
        fprintf(fid,'%s-%s\n',soz_chans_bi{a,1},soz_chans_bi{a,2});
    end
    fclose(fid);
end