function [ieeg, classes, sel_ids] = select_ftrs(ieeg, classes, load_core)
non_szr_ids = find(classes==0);
szr_ids = find(classes>0.5);
if(length(szr_ids)>1)
    pre_szr_ids = (max(1,szr_ids(1)-load_core.n_pre_szr):szr_ids(1))';
else
    pre_szr_ids =[];
end
% Randomly select subset of non-szr features
use_non_szr_ids = non_szr_ids(randi(length(non_szr_ids),1,load_core.n_nonszr_pfile));
sel_ids = cat(1, use_non_szr_ids, pre_szr_ids, szr_ids);
sel_ids = sort(unique(sel_ids));
otherdims = repmat({':'},1,ndims(ieeg)-1);
ieeg = ieeg(sel_ids, otherdims{:});
classes = classes(sel_ids);


end

