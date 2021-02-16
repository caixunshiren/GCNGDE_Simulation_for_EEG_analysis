function [ inv_dir ] = get_inv_dir (sub_id)

% patients have many of them
inv_subs=[1084, 1096, 1146, 253, 264, 273, 375, 384, 548, 565, 583, 590, 620, 862, 916, 922, 958, 970];
inv_subs2=[1073, 1077, 1125, 115, 1150, 139, 442, 635, 818];
inv_subs3=[13089, 13245, 732];
if ~isempty(intersect(sub_id,inv_subs)),
    inv_dir='inv';
elseif ~isempty(intersect(sub_id,inv_subs2)),
    inv_dir='inv2';
elseif ~isempty(intersect(sub_id,inv_subs3)),
    inv_dir='inv3';
else
    error('Could not find sub %d in inv, inv2, or inv3 subdirectories on external hard drive.', ...
        sub_id);
end


end

