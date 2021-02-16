subs=[264, 273, 862, 1125];
fp_per_day=[2694, 2303, 1907, 1424];
sens=[1, 1, 6/7, 1];

fprintf('Mean (SD): False positives per day %.1f (%.1f)\n', ...
    mean(fp_per_day),std(fp_per_day));
fprintf('Mean (SD): Clinical szr sensitivity %.1f (%.1f)\n', ...
    mean(sens),std(sens));

% Mean (SD): False positives per day 2082.0 (543.7)