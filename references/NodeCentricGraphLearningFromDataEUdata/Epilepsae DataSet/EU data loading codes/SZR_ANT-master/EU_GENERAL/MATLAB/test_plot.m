%%
figure(10);
clf();
subplot(2,1,1);
%plot(targ_raw_ieeg);
plot(time_dec(start_targ_id:stop_targ_id),ieeg(start_targ_id:stop_targ_id));
hold on;
plot(targ_raw_ieeg_sec,targ_raw_ieeg,'m:');
axis tight;
ylim=get(gca,'ylim');
mn_ylim=mean(ylim);
plot(time_dec(start_targ_id:stop_targ_id), ...
    targ_win_dec(start_targ_id:stop_targ_id)*(ylim(2)-mn_ylim)+mn_ylim, ...
    'r-');
% plot(time_dec(start_targ_id:stop_targ_id), ...
%     single(targ_win_dec(start_targ_id:stop_targ_id))*250,'r-');
axis tight;

subplot(2,1,2);
%plot(se_time_sec,se_ftrs);
hold on;
temp_labels=cell(1,6);
for a=1:6,
    plot(se_time_sec,se_ftrs(a,:),'linewidth',2);
    temp_id=find(ftr_labels{a}=='_');
    temp_labels{a}=ftr_labels{a}(1:temp_id(1)-1);
end
plot(se_time_sec,se_ftrs);
legend(temp_labels);
axis tight;

%%
figure(11);
plot(targ_raw_ieeg_sec,targ_raw_ieeg,'b-');
%plot(targ_raw_ieeg_sec,targ_raw_ieeg);

%%
figure(12);
plot(se_ftrs(1,:),ftrs_z(1,:),'.');