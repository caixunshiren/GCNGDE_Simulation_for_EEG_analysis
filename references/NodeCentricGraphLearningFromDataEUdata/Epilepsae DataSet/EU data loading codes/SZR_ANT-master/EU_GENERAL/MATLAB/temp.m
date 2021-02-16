%% Figure out what's going wrong with compute_se_labels_all.m
file_info=get_fnames_and_szr_times(862);

%%
n_files=length(file_info);
for a=1:n_files,
    if strcmpi(file_info(a).fname,'86201102_0141.head'),
   fprintf('%d: %s\n',a,file_info(a).fname);
    end
end

%% Get szr fnames
cli_szr_info=get_szr_fnames(862);
for a=1:length(cli_szr_info),
    dash_id=find(cli_szr_info(a).clinical_fname=='/');
   fprintf('Szr %d: %s\n',a,cli_szr_info(a).clinical_fname(dash_id(end)+1:end)); 
end

%% Plot timing of all file onsets:
%file_info=get_fnames_and_szr_times(862);
n_files2=length(file_info);
f_onset2=zeros(n_files,1);
for a=1:n_files2,
   f_onset2(a)=file_info(a).file_onset_sec(1);
end

%%
figure(1); clf;
plot(f_onset2);
xlabel('Sec');


%% Get and plot szr durations
subs=[115 253 264 273 442 565 590 620 862 922 958 970 1077 1096 1125];
dur=[];
subnszr=[];
ct=0;
for sub_id=subs,
fprintf('Sub %d\n',sub_id);
cli_szr_info=get_szr_fnames(sub_id);
n_cli_szr=length(cli_szr_info);

for a=1:n_cli_szr,
   fprintf('Clinical Szr %d: %d sec\n',a,round(cli_szr_info(a).clinical_offset_sec- ...
       cli_szr_info(a).clinical_onset_sec));
   ct=ct+1;
   dur=[dur (cli_szr_info(a).clinical_offset_sec- ...
       cli_szr_info(a).clinical_onset_sec)];
   subnszr{ct}=[num2str(sub_id) '-' num2str(a)];
end
end

%%
figure(1); clf(); hold on;
for a=1:length(dur),
h=plot(a,dur(a),'o');
clickText(h,subnszr{a});
end
%set(gca,'ylim',[0 500]);
