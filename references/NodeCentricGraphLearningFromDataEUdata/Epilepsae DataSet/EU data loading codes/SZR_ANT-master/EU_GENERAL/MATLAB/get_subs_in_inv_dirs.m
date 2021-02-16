% This script lists the subjects in each of the three EU inv* subirectories

f=dir('/Volumes/ValianteLabEuData/EU/inv/');
f=dir('/Volumes/ValianteLabEuData/EU/inv2/');
f=dir('/Volumes/ValianteLabEuData/EU/inv3/');
% subs=cell(1,1);
% subs_ct=0;
subs=[];
for a=1:length(f),
    if strncmp(f(a).name,'pat_',4),
%         subs_ct=subs_ct+1;
        %subs{subs_ct}=str2num(f(a).name(8:end));        
       subs=[subs ', ' f(a).name(8:end)];
    end
    %if isdir(fullfile(pth,f(a).name
end

disp(subs);