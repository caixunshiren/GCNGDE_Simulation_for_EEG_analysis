function [out_file_info] = sortFilesByTime(file_info)
loop_n_files = length(file_info);

% fileDates = FileDate();
% time = {'2014/8/4 6:42';'2014/8/5 0:14';'2015/1/31 11:41';'2015/2/1 1:02';'2014/11/9 11:29';'2014/12/31 1:02'};
% x = (1:6)';



time = cell(loop_n_files,1);
for floop=1:loop_n_files,
    file = file_info(floop);
    string = strsplit(file.file_onset_hrs{1}, '.');
    time{floop} = string{1};
%     splittedDate = strsplit(stringDate, {'-',' ',':','.'});
%     fileDates.year = [fileDates.year str2num(splittedDate{1})];
%     fileDates.month = [fileDates.year inMonth];
%     fileDates.day = [fileDates.year inDay];
%     fileDates.hour = [fileDates.year inHour];
%     fileDates.minute = [fileDates.year inMinute];
%     fileDates.second = [fileDates.year inSecond];
end
time = datetime(time,'InputFormat','yyyy-MM-dd HH:mm:ss');
t = table(time,file_info);
t = sortrows(t,'time');
end

