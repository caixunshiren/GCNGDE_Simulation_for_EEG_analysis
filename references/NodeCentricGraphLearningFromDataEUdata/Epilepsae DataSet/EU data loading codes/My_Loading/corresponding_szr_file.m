function [file, cli_szr_info] = corresponding_szr_file(file, cli_szr_info)

for counter=1:length(cli_szr_info)
    szr_file = cli_szr_info(counter);
    szr_fname = strsplit(szr_file.clinical_fname,'\');
    szr_fname = string(szr_fname(length(szr_fname)));
    szr_fname = strsplit(szr_fname,'.');
    szr_fname = szr_fname(1);
    
    file_fname = strsplit(file.fname,'.');
    file_fname = szr_fname(1);
    if(strcmp(file_fname, szr_fname))
        cli_szr_info(counter)=[];
        return;
    end
end

end

