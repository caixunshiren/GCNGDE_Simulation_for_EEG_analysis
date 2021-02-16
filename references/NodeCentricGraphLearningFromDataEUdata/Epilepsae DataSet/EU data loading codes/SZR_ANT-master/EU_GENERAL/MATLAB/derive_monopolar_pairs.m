function monopolar_labels=derive_monopolar_pairs(patient_number)
%function monopolar_labels=derive_monopolar_pairs(patient_number)
%
% patient number - EU patient # (e.g., 1096)
%
% Note that you need to manually make a file like this for each patient:
% /Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/elec_list_1096.csv

%% Load electrode csv file
if ismac,
    metadata_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/ELEC_LISTS/';
else
    metadata_dir='/home/dgroppe/GIT/SZR_ANT/EU_METADATA/ELEC_LISTS/';
end
elec_csv_fname=fullfile(metadata_dir, ...
    sprintf('elec_list_%d.csv',patient_number));
fprintf('Importing file %s\n',elec_csv_fname);
elec_csv=csv2Cell(elec_csv_fname);

n_group=size(elec_csv,1);
fprintf('%d groupings of electrodes\n',n_group);

clear monopolar_labels
monopolar_ct=0;
for a=1:n_group,
    elec_stem=elec_csv{a,1};
    dim1=str2num(elec_csv{a,2});
    dim2=str2num(elec_csv{a,3});
    if dim1==1,
        %strip or depth
        for b=1:dim2,
            monopolar_ct=monopolar_ct+1;
            monopolar_labels{monopolar_ct,1}=[elec_stem num2str(b)];
        end
    else
        error('I cannot deal with electrodes as grids.');
    end
end

