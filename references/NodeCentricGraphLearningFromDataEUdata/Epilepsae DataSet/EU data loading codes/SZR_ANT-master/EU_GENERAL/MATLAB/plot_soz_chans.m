sub_id=1096;
sub_id=1077;
sub_id=1125;
sub_id=862;
sub_id=253;
sub_id=620;
sub_id=565;
sub_id=273;
sub_id=590;

% Import electrode coordinates
if ismac,
    root_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT';
else
    root_dir='/home/dgroppe/GIT/SZR_ANT';
end
xyz_fname=sprintf('%s/EU_METADATA/ELEC_COORD/elec_coord_%d.csv', ...
    root_dir,sub_id);
xyz_csv=csv2Cell(xyz_fname,',',1);

n_elec=size(xyz_csv,1);
elec_mono_labels=cell(n_elec,1);
elec_xzy=zeros(n_elec,3);

for a=1:n_elec,
    elec_mono_labels{a}=xyz_csv{a,1};
    for b=1:3,
        if xyz_csv{a,b+4}=='-',
            elec_xyz(a,b)=NaN;
        else
            elec_xyz(a,b)=str2num(xyz_csv{a,b+4});
        end
    end
end


%% Load list of SOZ channels
soz_fname=sprintf('%s/EU_METADATA/SOZ_CHANS/%d_bi_soz_chans.txt', ...
    root_dir,sub_id);
soz_bi_csv=csv2Cell(soz_fname);
n_soz=length(soz_bi_csv);
fprintf('# of SOZ chans: %d\n',n_soz);

% Break bipolar labels up into monopolar labels
soz_mono=cell(n_soz*2,1);
ct=0;
for a=1:n_soz,
    id=find(soz_bi_csv{a}=='-');
    ct=ct+1;
    soz_mono{ct}=soz_bi_csv{a}(1:id-1);
    ct=ct+1;
    soz_mono{ct}=soz_bi_csv{a}(id+1:end);
end
soz_mono=unique(soz_mono);


%% Load list of bad channels
bad_fname=sprintf('%s/EU_METADATA/BAD_CHANS/bad_chans_%d.txt', ...
    root_dir,sub_id);
bad_chans=csv2Cell(bad_fname);
n_bad_chans=length(bad_chans);
fprintf('%d bad chans\n',n_bad_chans);

%% Plot electrodes
figure(1); clf(); hold on;
for a=1:n_elec,
    h=plot3(elec_xyz(a,1),elec_xyz(a,2),elec_xyz(a,3),'k.');
    set(h,'markersize',16);
    if findStrInCell(elec_mono_labels{a},soz_mono),
        set(h,'color','b','markersize',28);
    end
    clickText3D(h,elec_mono_labels{a},0.5);
end
axis square;

fprintf('Bad chans:\n');
for a=1:n_bad_chans,
    fprintf('Bad chans %s\n',bad_chans{a});
    id=find(bad_chans{a}=='-');
    chan1=bad_chans{a}(1:id-1);
    chan2=bad_chans{a}(id+1:end);
    chan1_id=findStrInCell(chan1,elec_mono_labels);
    chan2_id=findStrInCell(chan2,elec_mono_labels);
    h=plot3([elec_xyz(chan1_id,1) elec_xyz(chan2_id,1)], ...
        [elec_xyz(chan1_id,2) elec_xyz(chan2_id,2)], ...
        [elec_xyz(chan1_id,3) elec_xyz(chan2_id,3)],'r-');
    set(h,'linewidth',3);
end

disp('Bad channels are represented with red bars.');
disp('SOZ channels are blue circles.');