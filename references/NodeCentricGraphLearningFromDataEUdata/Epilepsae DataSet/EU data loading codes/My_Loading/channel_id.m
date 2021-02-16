function [ ids ] = channel_id(cell1, cell2)
cell1 = string(cell1);
cell1 = strcat(cell1(:,1),cell1(:,2));
cell2 = string(cell2);
cell2 = strcat(cell2(:,1),cell2(:,2));

% cell1rep = repmat(cell1, 1, size(cell2));
% cell2rep = repmat(cell2', size(cell1), 1);
% find(cell1rep'==cell2rep')
% bsxfun(@eq, cell1, (cell2)')
n_chan=size(cell1);
ids=[];
for a=1:n_chan,
    if(length(find(cell1(a)==cell2))>0)
        ids = [ids a];
    end
end

end

