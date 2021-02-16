function output = createRollingWindow(arr, window_size, stride)
% CREATEROLLINGWINDOW returns successive overlapping windows onto a vector
%   OUTPUT = CREATEROLLINGWINDOW(VECTOR, N) takes a numerical vector VECTOR
%   and a positive integer scalar N. The result OUTPUT is an MxN matrix,
%   where M = length(VECTOR)-N+1. The I'th row of OUTPUT contains
%   VECTOR(I:I+N-1).


% ind = 1:stride:size(arr,2)+1-window_size;
% arr(:, ind: ind + window_size)

% wind_indices = hankel(1:stride:l-window_size+1, l-window_size+1:l);
% wind_indices = wind_indices(1:stride:size(wind_indices,1),:);

otherdims = repmat({':'},1,ndims(arr)-1);

othersizes = size(arr);
othersizes = othersizes(1:ndims(arr)-1);

l = size(arr,ndims(arr));
wind_indices = bsxfun(@plus, 0:window_size-1, (1:stride:l-window_size+1)');
output = reshape(arr(otherdims{:},wind_indices), [othersizes, size(wind_indices,1), size(wind_indices,2)]);

end

% n=size(arr,1);
% k=size(arr,2);
% permute(0:n:(k-1)*n, [3,1,2])
% bsxfun(@plus, permute(0:n:(k-1)*n, [3,1,2]), bsxfun(@plus, 0:window_size-1, (1:window_size)'))