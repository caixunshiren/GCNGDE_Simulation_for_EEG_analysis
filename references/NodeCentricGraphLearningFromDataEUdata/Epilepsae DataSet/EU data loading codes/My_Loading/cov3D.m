function [out] = cov3D(x)
% Based on Matlab's cov, version 5.16.4.10

[m,n,p] = size(x);
out = zeros(m,n,n);
for i = 1:m
    xi = squeeze(x(i,:,:));
    out(i,:,:) = cov(xi');
end
end