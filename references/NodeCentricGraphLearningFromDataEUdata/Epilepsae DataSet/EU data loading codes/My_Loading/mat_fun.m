function [ out ] = mat_fun( fun, signal )

out = fun(squeeze(signal(1,:,:))');
for i = 2:size(signal,1)
    out = cat(3, out,fun(squeeze(signal(i,:,:))'));
end
out = out/size(signal,1);
out = permute(out, [3,1,2]);

end

