function A_calc = adjacency_calc( signal, load_core)

if(strcmp(load_core.adjacency_func, 'invcov'))
    A_calc = mat_fun(@(x)inv(cov(x)), signal); % cellfun(@(x)cov(x)\eye(s),ydcell,'un',0);
elseif(strcmp(load_core.adjacency_func, 'corr'))
    A_calc = mat_fun(@corr, signal);
elseif(strcmp(load_core.adjacency_func, 'SRPM'))
end
% A_calc = zeros(size(temp));
% A_calc(temp>=load_core.adjacency_threshold) = 1;
end

