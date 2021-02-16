function [ out ] = graphLasso(theta, cov_mat, landa)

out = -log(det(theta)) + tr(cov_mat*theta) + landa* norm(theta, 1 );


end

