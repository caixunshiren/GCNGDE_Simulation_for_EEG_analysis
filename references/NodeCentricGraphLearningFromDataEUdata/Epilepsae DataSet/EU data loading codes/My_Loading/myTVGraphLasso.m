function [ Theta, A ] = myTVGraphLasso( all_covs, settings )
[T, N, ~] = size(all_covs);
if(settings.mode =='one')
    cvx_begin
        variable Theta(T,N,N)
        variable gama(T)
        loss = 0;
        for counter=1:T % norm(squeeze(all_covs(1,:,:))-squeeze(all_covs(2,:,:)))
            cov_mat = squeeze(all_covs(counter,:,:));
            inTheta = squeeze(Theta(counter,:,:));
            graphLasso_term = -log_det(inTheta) + trace(cov_mat*inTheta);
            regul_term = norm(inTheta, 1 );
            loss = loss + graphLasso_term + settings.landa*regul_term; % (Theta, cov_mat, settings.landa)
        end
        minimize( loss )
        subject to
            norms(reshape(Theta, T, N*N), Inf, 2) <= gama
            norm(gama, 1) <= 0.4
    cvx_end
elseif(settings.mode =='two')
    cvx_begin
        variable Theta(T,N,N)
        variable gama(T)
        loss = 0;
        for counter=1:T % norm(squeeze(all_covs(1,:,:))-squeeze(all_covs(2,:,:)))
            cov_mat = squeeze(all_covs(counter,:,:));
            inTheta = squeeze(Theta(counter,:,:));
            graphLasso_term = -log_det(inTheta) + trace(cov_mat*inTheta);
%             regul_terms = norm(inTheta, 2);
            loss = loss + graphLasso_term; % + settings.landa*sqrt(regul_term); % (Theta, cov_mat, settings.landa)
        end
%         regul_terms = squeeze(norms(Theta, 2, 1));
%         regul_term = sum(pow_abs(regul_terms, 0.5));
        regul_term = sum(sum(squeeze(norms(Theta, 2, 3))));
        loss = loss + settings.landa * regul_term;
        minimize( loss )
    cvx_end
    
end
for counter=1:T
    squeeze(Theta(counter,:,:))
end
A = squeeze(mean(Theta, 1))>=0;

end

