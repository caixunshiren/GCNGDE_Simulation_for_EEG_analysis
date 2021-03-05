load 'C:\Users\caixu\Documents\GitHub\GCNGDE_Simulation_for_EGG_analysis\datasets\sample_patients\pat_FR_620'

%signal = X_train(800:900,:,:);%permute(X_train,[1 3 2]);

signal = permute(X_train,[1 3 2]);
adj = zeros(size(signal, 1), size(signal, 3), size(signal, 3) );

for i = 1:size(X_train, 1)
   aa =(cov(squeeze(signal(i,:,:))));
   if det(aa) ~= 0
        adj(i,:,:) = inv(aa);
   end 
   
end

%A_calc = mat_fun(@(x)inv(cov(x)), signal);

A = squeeze(mean(adj));

clf
figure(1)
heatmap(A);
colormap hot(12)

figure(2)
heatmap(squeeze(adj(93,:,:)));
colormap hot(12)

figure(3)
heatmap(squeeze(cov(squeeze(signal(93,:,:)))));
colormap hot(12)

figure(4)
heatmap(squeeze(cov(squeeze(signal(1,:,:)))));
colormap hot(12)

figure(5)
heatmap(squeeze(adj(1,:,:)));
colormap hot(12)

