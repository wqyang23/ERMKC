function [H_normalized,obj]= mykernelkmeans(K,cluster_count)
K = (K+K')/2;
[H] = eig2(K, cluster_count);
obj = trace(H' * K * H) - trace(K);
H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1,cluster_count);