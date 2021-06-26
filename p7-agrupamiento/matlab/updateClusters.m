function [J, Z] = updateClusters(D,mu)
% D(m,n), m datapoints, n dimensions
% mu(K,n) final centroids
%
% c(m) assignment of each datapoint to a class
[m, ~] = size(D);
[K, ~] = size(mu);

distancias = zeros(m, K);

for k = 1 : K
    %distancias(:, k) = sum( (abs(D - mu(k, :))) .^2, 2);
    distancias(:, k) = mean((abs(D - mu(k, :))) .^2, 2);
end

[J, Z] = min(distancias, [], 2); 
J = sum(J);
end