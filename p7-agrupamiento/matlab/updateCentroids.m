function munew = updateCentroids(D,c, K)
% D((m,n), m datapoints, n dimensions
% c(m) assignment of each datapoint to a class
%
% munew(K,n) new centroids
% Transparencia 5 Clustering

for k = 1 : K
    munew(k, :) = mean(D(c == k, :));
end
