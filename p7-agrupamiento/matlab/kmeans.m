function [mu, c, J] = kmeans(D,mu0, K)

% D(m,n), m datapoints, n dimensions
% mu0(K,n) K initial centroids
%
% mu(K,n) final centroids
% c(m) assignment of each datapoint to a class

% figure;
% grid on; hold on;
% title('Evolución J - Iteraciones');
% ylabel('J'); xlabel('Iteraciones');
Js = [];
[Js, Z] = updateClusters(D, mu0);
munew = updateCentroids(D, Z, K);
i = 1;
haCambiado = true;
while haCambiado
    [J, Znew] = updateClusters(D, munew);
    munew = updateCentroids(D, Znew, K);
    haCambiado = not( isequal(Z, Znew) );
    Z = Znew;
    
    i = i + 1; 
    Js = [ Js J];
end
i
mu = munew;
c = Z;
iteraciones = linspace(0, i, i);
% plot (iteraciones, Js);