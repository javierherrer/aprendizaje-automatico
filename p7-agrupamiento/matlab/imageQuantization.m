close all;
figure(1)
im = imread('imAgenes/eina3.png');
imshow(im)

%% datos
D = double(reshape(im,size(im,1)*size(im,2),3));

%% dimensiones
m = size(D,1)
n = size(D,2)

%% Kmeans 
K = 15;
%K = 8;

%% Inicializar centroides
% p = randperm(length(D));
% Dperm = D(p,:);
% mu0 = Dperm(1:K,:);


% Js = [];
% Rcomp = Inf;
% Rcomps = [];
% Ks = linspace(4,100,20);
% for K = Ks
    % Buscar K puntos equidistantes en matriz tridimensional
    mu0 = curvspace(D, K);

    %bucle kmeans
    [mu, c, J] = kmeans(D, mu0, K);
    Rcomp = (m*n)/((K*n)+m)
%     K
    J
%     
%     Js = [ Js J];
%     Rcomps = [Rcomps Rcomp];
% end

% figure(5);
% grid on; hold on;
% title('Evolución J - K');
% ylabel('J (función de distorsión)'); xlabel('K (número de clusters)');
% plot (Ks, Js);
% 
% figure(6);
% grid on; hold on;
% title('Evolución Rat. compresión - K');
% ylabel('Ratio de compresión'); xlabel('K (número de clusters)');
% plot (Ks, Rcomps);

%% reconstruir imagen
qIM=zeros(length(c),3);
for h=1:K,
    ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure(2)
imshow(uint8(qIM));
