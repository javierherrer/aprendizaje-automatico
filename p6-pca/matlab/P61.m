%% Practica 6.1: SVD 

clear all
close all

% Leer la imagen
I = imread('dni.jpg');

% Convertirla a blanco y negro
BW = rgb2gray(I);

% Convertir los datos a double
X=im2double(BW);

% graficar la imagen
figure(1);
colormap(gray);
imshow(X);
axis off;
pause

% Aplicar SVD
[U,S,V] = svd(X);

% Graficar las primeras 5 componentes
% for k = 1:5,
%     figure(2);
%     Xhat = U(:, k)*S(k, k)*V(:, k)';
%     imshow(Xhat);
%     colormap(gray);
%     axis off;
%     pause
% end

% Graficar la reconstrucci�n con las primeras 1, 2, 5, 10, 20, y total
% de componentes
% for k = [1 2 5 10 20 rank(X)],
%     figure(3);
%     Xhat = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)';
%     imshow(Xhat);
%     colormap(gray);
%     axis off;
%     pause
% end

% Encontrar el valor de k que mantenga al menos el 90% de la variabilidad
% TODO: cumsum y find
value = 0;
k = 0;
suma = trace(S);
while value < 0.9
    k = k + 1;
    value = value + (S(k,k) / suma);
end
k

% Graficar la reconsrtucci�n con las primeras k componentes
figure(4);
Xhat = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)';
imshow(Xhat);
colormap(gray);
axis off;
pause

% Calcular y mostrar el ahorro en espacio
%   Valores singulares
plot(diag(S));
%   Ratio de compresión
m = 1000
n = 1000
plot((m*n) ./ ((1:1:n)*(m+n+1)));
