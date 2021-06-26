function th = regresion(X, y, lambda)
%REGRESION Funcion de regresion
[X, mu, sig] = normalizar(X);

% Solucion con ecuación normal
th = X \ y;
% [~, D] = size(X);
% H = X'*X + lambda*diag([0 ones(1,D-1)]);
% th = H \ (X'*y);

% Solucion con descenso de gradiente HUBER
% utilizar las thetas calculadas en la interación anterior para facilitar
%  el descenso de gradiente
% th = descenso_gradiente_huber(0.001, 1000, th, X, y);

th = desnormalizar(th, mu, sig);
end