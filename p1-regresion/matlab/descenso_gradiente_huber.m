function th = descenso_gradiente_huber(alpha, delta, th, X,y)
% Descenso de gradiente con normalización
% Regresión robusta con coste de huber

%Normalizar los atributos
[X, mu, sig] = normalizar(X);

%Resolver la Regresión
% Se realiza una primera iteración fuera del bucle (do-while)
[J_aux, g] = CosteHuber(th, X, y, delta);

th = th - alpha *g;  % (trans. 28 Regresión)

[J, g] = CosteHuber(th, X, y, delta);

J_dif = abs(J_aux - J);

% Descenso de gradiente (trans. 22)
iter = 1;
figure;
plot(iter, J, 'b.');
title('Descenso de gradiente');
ylabel('Coste'); xlabel('Iteración');
grid on; hold on;

while J_dif > 0.001         % |deltaJ| < 10^-3 (trans. 22 Regresión)
    J_aux = J;
    iter = iter+1;
    
    th = th - alpha *g;                      % (trans. 28 Regresión)
    [J, g] = CosteHuber(th, X, y, delta);    % Coste Huber
    J_dif = abs(J_aux - J);                  % |deltaJ|
    plot([iter - 1, iter], [J_aux, J], 'Color', 'blue');
    plot(iter, J, 'b.');
end

%Des-Normalizar los pesos
th = desnormalizar(th, mu, sig);

fprintf('\nAlpha: %.4f Iteraciones: %d\n', alpha, iter);
end

