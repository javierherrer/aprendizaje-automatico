function th = descenso_gradiente(alpha, th, X,y)
% Descenso de gradiente con normalización

%Normalizar los atributos
[X, mu, sig] = normalizar(X);

%Resolver la Regresión
% Se realiza una primera iteración fuera del bucle (do-while)
r = X*th-y;            % Residuos
g = X' * r;            % Gradiente
J_aux = (1/2)*sum(r.^2);

th = th - alpha *g;  % (trans. 28 Regresión)
r = X*th-y;
J = (1/2)*sum(r.^2);

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
    
    g = X' * r;             % Gradiente
    th = th - alpha *g;     % (trans. 28 Regresión)
    r = X*th-y;             % Residuos
    J = (1/2)*sum(r.^2);    % Coste
    J_dif = abs(J_aux - J); % |deltaJ|
     plot([iter - 1, iter], [J_aux, J], 'Color', 'blue');
     plot(iter, J, 'b.');
end

%Des-Normalizar los pesos
th = desnormalizar(th, mu, sig);

end

