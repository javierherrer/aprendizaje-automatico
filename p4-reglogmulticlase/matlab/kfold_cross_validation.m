function [best_model] = kfold_cross_validation(k, X, y)
%KFOLD_CROSS_VALIDATION Elegir el grado de una regresión polinómica o el
%valor del parámetro de regularización.
%   Ten en cuenta que para resolver la regresión es necesario realizar la
%   expansión polinómica de los atributos y su normalización.
best_model = 0;
best_errV = Inf;
errT_aux = 0;
errV_aux = 0;
model_aux = 0;

figure;
graph = axes;
set(graph, 'xscale', 'log');
grid on; hold on;
title('Evolución Tasa de error - Lambda');
ylabel('Tasa de error'); xlabel('Lambda');

% para los distintos valores de los hyper-parámetros
for lambda = logspace(-10,0, 20)
    err_T = 0;
    err_V = 0;
    %separar N/k ejemplos para validación
    for fold = 1:k
        [Xcv, ycv, Xtr, ytr] = particion(fold, k, X, y);
        % aprender con el resto
        % LUEGO SE SUMAN TODOS LOS ERRORES
        [~, err_Taux, err_Vaux] = multiclass_training(lambda, Xcv, ycv, Xtr, ytr);
        err_T = err_T + err_Taux;
        err_V = err_V + err_Vaux; 
    end
    % calcular el error medio de las k veces
    err_T = err_T / k;
    err_V = err_V / k;
        
    if lambda > 10^-20
        plot([model_aux, lambda], [errT_aux, err_T], 'Color', 'blue');
        plot([model_aux, lambda], [errV_aux, err_V], 'Color', 'red');
    end
    plot(lambda, err_T, 'bx');
    plot(lambda, err_V, 'ro');
    
%     if  (abs(errT_aux - err_T) < 5) && (err_V < best_errV)
    if (err_V < best_errV)
        % guarda el mejor valor de los hyper-parámetros
        best_model = lambda;
        best_errV = err_V;
%         fprintf('Mejor modelo: %d\n', best_model);
    end
    
    model_aux = lambda;
    errT_aux = err_T;
    errV_aux = err_V;
end

legend('Jtrain', 'Jcv');

% aprender de nuevo con todos
%[h, err_T, err_V] = multiclass_training(best_model, Xcv, ycv, Xtr, ytr);
% hypothesis = minFunc(@CosteLogReg, theta_ini, options, ...
%             X, y, best_model);

