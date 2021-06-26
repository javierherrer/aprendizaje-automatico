function [hypothesis, best_model] = kfold_cross_validation(k, X, y)
%KFOLD_CROSS_VALIDATION Elegir el grado de una regresión polinómica o el
%valor del parámetro de regularización.
%   Ten en cuenta que para resolver la regresión es necesario realizar la
%   expansión polinómica de los atributos y su normalización.
best_model = 0;
best_errV = Inf;
errT_aux = 0;
errV_aux = 0;
model_aux = 0;
% col_aux = 0;

figure;
grid on; hold on;
title('Evolución RMSE - Lambda');
ylabel('RMSE'); xlabel('Lambda');

% [~, columnas] = size(X);
% h = [zeros(columnas,1)]; 

% para los distintos valores de los hyper-parámetros
for model = logspace(-10,-5)
    err_T = 0;
    err_V = 0;

    
%     [~, columnas] = size(Xexp);
%     
%     if model == 1
%         h = [zeros(columnas,1)];
%     else
%         h = [h ; zeros(columnas - col_aux,1)];
%     end
    
    %separar N/k ejemplos para validación
    for fold = 1:k
        [Xcv, ycv, Xtr, ytr] = particion(fold, k, X, y);
        % aprender con el resto
        h = regresion(Xtr, ytr, model);
        err_T = err_T + RMSE(h, Xtr, ytr);
        err_V = err_V + RMSE(h, Xcv, ycv);
        fprintf('Grado: %d, Fold: %d\n', model, fold);
    end
    % calcular el error medio de las k veces
    err_T = err_T / k;
    err_V = err_V / k;
    
    fprintf('Error medio lambda %d : %d\n', model, err_T);
    
    if model > 10^-10
        plot([model_aux, model], [errT_aux, err_T], 'Color', 'blue');
        plot([model_aux, model], [errV_aux, err_V], 'Color', 'red');
    end
    plot(model, err_T, 'bx');
    plot(model, err_V, 'ro');
    
%     if  (abs(errT_aux - err_T) < 5) && (err_V < best_errV)
    if (err_V < best_errV)
        % guarda el mejor valor de los hyper-parámetros
        best_model = model;
        best_errV = err_V;
        fprintf('Mejor modelo: %d\n', best_model);
    end
    
    model_aux = model;
    errT_aux = err_T;
    errV_aux = err_V;
%     col_aux = columnas;
end

legend('Jtrain', 'Jcv');

% aprender de nuevo con todos
hypothesis = regresion(X, y, best_model);

