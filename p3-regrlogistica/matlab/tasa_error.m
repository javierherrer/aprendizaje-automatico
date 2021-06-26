function error = tasa_error(theta, X, y)
%TASA_ERROR Cálculo de la tasa de errores 
%   (trans. 28 Regresión Logística)

% Predicción de la salida
h = 1./(1+exp(-(X*theta)));
y_pred = (h >= 0.5);
% Tasa de error
error = mean(y_pred ~= y);
end