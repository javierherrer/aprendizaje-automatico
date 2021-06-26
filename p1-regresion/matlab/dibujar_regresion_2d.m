function dibujar_regresion_2d(x,y, th)
%DIBUJAR_REGRESION_2D Summary of this function goes here
%   Detailed explanation goes here
figure;
plot(x, y, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on; 

% Ecuación de predicción obtenida
fprintf('Ecuación de predicción: y = %.2f + %.2fx\n', th(1,1), th(2,1));

Xextr = [1 min(x)  % Predicción para los valores extremos
         1 max(x)];
yextr = Xextr * th;  
plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción
legend('Datos Entrenamiento', 'Prediccion')

end

