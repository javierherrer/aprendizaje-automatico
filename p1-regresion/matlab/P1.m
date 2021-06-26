close all;
%% Cargar los datos
datos = load('PisosTrain.txt');
y = datos(:,3);  % Precio en Euros
x1 = datos(:,1); % m^2
x2 = datos(:,2); % Habitaciones
N = length(y);

datos = load('PisosTest.txt');
y_test = datos(:,3);  % Precio en Euros
x1_test = datos(:,1); % m^2
x2_test = datos(:,2); % Habitaciones
N_test = length(y_test);

%% APARTADO 2: Regresión monovariable con ecuación normal
fprintf('\nREGRESIÓN MONOVARIABLE CON ECUACIÓN NORMAL\n');

Xmono = [ones(N,1) x1];
Xmono_test = [ones(N_test,1) x1_test]; % Utilizada para comparar residuos

th2 = Xmono \ y;  % Ecuación Normal (trans. 16 Regresión)

dibujar_regresion_2d(x1, y, th2);

% Función de Coste (trans. 7, 12 Regresión)
error = RMSE(th2, Xmono, y);
error_test = RMSE(th2, Xmono_test, y_test);

fprintf('RMSE train = %.2f\n', error);
fprintf('RMSE test = %.2f\n', error_test);
% RMSE train > RMSE test pq al existir más muestras en train, hay más
%  datos atípicos.

%% APARTADO 3: Regresión multivariable con ecuación normal
fprintf('\nREGRESIÓN MULTIVARIABLE CON ECUACIÓN NORMAL\n');

Xmulti = [ones(N,1) x1 x2];
Xmulti_test = [ones(N_test,1) x1_test x2_test]; % Utilizada para comparar residuos
th3 = Xmulti \ y;  % Ecuación Normal (trans. 16 Regresión)

dibujar_regresion_3d(Xmulti, x1, x2, y, th3);

% Función de Coste (trans. 7, 12 Regresión)
error = RMSE(th3, Xmulti, y);
error_test = RMSE(th3, Xmulti_test, y_test);

fprintf('RMSE train = %.2f\n', error);
fprintf('RMSE test = %.2f\n', error_test);
% RMSE train > RMSE test pq al existir más muestras en train, hay más
%  datos atípicos.

% Comparación con apartado 2
X2 = [1, 100];
y2 = X2 * th2;
fprintf('Ap. 2: x1=100m^2          => y = %.2f\n', y2);

for dorm = 2:5
    X3 = [1, 100, dorm];
    y3 = X3 * th3;
    fprintf('Ap. 3: x1=100m^2 x2=%d hab => y = %.2f\n', dorm, y3);
end

%% APARTADO 4: Regresión monovariable con descenso de gradiente
fprintf('\nREGRESIÓN MONOVARIABLE CON DESCENSO DE GRADIENTE\n');

%Regresión con Normalización en Matlab (trans. 28 Regresión)
alpha = 0.001; % Factor de aprendizaje (trans. 24 Regresión)
th = [5000 1000]';     % Pongo un valor cualquiera de pesos
th4 = descenso_gradiente(alpha, th, Xmono, y);

dibujar_regresion_2d(x1, y, th4);

%% APARTADO 5: Regresión multivariable con descenso de gradiente
fprintf('\nREGRESIÓN MULTIVARIABLE CON DESCENSO DE GRADIENTE\n');

%Regresión con Normalización en Matlab (trans. 28 Regresión)
alpha = 0.003; % Factor de aprendizaje (trans. 24 Regresión)
th = [5000 1000 50000]';     % Pongo un valor cualquiera de pesos
th5 = descenso_gradiente(alpha, th, Xmulti, y);

dibujar_regresion_3d(Xmulti, x1, x2, y, th5);

%% APARTADO 6: Regresión robusta con el coste de Huber
fprintf('\nREGRESIÓN ROBUSTA CON EL COSTE DE HUBER\n');


alpha = 0.003;
delta = 50000;
th = [5000 1000 50000]';     % Pongo un valor cualquiera de pesos
th6 = descenso_gradiente_huber(alpha, delta, th, Xmulti, y);

dibujar_regresion_3d(Xmulti, x1, x2, y, th6);

%Función de Coste (trans. 7, 12 Regresión)
error = RMSE(th6, Xmulti, y);
error_test = RMSE(th6, Xmulti_test, y_test);

fprintf('RMSE train = %.2f\n', error);
fprintf('RMSE test = %.2f\n', error_test); 
