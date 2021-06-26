%% Based on exercise 2 of Machine Learning Online Class by Andrew Ng 

clear ; close all;

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('exam_data.txt');
y = data(:, 3);
N = length(y);
X = [ones(N,1) data(:, [1, 2])]; 
% Separa un 20% de los datos para test
[Xtest, ytest, Xtr, ytr] = particion(5, 5, X, y);
[Ntr, ~] = size(Xtr);
[Ntest, ~] = size(Xtest);

%% APARTADO 2: Regresión logística básica
theta_ini = zeros(3,1);
lambda = 0; %Regulación
options.display = 'final';
options.method = 'newton';
theta = minFunc(@CosteLogReg, theta_ini, options, ...
    Xtr, ytr, lambda);

% Cálculo de la tasa de errores (trans. 28 Regresión Logística)
% Predicción de la salida
htrain = 1./(1+exp(-(Xtr*theta)));
y_pred = (htrain >= 0.5);
% Tasa de error
error_tr = mean(y_pred ~= ytr)

htest = 1./(1+exp(-(Xtest*theta)));
y_pred = (htest >= 0.5);
% Tasa de error
error_test = mean(y_pred ~= ytest)


% Alumno con 45 puntos en primer examen
X_alumno = [ones(101, 1), 45*ones(101, 1), linspace(0,100,101)'];
h = 1./(1+exp(-(X_alumno*theta)));

figure;
plot(X_alumno(:,3), h, 'b-');
title('Probabilidad de admisión');
ylabel('Probabilidad'); xlabel('Calificación 2º examen');
grid on; hold on; 

% x=80 suspenso, x=81 aprobado