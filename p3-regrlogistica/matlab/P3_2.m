%% Based on exercise 2 of Machine Learning Online Class by Andrew Ng 
clear ; close all;

%% Load and Plot Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('mchip_data.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
N = length(y);
rng(0);
p = randperm(N); %reordena aleatoriamente los datos
X = X(p,:);
y = y(p);

%% APARTADO 3: Regularización

% Expansión de funciones base
Xexp = mapFeature(X(:,1), X(:,2));

% Separa un 20% de los datos para test
[Xtest, ytest, Xtr, ytr] = particion(5, 5, Xexp, y);
[Ntr, ~] = size(Xtr);
[Ntest, ~] = size(Xtest);

[w1, lambda] = kfold_cross_validation(10, Xtr, ytr);

errortest1 = tasa_error(w1, Xtest, ytest);

fprintf('Mejor lambda: %d\n', lambda);
fprintf('Tasa de error: %d\n', errortest1);

% Dibujar la Solucion
plotDecisionBoundary(w1, Xtest, ytest);
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')

% Regresión logística básica
[~, col] = size(Xtr);
theta_ini = zeros(col,1);
lambda = 0;
options.display = 'none';
options.method = 'newton';
w2 = minFunc(@CosteLogReg, theta_ini, options, ...
    Xtr, ytr, lambda);

errortest2 = tasa_error(w2, Xtest, ytest);

fprintf('--------------------\n');
fprintf('Lambda: %d\n', lambda);
fprintf('Tasa de error: %d\n', errortest2);

% Dibujar la Solucion
plotDecisionBoundary(w2, Xtest, ytest);
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')

fprintf('--------------------\n');
fprintf('Diferencia: %.2f\n', (errortest2 - errortest1)*100);

%% APARTADO 4.1: Precisión/Recall

% Predicción de la salida
h = 1./(1+exp(-(Xtest*w1)));
y_pred = (h >= 0.5);
% Tasa de error
pred1 = (y_pred == 1);
pred0 = (y_pred == 0);
real1 = (ytest == 1);
real0 = (ytest == 0);
 
cm = confusionchart(real1,y_pred);
cmvalues = cm.NormalizedValues;
TP = cmvalues(2,2);
TN = cmvalues(1,1);
FP = cmvalues(1,2);
FN = cmvalues(2,1);
fprintf('--------------------\n');
fprintf('Matriz de confusión\n');
fprintf('TP: %d \t FP: %d\n', TP, FP);
fprintf('FN: %d \t TN: %d\n', FN, TN);
precision = TP / (TP + FP);
recall = TP / (TP + FN);

fprintf('--------------------\n');
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);

%% APARTADO 4.2: Precision 95% 

h = 1./(1+exp(-(Xtr*w1)));
prec_rec(h, ytr);
real1 = (ytr == 1);


y_pred = (h >= 0.861);
hold off;
cm = confusionchart(real1,y_pred);
cmvalues = cm.NormalizedValues;
TP = cmvalues(2,2);
FP = cmvalues(1,2);
precision = TP / (TP + FP)


% Si se quiere que el 95% de los chips aceptados sean buenos,
% se debería aumentar el umbral, para aumentar la precisión, aunque
% disminuiría el recobrado




