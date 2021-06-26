%% Practica 5 
clear ; close all;
addpath(genpath('../minfunc'));

% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);
N = length(y);
Ntest = length(ytest);

%% APARTADO 3: Bayes ingenuo

%Hacer mapFeature aumentaría mucho el coste
% X     = [ones(N,1) X];
% Xtest = [ones(Ntest,1) Xtest];

% Separa un 20% de los datos para test
k = 5;
%lambda = kfold_cross_validation(k, X, y);
load('lambdaComp.mat');
lambda

[modelo, err_T, err_V] = multiclass_training(lambda, Xtest, ytest, X, y);

figure;
image(modelo{3}.Sigma,'CDataMapping','scaled' );

err_T
err_V

% Predicción de la salida
y_pred = clasificacionBayesiana(modelo, Xtest);

cm = confusionchart(ytest,y_pred);
confMat = cm.NormalizedValues;
prec_global = [];
recall_global = [];
for i =1:size(confMat,2)
    TP = confMat(i,i);
    FP = 0;
    FN = 0;
    for j =1:size(confMat,1)
       if i ~= j
          FP = FP + confMat(j, i);
          FN = FN + confMat(i, j);
       end
    end
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    fprintf('Precision digito %d: %.2f\n', i, precision);
    fprintf('Recall    digito %d: %.2f\n\n', i, recall);
    prec_global = [prec_global precision];
    recall_global = [recall_global recall];
end
precion_total = mean(prec_global)
recall_total = mean(recall_global)
 

verConfusiones(Xtest, ytest, y_pred);
