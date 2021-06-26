% load images 
% images size is 20x20. 
clear
close all

% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); 
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);
N = length(y);
Ntest = length(ytest);

nrows=20;
ncols=20;

nimages = size(X,1);

% Show the images
% for I=1:40:nimages, 
%     imshow(reshape(X(I,:),nrows,ncols))
%     pause(0.1)
% end

%% Estandarización
% Calculamos la media
Xmean = mean(X);
Xtestmean = mean(Xtest);

% Estandarizamos
Xnorm = X - Xmean;
Xtestnorm = Xtest - Xtestmean;

%% Perform PCA over all numbers

[u, z] = pca(Xnorm, 'NumComponents', 2);
 
R = corrcoef(z(:,1), z(:,2))

% Muestra las dos componentes principales
figure(100)
clf, hold on
plotwithcolor(z(:,1:2), y);

%% Use classifier from previous labs on the projected space
% Separa un 20% de los datos para test
[Xtest, ytest, Xtr, ytr] = particion(5, 5, z, y);

k = 5;
lambda = kfold_cross_validation(k, Xtr, ytr);
lambda

[modelo, err_T, err_V] = multiclass_training(lambda, Xtest, ytest, Xtr, ytr);

err_T
err_V

% Predicción de la salida
y_pred = clasificacionBayesiana(modelo, Xtest);

figure;
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
 

%verConfusiones(Xtest, ytest, y_pred);



