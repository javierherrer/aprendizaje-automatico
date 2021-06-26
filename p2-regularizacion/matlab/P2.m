close all;
%% Cargar los datos
datos = load('CochesTrain.txt');
ydatos = datos(:, 1);   % Precio en Euros
Xdatos = datos(:, 2:4); % Años, Km, CV
x1dibu = linspace(min(Xdatos(:,1)), max(Xdatos(:,1)), 100)'; %para dibujar

datos2 = load('CochesTest.txt');
ytest = datos2(:,1);  % Precio en Euros
Xtest = datos2(:,2:4); % Años, Km, CV
Ntest = length(ytest);


%% APARTADO 2

disp('********************APARTADO 2*******************');
n = 10;

%Xexp = expandir(Xdatos, [10 10 10]);
%[w, grado] = kfold_cross_validation(10, Xexp, ydatos);

%grado

Xexp = expandir(Xdatos, [5 6 6]);
Xtest = expandir(Xtest, [5 6 6]);

% Xexp = expandir(Xdatos, [10 10 10]);
% Xtest = expandir(Xtest, [10 10 10]);
w = regresion(Xexp, ydatos, (2.33e-07));
 
RMSEtr = RMSE(w, Xexp, ydatos);
RMSEcv = RMSE(w, Xtest, ytest);
 
fprintf('RMSE %.2f', RMSEcv);