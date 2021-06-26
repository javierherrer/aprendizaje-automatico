function [h_total, err_T, err_V] = multiclass_training(lambda, Xcv, ycv, Xtr, ytr)
%MULTICLASS_TRAINING Summary of this function goes here
%   Detailed explanation goes here
[~, col] = size(Xtr);
theta_ini = zeros(col,1);
options.display = 'iter';
%Lo ideal sería con newton normal, pero invertir tarda mucho
options.method = 'pnewton0';

for clase = 1:10 
    ytrain = (ytr == clase);
    h = minFunc(@CosteLogReg, theta_ini, options, ...
        Xtr, ytrain, lambda);
    h_total(:, clase) = h;
end

err_T = tasa_error(h_total, Xtr, ytr);
err_V = tasa_error(h_total, Xcv, ycv);
end

