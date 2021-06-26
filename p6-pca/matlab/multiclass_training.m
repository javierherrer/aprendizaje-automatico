function [modelo, err_T, err_V] = multiclass_training(lambda, Xcv, ycv, Xtr, ytr)
%MULTICLASS_TRAINING

modelo = entrenarGaussianas( Xtr, ytr, 10, 0, lambda);

yhat_tr = clasificacionBayesiana( modelo, Xtr);
yhat_cv = clasificacionBayesiana( modelo, Xcv);

% Tasa de error
err_T = mean(yhat_tr ~= ytr);
err_V = mean(yhat_cv ~= ycv);

end

