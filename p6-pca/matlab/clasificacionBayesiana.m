function yhat = clasificacionBayesiana( modelo, X)
% Con los modelos entrenados, predice la clase para cada muestra X

pred = [];

for clase = 1:10
    y = gaussLog(modelo{clase}.mu, modelo{clase}.Sigma, X);
    pred = [pred y];
end

[~, yhat] = max(pred, [], 2);