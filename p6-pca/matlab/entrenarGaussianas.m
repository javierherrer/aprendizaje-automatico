function modelo = entrenarGaussianas( Xtr, ytr, nc, NaiveBayes, landa )
% Entrena una Gaussana para cada clase y devuelve:
% modelo{i}.N     : Numero de muestras de la clase i
% modelo{i}.mu    : Media de la clase i
% modelo{i}.Sigma : Covarianza de la clase i
% Si NaiveBayes = 1, las matrices de Covarianza serán diagonales
% Se regularizarán las covarianzas mediante: Sigma = Sigma + landa*eye(D)


% Transparencia 17 Modelos Generativos
for j = 1:nc
    muestras = ytr == j;
    
    N_j = sum(muestras);
    modelo{j}.N = N_j;
    
    x_j = Xtr(muestras, :);
    
    media = mean(x_j);
    modelo{j}.mu = media;
    
    covarianza = cov(x_j);
    modelo{j}.Sigma = covarianza;
    
    % Transparencia 23 Modelos Generativos
    if NaiveBayes == 1
        % Bayes ingenuo
        modelo{j}.Sigma = diag(diag(covarianza));
    end
    
    % Regularización en la estimación de las coviarianzas
    modelo{j}.Sigma = modelo{j}.Sigma + landa * eye(length(media));
end