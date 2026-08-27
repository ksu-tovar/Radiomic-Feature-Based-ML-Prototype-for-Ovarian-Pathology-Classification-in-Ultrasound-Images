function Entrenamiento(archivo_csv, archivo_modelo)
% ENTRENAR_MODELO - Versión Limpia y Directa (Gaussian/Poly SVM + RF + KNN)
% NOTA: el CSV de entrada está aumentado con SMOTE

    % 1. Preparación de datos
    T = readtable(archivo_csv);
    vars = {'contraste', 'area', 'asimetria', 'curtosis','correlacion', ...
            'homogeneidad', 'solidez', 'excentricidad', 'compacidad', ...
            'circularidad', 'energia', 'lbp_energia', 'lbp_entropia', ...
            'intensidad_media', 'desviacion_std'};

    X = T{:, vars}; 
    y = T{:, end};     

    [X, mu, sigma] = zscore(X);

    % Configuración Grid Search

    rng(42, 'twister');
    cvp = cvpartition(y, 'KFold', 5, 'Stratify', true);

    opts = struct('Optimizer', 'gridsearch', ...
                  'NumGridDivisions', 5, ... 
                  'CVPartition', cvp, ...
                  'Verbose', 0, ...
                  'ShowPlots', false);

    %% --- 1. SVM (RBF vs Polynomial) ---
    disp('--- Iniciando optimización SVM ---');
    
    kernels = {'rbf', 'polynomial'};
    bestSVMModel = [];
    bestSVMLoss = Inf; 
    bestKernelName = '';
    bestSVMParams = [];

    for k = 1:length(kernels)
        kName = kernels{k};
        fprintf('Probando Kernel: %s... ', kName);
        
        if strcmp(kName, 'rbf')
            params = {'BoxConstraint', 'KernelScale'};
            t = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
        else
            params = {'BoxConstraint', 'PolynomialOrder'};
            t = templateSVM('KernelFunction', 'polynomial', 'Standardize', false);
        end
        
        % Entrenamiento con Grid Search
        mdl = fitcecoc(X, y, ...
            'Learners', t, ...
            'OptimizeHyperparameters', params, ...
            'HyperparameterOptimizationOptions', opts);
        
        % Buscamos el mínimo en la columna 'Objective' (Error de validación)
        R = mdl.HyperparameterOptimizationResults;
        [currentLoss, idxMin] = min(R.Objective);
        
        fprintf('Error CV: %.2f%%\n', currentLoss*100);
        
        if currentLoss < bestSVMLoss
            bestSVMLoss = currentLoss;
            bestSVMModel = mdl;
            bestKernelName = kName;
            bestSVMParams = R(idxMin, :);
        end
    end

    disp(['>>> Mejor SVM: ', bestKernelName, ' (Acc: ', num2str((1-bestSVMLoss)*100), '%)']);

    %% --- 2. RANDOM FOREST ---
    disp('--- Iniciando optimización Random Forest ---');

    t_rf = templateTree('Reproducible', true); 
    
    bestRFModel = fitcensemble(X, y, 'Method', 'Bag', ...
        'NumLearningCycles', 300, ... 
        'Learners', t_rf, ...
        'OptimizeHyperparameters', {'MinLeafSize', 'NumVariablesToSample'}, ...
        'HyperparameterOptimizationOptions', opts);

    % Acceso directo a la tabla de resultados
    accRF = 1 - min(bestRFModel.HyperparameterOptimizationResults.Objective);
    
    R_rf = bestRFModel.HyperparameterOptimizationResults;
    [~, idxRF] = min(R_rf.Objective);
    bestRFParams = R_rf(idxRF, :);    
    
    disp(['>>> MEJOR RF Acc: ', num2str(accRF*100), '%']);

    %% --- 3. KNN (K-Nearest Neighbors) ---
    disp('--- Iniciando optimización KNN ---');

    params_knn = hyperparameters('fitcknn', X, y);
    for i = 1:numel(params_knn)
        params_knn(i).Optimize = ismember(params_knn(i).Name, {'NumNeighbors','Distance'});
        if strcmp(params_knn(i).Name, 'NumNeighbors')
            params_knn(i).Range = [3 25];
        end
    end

    bestKNNModel = fitcknn(X, y, ...
        'OptimizeHyperparameters', params_knn, ...
        'HyperparameterOptimizationOptions', opts);

    accKNN = 1 - min(bestKNNModel.HyperparameterOptimizationResults.Objective);
    
    disp(['>>> MEJOR KNN Acc: ', num2str(accKNN*100), '%']);

    %% --- Resumen de configuración para el paper ---
    fprintf('\n========== CONFIGURACION PARA EL PAPER ==========\n');
    fprintf('Datos: %d muestras, %d caracteristicas\n', size(X,1), size(X,2));
    fprintf('Clases (n por clase): '); fprintf('%d ', histcounts(y, numel(unique(y)))); fprintf('\n');
    fprintf('Normalizacion: z-score con mu/sigma del entrenamiento\n');
    fprintf('Seleccion: grid search, 5-fold estratificado, %d divisiones, semilla 42\n\n', ...
        opts.NumGridDivisions);

    fprintf('--- SVM (ECOC uno-contra-uno) ---\n');
    fprintf('Kernel seleccionado: %s\n', bestKernelName);
    disp(bestSVMParams);

    fprintf('--- Random Forest (bagging) ---\n');
    fprintf('NumLearningCycles (arboles): %d\n', bestRFModel.NumTrained);
    disp(bestRFParams);

    fprintf('--- KNN ---\n');
    fprintf('NumNeighbors: %d\n', bestKNNModel.NumNeighbors);
    fprintf('Distance: %s\n', char(bestKNNModel.Distance));
    fprintf('=================================================\n\n');

    % --- Guardar ---
    % Se agrega 'bestKNNModel' a la lista de variables guardadas
    save(archivo_modelo, 'bestSVMModel','bestRFModel', 'bestKNNModel', 'mu', 'sigma');
    disp(['Modelos guardados en ', archivo_modelo]);
end