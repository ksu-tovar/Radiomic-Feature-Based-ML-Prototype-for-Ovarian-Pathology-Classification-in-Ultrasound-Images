function Validacion(archivo_csv, archivo_modelo)
    % VALIDAR_MODELO - Evalúa el modelo con un conjunto de validación y muestra métricas

    % Cargar modelo y parametros de normalizacion
    load(archivo_modelo, 'bestSVMModel','bestRFModel', 'bestKNNModel', 'mu', 'sigma');

    % Leer datos
    T = readtable(archivo_csv);
    vars = {'contraste', 'area', 'asimetria', 'curtosis','correlacion', ...
            'homogeneidad', 'solidez', 'excentricidad', 'compacidad', ...
            'circularidad', 'energia', 'lbp_energia', 'lbp_entropia', ...
            'intensidad_media', 'desviacion_std'};

    X_raw = T{:, vars};
    y = T{:, end};

    % --- Normalizar
    % Formula: (Val - Mean_Train) / Sigma_Train
    X = (X_raw - mu) ./ sigma;

    % Predicciones
    y_pred_SVM = predict(bestSVMModel, X);
    y_pred_RF  = predict(bestRFModel, X);
    y_pred_KNN = predict(bestKNNModel, X);

    % Calcular métricas
    exactitud_SVM = sum(y_pred_SVM == y) / numel(y);
    exactitud_RF  = sum(y_pred_RF  == y) / numel(y);
    exactitud_KNN = sum(y_pred_KNN == y) / numel(y);

    disp(['Exactitud global SVM: ', num2str(exactitud_SVM * 100), '%']);
    disp(['Exactitud global RF:  ', num2str(exactitud_RF  * 100), '%']);
    disp(['Exactitud global KNN: ', num2str(exactitud_KNN * 100), '%']);

    disp("===== Metrics SVM =====");
    calcularMetricas(y, y_pred_SVM);
    disp("===== Metrics RF =====");
    calcularMetricas(y, y_pred_RF);
    disp("===== Metrics KNN =====");
    calcularMetricas(y, y_pred_KNN);

    classNames = {'Low Risk', 'Borderline', 'Malignant'};
    y_cat     = categorical(y,          [1 2 3], classNames);
    pred_SVM  = categorical(y_pred_SVM, [1 2 3], classNames);
    pred_RF   = categorical(y_pred_RF,  [1 2 3], classNames);
    pred_KNN  = categorical(y_pred_KNN, [1 2 3], classNames);
    
    figure('Position',[100 100 650 280]);
    cm = confusionchart(y_cat, pred_SVM, 'FontSize', 14);
    cm.Title = 'Confusion Matrix - SVM';
    
    figure('Position',[100 100 650 280]);
    cm = confusionchart(y_cat, pred_RF, 'FontSize', 14);
    cm.Title = 'Confusion Matrix - Random Forest';
    
    figure('Position',[100 100 650 280]);
    cm = confusionchart(y_cat, pred_KNN, 'FontSize', 14);
    cm.Title = 'Confusion Matrix - KNN';
end

% FUNCIÓN PARA CALCULAR MÉTRICAS
% -------------------------------------------------------
function calcularMetricas(y_true, y_pred)

    classes = unique(y_true);

    % Promedio macro (cada clase aporta por igual)
    total_precision = 0;
    total_sensibilidad = 0;
    total_especificidad = 0;
    total_f1 = 0;

    for c = classes'
        % Verdaderos Positivos
        VP = sum((y_pred == c) & (y_true == c));
        % Falsos Positivos
        FP = sum((y_pred == c) & (y_true ~= c));
        % Falsos Negativos
        FN = sum((y_pred ~= c) & (y_true == c));
        % Verdaderos Negativos
        VN = sum((y_pred ~= c) & (y_true ~= c));

        % --- Métricas por clase ---
        % VP+FP puede ser 0 si el modelo nunca predice esta clase
        if VP + FP == 0
            warning('El modelo nunca predijo la clase %d.', c);
            precision = 0;
        else
            precision = VP / (VP + FP);
        end
        % VP+FN y VN+FP nunca son 0: no hacen falta guardas
        sensibilidad = VP / (VP + FN);
        especificidad = VN / (VN + FP);
        if precision + sensibilidad == 0
            f1 = 0;
        else
            f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad);
        end

        % Acumular
        total_precision = total_precision + precision;
        total_sensibilidad = total_sensibilidad + sensibilidad;
        total_especificidad = total_especificidad + especificidad;
        total_f1 = total_f1 + f1;
    end

    % Promedio macro final
    n = numel(classes);
    fprintf('Precisión:     %.2f%%\n', 100 * total_precision / n);
    fprintf('Sensibilidad:  %.2f%%\n', 100 * total_sensibilidad / n);
    fprintf('Especificidad: %.2f%%\n', 100 * total_especificidad / n);
    fprintf('F1-Score:      %.2f%%\n\n', 100 * total_f1 / n);
end