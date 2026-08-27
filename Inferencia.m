function Inferencia(archivo_modelo)
% INFERENCIA - Predicción con mapeo correcto de variables

    % 1. Cargar modelos y parámetros de normalización
    if exist(archivo_modelo, 'file')
        load(archivo_modelo, 'bestSVMModel', 'bestRFModel', 'bestKNNModel', 'mu', 'sigma');
    else
        error('El archivo de modelo no existe.');
    end

      % Seleccionar imagen
    [archivo, ruta] = uigetfile({'.jpg';'.png'}, 'Selecciona una imagen de ecografía');
    img = imread(fullfile(ruta, archivo));

    if size(img,3) == 3
        img = rgb2gray(img);
    end

    % 3. Dibujo y Máscara
    figure; imshow(img); title('Dibuja el contorno de la lesión');
    h = drawfreehand('Color','r','LineWidth',1.3);
    wait(h); 
    mask = createMask(h);

    % Mostrar la máscara creada
    figure; imshow(mask); title('Máscara generada automáticamente');

    % 4. Recorte (ROI)
    props = regionprops(mask, 'BoundingBox');
    if isempty(props), error('Selección vacía.'); end
    bbox = props(1).BoundingBox;
    
    subimg = imcrop(img, bbox);
    mask_crop = imcrop(mask, bbox);

    % 5. Preprocesamiento
    I_gray = mat2gray(subimg);
    I_filt = medfilt2(I_gray, [3 3]);

    % ---------------------------------------------------------
    % 6. EXTRACCIÓN Y MAPEO DE VARIABLES (CORREGIDO)
    % ---------------------------------------------------------
    
    % A) Extraer TODAS las características (Vector crudo)
    c_raw = ExtraerCaracteristicas(I_filt, mask_crop); 

    % B) Definir las cabeceras TOTALES (El orden debe coincidir con tu función ExtraerCaracteristicas)
    cabeceras_totales = {'intensidad_media', 'mediana', 'desviacion_std', 'varianza', 'asimetria', 'curtosis', ...
             'p10', 'p90', 'rango', ...
             'homogeneidad', 'energia', 'contraste', 'correlacion', ...
             'entropia', ...
             'lbp_entropia', 'lbp_energia', ...
             'area', 'circularidad', 'compacidad', ...
             'solidez', 'excentricidad', 'extension'};

    % C) Crear tabla completa para poder llamar a las columnas por nombre
    T_full = array2table(c_raw, 'VariableNames', cabeceras_totales);

    % D) Definir SOLO las variables usadas en el entrenamiento
    vars_modelo = {'contraste', 'area', 'asimetria', 'curtosis','correlacion', ...
            'homogeneidad', 'solidez', 'excentricidad', 'compacidad', ...
            'circularidad', 'energia', 'lbp_energia', 'lbp_entropia', ...
            'intensidad_media', 'desviacion_std'};

    % E) Filtrar: Seleccionamos de la tabla grande solo lo que necesitamos
    % Esto asegura que el orden sea exactamente el mismo que en el entrenamiento
    T_selected = T_full(:, vars_modelo);

    % F) Convertir a Array para poder operar matemáticamente
    X_vector = table2array(T_selected);

    % ---------------------------------------------------------
    % 7. NORMALIZACIÓN
    % ---------------------------------------------------------
    % Ahora sí, X_vector tiene el mismo tamaño (1x15) que mu y sigma
    X_norm = (X_vector - mu) ./ sigma;

    % ---------------------------------------------------------
    % 8. PREDICCIONES
    % ---------------------------------------------------------
    p_svm = predict(bestSVMModel, X_norm);
    p_rf  = predict(bestRFModel, X_norm);
    p_knn = predict(bestKNNModel, X_norm);

    % ---------------------------------------------------------
    % 9. RESULTADOS EN PANTALLA
    % ---------------------------------------------------------
    clc;
    disp('=============================================');
    disp('       RESULTADOS DEL DIAGNÓSTICO           ');
    disp('=============================================');
    
    imprimir_detalle('SVM (Máquinas de Soporte)', p_svm);
    imprimir_detalle('Random Forest (Bosques)', p_rf);
    imprimir_detalle('KNN (Vecinos Cercanos)', p_knn);
    
    disp('---------------------------------------------');
    
    % % Consenso (Moda)
    % consenso = mode([p_svm, p_rf, p_knn]);
    fprintf('>>> DIAGNÓSTICO SUGERIDO: ');
    imprimir_detalle('', p_knn);
    disp('=============================================');
end

function imprimir_detalle(metodo, clase)
    % Diccionario de riesgos
    switch clase
        case 1
            txt = 'BAJO RIESGO (Benigno)';
        case 2
            txt = 'RIESGO MODERADO (Borderline/Sospechoso)';
        case 3
            txt = 'ALTO RIESGO (Maligno/Cáncer)';
        otherwise
            txt = 'Clase Desconocida';
    end

    if ~isempty(metodo)
        fprintf('%-25s : [%d] %s\n', metodo, clase, txt);
    else
        fprintf('[%d] %s\n', clase, txt);
    end
end