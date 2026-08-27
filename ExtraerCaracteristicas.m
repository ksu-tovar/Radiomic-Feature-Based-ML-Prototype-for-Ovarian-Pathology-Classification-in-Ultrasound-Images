function c = ExtraerCaracteristicas(subimg_masked, mask_crop)
% EXTRAER_CARACTERISTICAS - Calcula características básicas de la región
% normalizada
% Ahora usa la imagen preprocesada y la máscara del tumor para limitar cálculos
    subimg_masked = double(subimg_masked);
    mask_crop = logical(mask_crop);

    % Aplicar máscara dentro del recorte
    valores = subimg_masked(mask_crop);  % valores solo dentro del tumor

    % --- Intensidad ---
    intensidad_media = mean(valores);

    % --- Características de primer orden ---
    mediana = median(valores);
    desviacion_std = std(valores);
    varianza = var(valores);
    asimetria = skewness(valores);
    curtosis = kurtosis(valores);

    % --- Percentiles ---
    p10 = prctile(valores,10);
    p90 = prctile(valores,90);
    rango = p90 - p10;

    % --- Textura GLCM ---
    % --- Fondo fuera del tumor a un nivel extra 32 para evitar utilizarlo en el cálculo ---
    subimg_uint = zeros(size(subimg_masked), 'uint8'); % Crea imagen nueva del mismo tamano que la original
    subimg_uint(mask_crop) = uint8(subimg_masked(mask_crop) * 31); % Escala los valores de la imagen en la máscara a 32 niveles
    subimg_uint(~mask_crop) = 32;  % Asigna un nivel extra para el fondo, asegurando que no se incluya en el cálculo de GLCM

    offsets = [0 1; -1 1; -1 0; -1 -1];  % 0°, 45°, 90°, 135°
    glcm = graycomatrix(subimg_uint, 'Offset', offsets, 'Symmetric', true, 'NumLevels', 33); % 32 + 1 fondo
    
    % --- LBP (Local Binary Patterns) ---
    % extractLBPFeatures devuelve un histograma.
    lbp_hist = extractLBPFeatures(subimg_masked, 'NumNeighbors',8,'Radius',1,'Upright',false);
    
    % La entropía dice qué tan compleja es la textura local.
    p = lbp_hist / (sum(lbp_hist) + eps); % Asegurar probabilidad
    idx = p > 0;
    lbp_entropia = -sum(p(idx) .* log2(p(idx))); % Entropía LBP
    lbp_energia = sum(p.^2); % (Energia) Uniformidad LBP

    % --- Propiedades GLCM ---
    % --- Solo considerar los niveles del tumor (1:32) ---
    stats = graycoprops(glcm(1:32,1:32,:), {'Homogeneity','Energy', 'Contrast', 'Correlation'});
    homogeneidad = mean(stats.Homogeneity);
    energia = mean(stats.Energy);
    contraste = mean(stats.Contrast);
    correlacion = mean(stats.Correlation);

    % --- Entropía ---
    entropia = entropy(valores);

    % --- Morfología ---
    % Calculamos propiedades de forma estándar
    props_forma = regionprops(mask_crop, 'Area', 'Perimeter', 'Solidity', 'Eccentricity', 'Extent');
    
    % En caso de que haya multiples regiones desconectadas pequeñas, tomamos la mayor
    % if numel(props_forma) > 1
    %     [~, idx] = max([props_forma.Area]);
    %     props_forma = props_forma(idx);
    % end
    
    area = props_forma.Area;
    perimetro = props_forma.Perimeter;
    if perimetro == 0
       perimetro = 1; 
    end
    
    circularidad = (4 * pi * area) / (perimetro ^ 2);
    compacidad = perimetro^2 / (4 * pi * area);
    
    solidez = props_forma.Solidity;         % Cuán convexo es el tumor (útil para bordes irregulares)
    excentricidad = props_forma.Eccentricity; % Cuán alargado es (0=círculo, 1=línea)
    extension = props_forma.Extent;         % Relación área tumor vs su bounding box

    % --- Vector final ---
    c = [intensidad_media, mediana, desviacion_std, varianza, asimetria, curtosis, ...
         p10, p90, rango, ...
         homogeneidad, energia, contraste, correlacion, ...
         entropia, ...
         lbp_entropia, lbp_energia, ...
         area, circularidad, compacidad, ...
         solidez, excentricidad, extension]; 
end