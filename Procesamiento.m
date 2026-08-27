function Procesamiento(ruta_imagenes, ruta_masks, train_txt, archivo_csv)
    % PROCESAR_IMAGENES - Lee imágenes y máscaras según el archivo de entrenamiento
    % Extrae características (intensidad, textura, área) y exporta CSV
    % 
    % ruta_imagenes: carpeta de JPGs
    % ruta_masks: carpeta de máscaras PNG
    % archivo_txt: archivo con "imagen clase" por línea
    % archivo_csv: archivo de salida CSV
 

    % --- Leer archivo TXT (train o val) ---
    datos_txt = readlines(train_txt);
    datos = [];

    for i = 1:length(datos_txt)
        linea = strtrim(datos_txt(i));
        if isempty(linea)
             continue; 
        end

    % Separar nombre de la imagen y clase
        partes = strsplit(strtrim(linea));            % limpia espacios extras
        partes = partes(~cellfun('isempty', partes)); % elimina celdas vacías

        % Validar que haya al menos 2 elementos
        if numel(partes) < 2
            warning('⚠️ Línea %d no tiene nombre y clase: "%s"', i, linea);
            continue;
        end

        nombre_img = partes{1};
        clase_original = str2double(partes{2});

        % RE-AGRUPAMIENTO (0-7)
        
        % GRUPO 1: Normal y Quiste Simple (Baja complejidad)
        if clase_original == 5 || clase_original == 4
            clase = 1; 
            
        % GRUPO 3: Maligno (Alta prioridad)
        elseif clase_original == 7
            clase = 3; 
            
        % GRUPO 2: Benignos Complejos (Todo lo demás)
        % Incluye: Chocolate(0), Seroso(1), Teratoma(2), Teca(3), Mucinoso(6)
        else
            clase = 2; 
        end

        % Leer imagen
        ruta_img = fullfile(ruta_imagenes, nombre_img);
        if ~isfile(ruta_img)
            warning('No se encontró %s, se salta.', ruta_img);
            continue;
        end

        img = imread(ruta_img);
        if size(img,3) == 3
            img = rgb2gray(img);
        end

        % Leer máscara correspondiente
        [~, imagen_n] = fileparts(nombre_img);
        nombre_mask = [imagen_n, '.PNG'];
        disp(nombre_mask);
        ruta_mask = fullfile(ruta_masks, nombre_mask);

        %Si existe la mascara sigue, sino se la salta
        if ~isfile(ruta_mask)
            warning('No se encontró la máscara %s, se salta.', ruta_mask);
            continue;
        end

        mask = imread(ruta_mask); 
        % Si la máscara es RGB (3 canales), convertirla a 2D lógica
        if size(mask,3) == 3    
        % cualquier píxel distinto de negro (0,0,0) se vuelve "1"
            mask_bin = any(mask > 0, 3);
        else     
            % si ya es 2D, basta con umbral binario
            mask_bin = mask > 0;
        end

        % Asegurar que sea lógica pura, convierte cada pixel en 1 o 0
        mask_bin = logical(mask_bin);

        % Obtener bounding box
        fprintf('\n📄 Analizando máscara: %s\n', nombre_mask);
        props = regionprops(mask_bin, 'BoundingBox', 'Area');
        if isempty(props)
            fprintf('No se encontró ninguna región en %s\n', nombre_img);
        else
            fprintf('Si, %d regiones encontradas en %s\n', numel(props), nombre_img);
            disp([props.BoundingBox]);
        end
        if isempty(props) 
            continue; 
        end

        % Aplicar máscara para extraer área exacta del tumor ---
        % Subimagen = imagen original * máscara
        % subimg_masked = double(img) .* mask_bin_full; 
        subimg = double(img);

        % Opcional: recortar solo la bounding box para reducir cálculo
        bbox = props(1).BoundingBox;
        subimg = imcrop(subimg, bbox);
        mask_cropped = imcrop(mask_bin, bbox); % máscara recortada

        % --- Filtro de Mediana ---
        I_gray = mat2gray(subimg);
        % Preprocesamiento suave: primero CLAHE para mejorar contraste sin perder textura
        % I_clahe = adapthisteq(I_gray);
        % Luego mediana para eliminar moteado sin destruir patrones
        I_filt = medfilt2(I_gray, [3 3]);

        % --- Extraer características desde la imagen recortada preprocesada usando la máscara ---
        caracteristicas = ExtraerCaracteristicas(I_filt, mask_cropped); 

        % Guardar fila
        datos = [datos; caracteristicas, clase];
    end

    % Exportar CSV
    % cabeceras = {'intensidad_media', 'homogeneidad', 'energia', 'entropia', ...
    %       'area', 'circularidad', 'Clase'};

    cabeceras = {'intensidad_media', 'mediana', 'desviacion_std', 'varianza', 'asimetria', 'curtosis', ...
             'p10', 'p90', 'rango', ...
             'homogeneidad', 'energia', 'contraste', 'correlacion', ...
             'entropia', ...
             'lbp_entropia', 'lbp_energia', ...
             'area', 'circularidad', 'compacidad', ...
             'solidez', 'excentricidad', 'extension', ...
             'Clase'};

    T = array2table(datos, 'VariableNames', cabeceras);
    writetable(T, archivo_csv);
    disp(['Datos exportados a ', archivo_csv]);
end
