% MAIN.M
%Se va a encargar de ejecutar todo

clear all;
clc; 
close all;

% Rutas de trabajo     %Nombre de las carpetas donde tengo los datos
ruta_imagenes = 'Imagenes';      
ruta_masks = 'Anotaciones';
train_txt = 'txt files\train_cls.txt';   %Porque tiene la clasificacion
val_txt = 'txt files\val_cls.txt';
archivo_modelo = 'modeloYparametros.mat';
csv_train = "Dataset_train_balanceado.csv";
%csv_train = "Dataset_train.csv";
csv_val   = "Dataset_val.csv";

% %1. Procesar TRAIN
% disp("Procesando TRAIN...");
% Procesamiento(ruta_imagenes, ruta_masks, train_txt, csv_train);
% 
% %2. Procesar VALIDATION
% disp("Procesando VALIDATION...");
% Procesamiento(ruta_imagenes, ruta_masks, val_txt, csv_val);

%3. Entrenar modelo con TRAIN
%disp("Entrenando modelo...");
%Entrenamiento(csv_train, archivo_modelo);

% %4. Validar con VAL
disp("Validando modelo...");
Validacion(csv_val, archivo_modelo);

%% 4️ Inferencia manual (opcional)
% resp = input('¿Deseas probar una imagen nueva? (s/n): ', 's');
% if lower(resp) == 's'
%    Inferencia(archivo_modelo);
% end

% disp('Flujo completo finalizado');