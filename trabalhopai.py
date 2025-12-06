import sys
import platform
import io
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PySide6.QtCore import Qt
from PySide6.QtGui import (QAction,QFont,QPixmap,QImage,QPainter,QColor,QTextCursor)
from PySide6.QtWidgets import (QApplication,QMainWindow, QWidget, QTabWidget,QVBoxLayout, QHBoxLayout, QStatusBar, QFileDialog,QGraphicsView, QGraphicsScene, 
    QGraphicsPixmapItem,QLabel,QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,QSplitter, QTextEdit, QMessageBox,QMenuBar,QMenu, QDialog)
import numpy as np
import nibabel as nib
from PIL import Image, ImageQt
import pandas as pd
import joblib
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.filters import threshold_otsu, gaussian
from skimage.exposure import rescale_intensity, equalize_adapthist
from sklearn.cluster import KMeans


import random
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error


import itertools
import math
from PySide6.QtWidgets import QScrollArea



ROOT = Path('.')
OUTPUT_DIR = ROOT / 'out'
MODELS_DIR = ROOT / 'models'
DB_ROOT = ROOT / 'database'
AXL_DIR = DB_ROOT / 'axl'
OASIS_CSV_PATH = DB_ROOT / 'oasis_longitudinal_demographic.csv'

MODEL_LR_DEMENCIA_PATH = MODELS_DIR / 'modelo_lr.pkl'
MODEL_LR_DEMENCIA_THRESHOLD_PATH = MODELS_DIR / 'optimal_threshold.pkl'
MODEL_XGB_IDADE_PATH = MODELS_DIR / 'modelo_xgb.pkl' 

MODEL_DL_DEMENCIA_PATH = MODELS_DIR / 'efficientnet_classification.keras'
MODEL_DL_IDADE_PATH = MODELS_DIR / 'efficientnet_age_regression.keras'
MODEL_DL_AGE_STATS_PATH = MODELS_DIR / 'age_min_max.npy'

CM_DL_DEMENCIA_PATH = OUTPUT_DIR / 'dl_confusion_matrix.png'
CURVES_DL_DEMENCIA_PATH = OUTPUT_DIR / 'dl_classification_curves.png'
SCATTER_DL_IDADE_PATH = OUTPUT_DIR / 'dl_age_scatter_plot.png'
CURVES_DL_IDADE_PATH = OUTPUT_DIR / 'dl_regression_curves.png'

CM_SHALLOW_PATH = OUTPUT_DIR / 'sl_cm_linear.png'
SCATTER_SHALLOW_PATH = OUTPUT_DIR / 'sl_scatter_xgboost.png'
SCATTER_FEATURES_PATH = OUTPUT_DIR / 'features_scatter_pairs.png'


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 20 
FINE_TUNE_EPOCHS = 15 
LEARNING_RATE = 0.0001
LEARNING_RATE_FT = LEARNING_RATE / 10


CLASS_MAP = {'NonDemented': 0, 'Demented': 1}
CLASS_NAMES = list(CLASS_MAP.keys())


N_CLUSTERS = 4
MIN_AREA_VENTRICLE = 100
MAX_AREA_VENTRICLE = 15000
MIN_AREA_BRAIN = 5000
CENTER_TOLERANCE_RATIO = 0.2

def adicionar_paths(df, target_col='y'):
    df = df.copy()
    df['path'] = df['MRI ID'].apply(lambda mid: str(AXL_DIR / f"{mid}_axl.nii.gz"))
    return df[['path', target_col]]

def montar_matriz_img(path_tensor):
        path_str = path_tensor.numpy().decode('utf-8')
        img = nib.load(path_str).get_fdata()
        img_slice = img
        img_slice = np.rot90(img_slice)
        return img_slice.astype(np.float32)

def realizar_preprocess_input(path, label):
    image_data = tf.py_function(montar_matriz_img, [path], tf.float32)
    image_data.set_shape([None, None])
    
    image_data = image_data / (tf.reduce_max(image_data) + 1e-6)
    
    
    image_data = tf.expand_dims(image_data, axis=-1)
    image_data = tf.image.resize(image_data, [IMG_SIZE, IMG_SIZE])
    image_data = tf.image.grayscale_to_rgb(image_data)

    
    image_data = tf.keras.applications.efficientnet.preprocess_input(image_data)
    image_data.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return image_data, label

def criar_dataset(df, batch_size=BATCH_SIZE, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((df['path'], df['y']))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=SEED)
    dataset = dataset.map(realizar_preprocess_input, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def buildar_modelo(img_size=IMG_SIZE):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )

    base_model.trainable = False

    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )

    return model, base_model

def buildar_modelo_regressao(img_size=IMG_SIZE):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    inputs= base_model.input
    x= base_model.output
    x= GlobalAveragePooling2D()(x)
    x= Dropout(0.3)(x)
    outputs= Dense(1, activation='sigmoid')(x)
    model= Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mean_absolute_error', 
        metrics=['mae', 'mse'] )
    return model, base_model

def plot_learning_curves_dl(history, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    metric_key= 'accuracy' if 'accuracy' in history.history else 'mae'
    val_metric_key= 'val_' + metric_key
    ax[0].plot(history.history[metric_key], label=f'{metric_key.capitalize()} (treino)')
    ax[0].plot(history.history[val_metric_key], label=f'{metric_key.capitalize()} (validação)')
    ax[0].set_title(f'Curva de {metric_key.capitalize()}')
    ax[0].set_xlabel('Epoca')
    ax[0].set_ylabel(metric_key.capitalize())
    ax[0].legend()
    
    ax[1].plot(history.history['loss'], label='Perda (treino)')
    ax[1].plot(history.history['val_loss'], label='Perda (validação)')
    ax[1].set_title('Curva de perda')
    ax[1].set_xlabel('Epoca')
    ax[1].set_ylabel('Perda')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Curvas de aprendizado salvas em: {save_path}")

def carregar_df_csv():
    try:
        df_oasis = pd.read_csv(OASIS_CSV_PATH, sep=';', decimal=',')
    except FileNotFoundError:
        print(f"FATAL ERROR: Arquivo CSV não encontrado em {OASIS_CSV_PATH}")
        return None

    
    df_oasis['Group'] = df_oasis['Group'].astype(str).str.strip()
    df_oasis['MRI ID'] = df_oasis['MRI ID'].astype(str).str.strip()
    df_oasis['Subject ID'] = df_oasis['Subject ID'].astype(str).str.strip()
    df_oasis['CDR'] = pd.to_numeric(df_oasis['CDR'], errors='coerce')

    
    df_oasis = df_oasis[df_oasis['Group'].isin(['Nondemented', 'Demented', 'Converted'])].copy()

    
    def map_class_dl(row):
        g = str(row['Group']).strip()
        cdr = row['CDR']

        if g == 'Converted':
            
            if pd.isna(cdr) or cdr == 0:
                return 'NonDemented'
            else:
                return 'Demented'
        elif g == 'Nondemented':
            return 'NonDemented'
        else:
            
            return 'Demented'

    df_oasis['Group_DL'] = df_oasis.apply(map_class_dl, axis=1)

    print(f"Verificando arquivos em: {AXL_DIR}")
    all_files = set(f.name for f in AXL_DIR.glob("*.nii.gz"))
    df_oasis['exists'] = df_oasis['MRI ID'].apply(
        lambda mid: f"{mid}_axl.nii.gz" in all_files
    )
    df_oasis = df_oasis[df_oasis['exists']].copy()

    if df_oasis.empty:
        print(f"ERRO: Nenhum arquivo .nii.gz encontrado em {AXL_DIR} que corresponda ao arquivo CSV")
        return None

    
    df_oasis['y'] = df_oasis['Group_DL'].map(CLASS_MAP)

    print("Distribuição de classes (DL) depois do filtro:")
    print(df_oasis['Group_DL'].value_counts())

    
    
    patient_labels = df_oasis.groupby('Subject ID')['y'].max()
    patient_ids = patient_labels.index.to_list()
    labels = patient_labels.values

    
    trainval_patients, test_patients, trainval_labels, _ = train_test_split(
        patient_ids,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    
    train_patients, val_patients, _, _ = train_test_split(
        trainval_patients,
        trainval_labels,
        test_size=0.2,
        stratify=trainval_labels,
        random_state=SEED
    )

    
    train_df = df_oasis[df_oasis['Subject ID'].isin(train_patients)].copy()
    val_df = df_oasis[df_oasis['Subject ID'].isin(val_patients)].copy()
    test_df = df_oasis[df_oasis['Subject ID'].isin(test_patients)].copy()

    return train_df, val_df, test_df

def _load_nifti(path_tensor):
        path_str = path_tensor.numpy().decode('utf-8')
        img = nib.load(path_str).get_fdata().astype(np.float32)  

        if img.ndim != 2:
            raise ValueError(f"Shape inesperado (esperado 2D): {img.shape}")

        
        img_rgb = np.stack([img, img, img], axis=-1)

        return img_rgb.astype(np.float32)

def load_and_preprocess_nifti(path, label):
    
    image_data = tf.py_function(_load_nifti, [path], tf.float32)
    image_data.set_shape([None, None, 3])

    
    max_val = tf.reduce_max(image_data)
    image_data = image_data / (max_val + 1e-6)

    
    image_data = image_data * 255.0

    
    image_data = tf.image.resize(image_data, [IMG_SIZE, IMG_SIZE])

    
    image_data = tf.keras.applications.efficientnet.preprocess_input(image_data)

    image_data.set_shape([IMG_SIZE, IMG_SIZE, 3])

    return image_data, label

def build_split_df(split_name):
    split_dir = os.path.join(DB_ROOT, split_name)
    csv_path  = os.path.join(split_dir, "features_full.csv")
    img_dir   = os.path.join(split_dir, "imgs")

    df = pd.read_csv(csv_path, sep=';', decimal=',')

    
    df = df[df["Group"].isin(["Demented", "NonDemented"])].copy()

    
    label_map = {"NonDemented": 0, "Demented": 1}
    df["y"] = df["Group"].map(label_map)

    
    df["path"] = df["MRI ID"].apply(
    lambda mid: os.path.join(img_dir, f"{mid}_axl.nii.gz")
    )
    
    return df[["path", "y", "Age", "MRI ID"]]

def make_dataset(df_paths, training=False):
    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
    ])
    x = df_paths["path"].values
    y = df_paths["y"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((x, y))

    
    if training:
        ds = ds.shuffle(buffer_size=len(df_paths), reshuffle_each_iteration=True, seed=42)

    ds = ds.map(
        load_and_preprocess_nifti,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if training:
        ds = ds.map(
            lambda img, label: (data_augmentation(img, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def iniciar_treino_classificacao():
    print("-- DEBUG -- iniciando treinamento: classificação (Classificação de Grupo - DL)")
    
    train_df = build_split_df("treino")
    val_df   = build_split_df("validacao")
    test_df  = build_split_df("teste")
    
    train_ds = make_dataset(train_df, training=True)
    val_ds   = make_dataset(val_df, training=False)
    test_ds  = make_dataset(test_df, training=False)
    
    model, base_model = buildar_modelo()
    
    print("-- DEBUG -- iniciando o treinamento..")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    print("-- DEBUG -- realizando o fine-tuning...")
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * 2 / 3)

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    
    LEARNING_RATE_FT = LEARNING_RATE / 10
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FT),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalsePositives(name='fp')
        ]
    )
    
    TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS
    history_fine_tune = model.fit(
        train_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    print("Treinamento de classificação DL finalizado com sucesso!")
    
    
    history.history['accuracy'].extend(history_fine_tune.history['accuracy'])
    history.history['val_accuracy'].extend(history_fine_tune.history['val_accuracy'])
    history.history['loss'].extend(history_fine_tune.history['loss'])
    history.history['val_loss'].extend(history_fine_tune.history['val_loss'])
    plot_learning_curves_dl(history, CURVES_DL_DEMENCIA_PATH)
    print("Salvando modelo de classificação..")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_DL_DEMENCIA_PATH)
    print("Realizando avaliação no conjunto de teste...")
    results = model.evaluate(test_ds)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    print("Relatório de classificação:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots()
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Matriz de Confusão (EfficientNet)")
    plt.savefig(CM_DL_DEMENCIA_PATH)
    plt.close(fig)
    print(f"Matriz de confusão salva em: {CM_DL_DEMENCIA_PATH}")
    
    return True, "Treinamento de classificação DL finalizado!"

def iniciar_treino_regressao():
    print("-- DEBUG -- Iniciando treinamento: regressão (idade - DL)")
    data_split = carregar_df_csv()
    if data_split is None:
        return False, "Falha ao carregar dados para regressão"
    
    train_df, val_df, test_df = data_split
    
    age_min = train_df['Age'].min()
    age_max = train_df['Age'].max()
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(MODEL_DL_AGE_STATS_PATH, np.array([age_min, age_max]))
    print(f"idade min/max (treino): {age_min:.1f} / {age_max:.1f} Salvo em {MODEL_DL_AGE_STATS_PATH}")
    train_df['y'] = (train_df['Age'] - age_min) / (age_max - age_min)
    val_df['y'] = (val_df['Age'] - age_min) / (age_max - age_min)
    test_df['y'] = (test_df['Age'] - age_min) / (age_max - age_min)
    train_paths = adicionar_paths(train_df, 'y')
    val_paths   = adicionar_paths(val_df, 'y')
    test_paths  = adicionar_paths(test_df, 'y')
    train_ds_reg = criar_dataset(train_paths, is_training=True)
    val_ds_reg = criar_dataset(val_paths, is_training=False)
    test_ds_reg = criar_dataset(test_paths, is_training=False)
    reg_model, reg_base_model = buildar_modelo_regressao()

    print("-- DEBUG -- iniciando treinamento do regressor")
    reg_history = reg_model.fit(
        train_ds_reg,
        epochs=EPOCHS,
        validation_data=val_ds_reg,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )

    print("-- DEBUG -- realizando o fine-tuning")
    reg_base_model.trainable = True
    reg_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FT),
        loss='mean_absolute_error',
        metrics=['mae', 'mse']
    )
    
    TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS
    reg_history_ft = reg_model.fit(
        train_ds_reg,
        epochs=TOTAL_EPOCHS,
        initial_epoch=reg_history.epoch[-1] if reg_history.epoch else 0,
        validation_data=val_ds_reg,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
    )
    
    print("Treinamento de regressão DL finlizado com sucesso!")
    reg_history.history['mae'].extend(reg_history_ft.history['mae'])
    reg_history.history['val_mae'].extend(reg_history_ft.history['val_mae'])
    reg_history.history['loss'].extend(reg_history_ft.history['loss'])
    reg_history.history['val_loss'].extend(reg_history_ft.history['val_loss'])
    
    plot_learning_curves_dl(reg_history, CURVES_DL_IDADE_PATH)
    
    print("Salvando modelo de regressao..")
    reg_model.save(MODEL_DL_IDADE_PATH)
    
    print("Avaliando regressor(idade) no conjunto de teste...")
    y_pred_norm = reg_model.predict(test_ds_reg).flatten()
    
    
    y_true_real = test_df['Age'].values
    
    
    y_pred_real = (y_pred_norm * (age_max - age_min)) + age_min

    mae_real = mean_absolute_error(y_true_real, y_pred_real)
    print(f"Erro médio absoluto (MAE) do teste: {mae_real:.2f}  anos")

    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_real, y_pred_real, alpha=0.6)
    plt.title(f'Idade real vs. prevista (EfficientNet)\n = {mae_real:.2f} anos')
    plt.xlabel('Idade real (em anos)')
    plt.ylabel('Idade prevista (em anos')
    lims = [min(age_min, np.min(y_true_real)), max(age_max, np.max(y_true_real))]
    plt.plot(lims, lims, 'r--', label='Previsão perfeita (y=x)')
    plt.legend()
    plt.grid(True)
    plt.savefig(SCATTER_DL_IDADE_PATH)
    plt.close()
    print(f"Gráfico de dispersão da regressão salvo em: {SCATTER_DL_IDADE_PATH}")
    
    return True, "Treinamento de regressão DL finalizado!"


def iniciar_treinamento_raso():
    print("-- DEBUG -- Iniciando treinamento: Classificador Raso (LR/XGBoost)")
    
    
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
    
    
    caminho_treino = DB_ROOT / 'treino' / 'features_full.csv'
    caminho_validacao = DB_ROOT / 'validacao' / 'features_full.csv'
    caminho_teste = DB_ROOT / 'teste' / 'features_full.csv'
    
    
    group_map = {'NonDemented': 0, 'Demented': 1} 
    features = [
        'Ventricle_Area', 'Ventricle_Perimeter', 'Ventricle_Circularity', 
        'Ventricle_Eccentricity', 'Ventricle_Solidity', 'Ventricle_MajorAxisLength'
    ]

    def processar_csv(caminho):
        try:
            df = pd.read_csv(caminho, sep=';', decimal=',')
            df['Group_Encoded'] = df['Group'].map(group_map)
            df = df.dropna(subset=['Group_Encoded'] + features)
            X = df[features]
            y_class = df['Group_Encoded'] 
            y_reg = df['Age']             
            return X, y_class, y_reg
        except FileNotFoundError:
             return None, None, None
    
    X_train, y_class_train, y_reg_train = processar_csv(caminho_treino)
    X_val, y_class_val, y_reg_val = processar_csv(caminho_validacao)
    X_test, y_class_test, y_reg_test = processar_csv(caminho_teste)
    
    if X_train is None or X_test is None or X_val is None:
        return False, f"ERRO: Arquivos CSV de treino/validação/teste não encontrados em {DB_ROOT}. Execute 'Preparar base de dados' primeiro."
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        
        lin_reg_classifier = LinearRegression()
        lin_reg_classifier.fit(X_train, y_class_train)
        joblib.dump(lin_reg_classifier, MODEL_LR_DEMENCIA_PATH)
        
        
        raw_predictions_val = lin_reg_classifier.predict(X_val)
        best_threshold = 0.5
        best_accuracy = 0
        for threshold in np.arange(0.00, 1.01, 0.01):
            temp_preds = [1 if val >= threshold else 0 for val in raw_predictions_val]
            current_accuracy = accuracy_score(y_class_val, temp_preds)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_threshold = threshold
        
        joblib.dump(best_threshold, MODEL_LR_DEMENCIA_THRESHOLD_PATH)
        
        
        raw_predictions_test = lin_reg_classifier.predict(X_test)
        class_predictions_test = [1 if val >= best_threshold else 0 for val in raw_predictions_test]
        cm = confusion_matrix(y_class_test, class_predictions_test)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        plt.title(f'Matriz de Confusão - LR (Limiar={best_threshold:.2f})')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.savefig(CM_SHALLOW_PATH, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        
        xgb_regressor = XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=500, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42
        )
        xgb_regressor.fit(X_train, y_reg_train)
        joblib.dump(xgb_regressor, MODEL_XGB_IDADE_PATH)
        
        
        age_predictions = xgb_regressor.predict(X_test)
        mae = mean_absolute_error(y_reg_test, age_predictions)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_reg_test, age_predictions, alpha=0.7, color='green', edgecolors='k')
        plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'k--', lw=2)
        plt.xlabel('Idade Real')
        plt.ylabel('Idade Predita (XGBoost)')
        plt.title(f'Regressão: Idade Real vs Predita (MAE: {mae:.2f} anos)')
        plt.grid(True)
        plt.savefig(SCATTER_SHALLOW_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True, "Treinamento Raso (LR/XGBoost) finalizado com sucesso!"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Erro crítico durante o treinamento raso: {e}"

def predict_single_classification_dl(nii_path):
    try:
        model = load_model(MODEL_DL_DEMENCIA_PATH)
    except Exception as e:
        print(f"Erro ao carregar {MODEL_DL_DEMENCIA_PATH}: {e}")
        return 0.0, "Erro: Modelo não encontrado!!!"

    
    df_tmp = pd.DataFrame({"path": [nii_path], "y": [0]})

    
    ds_tmp = make_dataset(df_tmp, training=False)

    
    prob = float(model.predict(ds_tmp, verbose=0)[0, 0])
    pred_label_int = 1 if prob >= 0.5 else 0
    pred_label_str = "Demented" if pred_label_int == 1 else "NonDemented"

    return prob, pred_label_str

def predict_single_age_dl(nii_path):
    try:
        reg_model = load_model(MODEL_DL_IDADE_PATH)
        age_min, age_max = np.load(MODEL_DL_AGE_STATS_PATH)
    except Exception as e:
        print(f"Erro ao  carregar {MODEL_DL_IDADE_PATH} ou {MODEL_DL_AGE_STATS_PATH} : {e}")
        return 0.0, 0.0

    
    df_tmp = pd.DataFrame({"path": [nii_path], "y": [0.0]})

    
    ds_tmp = make_dataset(df_tmp, training=False)

    
    age_norm = float(reg_model.predict(ds_tmp, verbose=0)[0, 0])

    
    age_real = (age_norm * (age_max - age_min)) + age_min

    return age_real, age_norm






class LabelComZoom(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap_original = None
        
        self._escala = 0.6
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_pixmap_original(self, pixmap):
        self._pixmap_original = pixmap
        
        self._escala = 0.6
        self._atualizar_pixmap()

    def _atualizar_pixmap(self):
        if self._pixmap_original is None:
            return

        largura = int(self._pixmap_original.width() * self._escala)
        altura = int(self._pixmap_original.height() * self._escala)
        if largura <= 0 or altura <= 0:
            return

        img_redimensionada = self._pixmap_original.scaled(
            largura,
            altura,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(img_redimensionada)

    def wheelEvent(self, event):
        if self._pixmap_original is None:
            return

        delta = event.angleDelta().y()
        if delta > 0:
            self._escala *= 1.1
        else:
            self._escala /= 1.1

        
        self._escala = max(0.4, min(self._escala, 3.0))
        self._atualizar_pixmap()

    def mouseDoubleClickEvent(self, event):
        if self._pixmap_original is None:
            return
        self._escala = 0.6
        self._atualizar_pixmap()
        super().mouseDoubleClickEvent(event)

    
    def zoom_in(self, factor: float = 1.2):
        if self._pixmap_original is None:
            return
        self._escala *= factor
        self._escala = max(0.4, min(self._escala, 3.0))
        self._atualizar_pixmap()

    def zoom_out(self, factor: float = 1.2):
        if self._pixmap_original is None:
            return
        self._escala /= factor
        self._escala = max(0.4, min(self._escala, 3.0))
        self._atualizar_pixmap()

    def reset_zoom(self):
        if self._pixmap_original is None:
            return
        
        self._escala = 0.6
        self._atualizar_pixmap()


class InterfaceGrafica(QMainWindow):
    def __init__(self):
        super().__init__()
        self.imagem_carregada = False
        self.dataframe = self.load_dataframe()
        if self.dataframe is None:
            print("Erro: Dataframe não encontrado. Favor verificar  se o dataframe está no caminho esperado")
        self.setGeometry(105, 105, 1400, 900)
        self.load_menu_bar()
        self.load_barra_status()
        self.gerar_abas()
        self.update_abas()

    def load_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        menu_modelo = menu_bar.addMenu("Modelos")

        acao_treinar_raso = QAction("Realizar treinamento (Classificador Raso)", self)
        acao_treinar_raso.triggered.connect(self.iniciar_treinamento_raso_handler)
        menu_modelo.addAction(acao_treinar_raso)
        
        acao_treinar = QAction("Realizar treinamento (Deep Learning)", self)
        acao_treinar.triggered.connect(self.iniciar_treinamento)
        menu_modelo.addAction(acao_treinar)
        
        acao_preparar = QAction("Preparar base de dados", self)
        acao_preparar.triggered.connect(self.iniciar_preparacao_base)
        menu_modelo.addAction(acao_preparar)

        
        acao_scatter_features = QAction("Visualizar scatterplots das features", self)
        acao_scatter_features.triggered.connect(self.visualizar_scatter_features)
        menu_modelo.addAction(acao_scatter_features)
        
        menu_modelo.addSeparator()
        
        acao_visualizar_shallow = QAction("Visualizar gráficos (Classificador Raso)", self)
        acao_visualizar_shallow.triggered.connect(self.visualizar_performance_shallow)
        menu_modelo.addAction(acao_visualizar_shallow)

        acao_visualizar = QAction("Visualizar graficos de treinamento (Deep Learning)", self)
        acao_visualizar.triggered.connect(self.visualizar_performance)
        menu_modelo.addAction(acao_visualizar)

        
        menu_acess = menu_bar.addMenu("Acessibilidade")
        acao_fonte_mais = QAction("Aumentar fonte", self)
        acao_fonte_mais.triggered.connect(self.aumentar_fonte_global)
        menu_acess.addAction(acao_fonte_mais)

        acao_fonte_menos = QAction("Diminuir fonte", self)
        acao_fonte_menos.triggered.connect(self.diminuir_fonte_global)
        menu_acess.addAction(acao_fonte_menos)


    def load_barra_status(self):
        self.barra_status = QStatusBar()
        self.setStatusBar(self.barra_status)
        self.barra_status.showMessage("Carregue uma imagem para iniciar o processamento")

    def gerar_abas(self):
        self.abas_centrais = QTabWidget(self)
        self.abas_centrais.currentChanged.connect(self.ao_trocar_aba)
        self.aba_visualizador = QWidget()
        self.gerar_visualizador_imagens()
        self.abas_centrais.addTab(self.aba_visualizador, "Carregar imagem")
        self.aba_segmentacao = QWidget()
        self._montar_aba_segmentacao()
        self.abas_centrais.addTab(self.aba_segmentacao, "Resultados ")
        self.setCentralWidget(self.abas_centrais)

    def gerar_visualizador_imagens(self):
        layout = QVBoxLayout(self.aba_visualizador)
        controles_layout = QHBoxLayout()
        
        self.btn_abrir_imagem = QPushButton("Carregar imagem")
        self.btn_abrir_imagem.clicked.connect(self.abrir_processar_imagem)
        controles_layout.addWidget(self.btn_abrir_imagem)
        controles_layout.addStretch()
        
        self.btn_zoom_in = QPushButton("Zoom (+)")
        self.btn_zoom_in.clicked.connect(self.aplicar_zoom_in)
        controles_layout.addWidget(self.btn_zoom_in)
        self.btn_zoom_out = QPushButton("Zoom (-)")
        self.btn_zoom_out.clicked.connect(self.aplicar_zoom_out)
        controles_layout.addWidget(self.btn_zoom_out)
        self.btn_reset_zoom = QPushButton("Resetar Zoom")
        self.btn_reset_zoom.clicked.connect(self.resetar_zoom)
        controles_layout.addWidget(self.btn_reset_zoom)
        layout.addLayout(controles_layout)
        self.cena_visualizador = QGraphicsScene(self)
        self.view_visualizador = QGraphicsView(self.cena_visualizador)
        self.view_visualizador.setRenderHint(QPainter.RenderHint.Antialiasing) 
        self.view_visualizador.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.item_pixmap_visualizador = QGraphicsPixmapItem()
        self.cena_visualizador.addItem(self.item_pixmap_visualizador)
        layout.addWidget(self.view_visualizador)
        
        self.btn_zoom_in.setEnabled(False)
        self.btn_zoom_out.setEnabled(False)
        self.btn_reset_zoom.setEnabled(False)
        
        self.btn_abrir_imagem.setFixedSize(150, 40)
        self.btn_zoom_in.setFixedSize(200, 65)
        self.btn_zoom_out.setFixedSize(200, 65)
        self.btn_reset_zoom.setFixedSize(200, 65)

    def _criar_view_processamento(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        cena = QGraphicsScene(self)
        view = QGraphicsView(cena)
        view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(view)
        widget.setContentsMargins(0,0,0,0)
        layout.setContentsMargins(0,0,0,0)
        return widget, cena, view

    def _montar_aba_segmentacao(self):
        layout = QVBoxLayout(self.aba_segmentacao) 
        divisor = QSplitter(Qt.Orientation.Vertical) 
        painel_superior = QWidget()
        layout_superior = QVBoxLayout(painel_superior)
        controles_zoom_layout = QHBoxLayout()
        controles_zoom_layout.addStretch()
        btn_zoom_in_seg = QPushButton("Zoom +")
        btn_zoom_in_seg.clicked.connect(self.aplicar_zoom_in)
        controles_zoom_layout.addWidget(btn_zoom_in_seg)
        btn_zoom_out_seg = QPushButton("Zoom -")
        btn_zoom_out_seg.clicked.connect(self.aplicar_zoom_out)
        controles_zoom_layout.addWidget(btn_zoom_out_seg)
        btn_reset_zoom_seg = QPushButton("Resetar zoom")
        btn_reset_zoom_seg.clicked.connect(self.resetar_zoom)
        controles_zoom_layout.addWidget(btn_reset_zoom_seg)
        layout_superior.addLayout(controles_zoom_layout)
        btn_zoom_in_seg.setFixedSize(200, 65)
        btn_zoom_out_seg.setFixedSize(200, 65)
        btn_reset_zoom_seg.setFixedSize(200, 65)
        
        
        self.abas_processamento = QTabWidget()
        widget_orig_proc, self.cena_orig_proc, self.view_orig_proc = self._criar_view_processamento()
        self.item_pixmap_orig_proc = QGraphicsPixmapItem()
        self.cena_orig_proc.addItem(self.item_pixmap_orig_proc)
        self.abas_processamento.addTab(widget_orig_proc, "Original")
        widget_preproc, self.cena_preproc, self.view_preproc = self._criar_view_processamento()
        self.item_pixmap_preproc = QGraphicsPixmapItem()
        self.cena_preproc.addItem(self.item_pixmap_preproc)
        self.abas_processamento.addTab(widget_preproc, "Pré-processamento")
        widget_seg_final, self.cena_seg_final, self.view_seg_final = self._criar_view_processamento()
        self.item_pixmap_seg_final = QGraphicsPixmapItem()
        self.cena_seg_final.addItem(self.item_pixmap_seg_final)
        self.abas_processamento.addTab(widget_seg_final, "Segmentada")
        layout_superior.addWidget(self.abas_processamento)
        divisor.addWidget(painel_superior)
        
        painel_inferior = QWidget()
        layout_inferior = QVBoxLayout(painel_inferior)
        
        layout_inferior.addWidget(QLabel("Características extraídas (usadas nos modelos Rasos)"))
        self.tabela_caracteristicas = QTableWidget()
        self.tabela_caracteristicas.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabela_caracteristicas.setAlternatingRowColors(True)
        layout_inferior.addWidget(self.tabela_caracteristicas, 1) 
        
        
        layout_resultados = QHBoxLayout()
        widget_joblib = QWidget()
        layout_joblib = QVBoxLayout(widget_joblib)
        layout_joblib.addWidget(QLabel("Resultados (Classificador Raso - LR/XGBoost):"))
        self.log_classificacao = QTextEdit()
        self.log_classificacao.setReadOnly(True)
        self.log_classificacao.setMaximumHeight(100)
        layout_joblib.addWidget(self.log_classificacao)
        self.log_regressao = QTextEdit()
        self.log_regressao.setReadOnly(True)
        self.log_regressao.setMaximumHeight(100)
        layout_joblib.addWidget(self.log_regressao)
        layout_resultados.addWidget(widget_joblib)

        
        widget_dl = QWidget()
        layout_dl = QVBoxLayout(widget_dl)
        layout_dl.addWidget(QLabel("Resultados (Deep Learning - EfficientNet):"))
        self.log_classificacao_dl = QTextEdit()
        self.log_classificacao_dl.setReadOnly(True)
        self.log_classificacao_dl.setMaximumHeight(100)
        layout_dl.addWidget(self.log_classificacao_dl)
        self.log_regressao_dl = QTextEdit()
        self.log_regressao_dl.setReadOnly(True)
        self.log_regressao_dl.setMaximumHeight(100)
        layout_dl.addWidget(self.log_regressao_dl)
        layout_resultados.addWidget(widget_dl)

        layout_inferior.addLayout(layout_resultados)
        
        divisor.addWidget(painel_inferior)
        divisor.setSizes([600, 400])
        layout.addWidget(divisor)

    def load_dataframe(self):
        try:
            df_oasis = pd.read_csv(OASIS_CSV_PATH, sep=';', decimal=',')
            df_oasis['Group'] = df_oasis['Group'].map({
                'Nondemented': 'NonDemented', 
                'Demented': 'Demented', 
                'Converted': 'Converted'
            })
            df_oasis = df_oasis.dropna(subset=['Group'])
            print("Metadados OK")
            return df_oasis
        except FileNotFoundError:
            print(f"Erro {OASIS_CSV_PATH} não encontrado.")
            return None
            
    def processar_img_input(self, caminho_arquivo):
        
        features = [
            'Ventricle_Area', 'Ventricle_Perimeter', 'Ventricle_Circularity', 
            'Ventricle_Eccentricity', 'Ventricle_Solidity', 'Ventricle_MajorAxisLength'
        ]
            
        resultados = { "error": None }
        try:
            if self.dataframe is None:
                return {"error": f"Falha ao carregar CSV ({OASIS_CSV_PATH}). Verifique o caminho."}
            if caminho_arquivo.endswith(('.nii', '.nii.gz')):
                nii_img = nib.load(caminho_arquivo)
                data = nii_img.get_fdata()
                if data.ndim == 3:
                    slice_z = data.shape[2] // 2
                    image_slice = data[:, :, slice_z]
                elif data.ndim == 2:
                    image_slice = data
                elif data.ndim == 4:
                    slice_z = data.shape[2] // 2
                    image_slice = data[:, :, slice_z, 0]
                else:
                    raise ValueError(f"Dimensionalidade inesperada: {data.ndim}")
            else:
                pil = Image.open(caminho_arquivo).convert('L')
                image_slice = np.array(pil)
            
            image_slice = np.rot90(image_slice)
            resultados["image_slice_np"] = image_slice

            print("Executando segmentação de ventrículos...")
            seg_results = self.exec_segmentacao(image_slice)
            ventricle_mask = seg_results['ventriculos']
            preprocessed_img = seg_results['pre_processamento']
            
            resultados["pixmap_original"] = self.conversao_np_qpixmap(image_slice)
            resultados["pixmap_preproc"] = self.conversao_np_qpixmap(preprocessed_img)

            print("Extraindo features morfológicas...")
            features_extraidas = self.extract_features(ventricle_mask)

            mri_id = self.get_id_img(caminho_arquivo)
            
            
            try:
                metadata = self.dataframe[self.dataframe['MRI ID'] == mri_id].iloc[0]
                subject_id = metadata['Subject ID']
                age = metadata['Age']
                group = metadata['Group'] 
            except IndexError:
                 return {"error": f"MRI ID '{mri_id}' não encontrado no arquivo CSV: {OASIS_CSV_PATH}."}

            data_dict = {
                'Subject ID': subject_id, 'MRI ID': mri_id, 'Group': group, 'Age': age,
                **features_extraidas
            }
            df_single = pd.DataFrame([data_dict])
            df_single['Group_num'] = df_single['Group'].map({'NonDemented': 0, 'Demented': 1, 'Converted': 1})
            
            resultados["features_tabela"] = df_single[['Subject ID','MRI ID','Group','Age','Ventricle_Area','Ventricle_Perimeter','Ventricle_Circularity','Ventricle_Eccentricity','Ventricle_Solidity','Ventricle_MajorAxisLength']].copy()

            
            print("Executando predição de Classificação (Regressão Linear)...")
            X_demencia_lr = df_single[features] 
            
            model_lr_dem = joblib.load(MODEL_LR_DEMENCIA_PATH)
            optimal_threshold = joblib.load(MODEL_LR_DEMENCIA_THRESHOLD_PATH) 

            raw_pred_lr = model_lr_dem.predict(X_demencia_lr)[0]
            
            
            pred_lr_dem_label = 'Demented' if raw_pred_lr >= optimal_threshold else 'NonDemented'

            resultados["reporte_classificacao"] = (
                f"Grupo Real: {group}\n"
                f"Predição (Regressão Linear): {pred_lr_dem_label} (Score Bruto: {raw_pred_lr:.4f}, Limiar: {optimal_threshold:.2f})")

            
            print("Executando predição de Regressão (XGBoost)...")
            X_idade_xgb = df_single[features] 
            
            model_xgb_age = joblib.load(MODEL_XGB_IDADE_PATH)
            pred_xgb_age = model_xgb_age.predict(X_idade_xgb)[0]
            
            resultados["reporte_regressao"] = (
                f"Idade Real: {age}\n"
                f"Idade Predita (XGBoost): {pred_xgb_age:.2f} anos")

            print("Executando predição (Deep Learning)...")
            prob_dl, label_dl = predict_single_classification_dl(caminho_arquivo)
            age_dl, _ = predict_single_age_dl(caminho_arquivo)

            resultados["reporte_classificacao_dl"] = (
                f"Grupo Real: {group}\n"
                f"Predição (EfficientNet): {label_dl} (Prob: {prob_dl:.4f})")
            
            resultados["reporte_regressao_dl"] = (
                f"Idade Real: {age}\n"
                f"Idade Predita (EfficientNet): {age_dl:.2f} anos")

            
            fig, ax = plt.subplots(figsize=(6,6), dpi=100)
            ax.imshow(image_slice, cmap='gray')
            ax.contour(ventricle_mask, levels=[0.5], colors='yellow', linewidths=1)
            ax.set_axis_off()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            pil_seg = Image.open(buf).convert('RGB')
            q_img_seg = ImageQt.ImageQt(pil_seg)
            resultados["pixmap_segmentada"] = QPixmap.fromImage(q_img_seg)
            
            return resultados
            
        except FileNotFoundError as e:
            return {"error": f"Modelo não encontrado: {e.filename}. Execute o treinamento."}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Erro inesperado no processamento: {e}"}

    def treinar_modelo(self):

        print("Iniciando pipeline de treinamento Deep Learning...")
        os.makedirs(OUTPUT_DIR, exist_ok=True) 
        os.makedirs(MODELS_DIR, exist_ok=True) 
        
        try:
            
            sucesso_class, msg_class = iniciar_treino_classificacao()
            if not sucesso_class:
                return False, f"Falha na classificação: {msg_class}"
            
            
            sucesso_reg, msg_reg = iniciar_treino_regressao()
            if not sucesso_reg:
                return False, f"Falha na regressão: {msg_reg}"

            msg_final = f"Treinamento DL finalizado!\nClassificação: {msg_class}\nRegressão: {msg_reg}"
            print(msg_final)
            return True, msg_final

        except Exception as e:
            import traceback
            traceback.print_exc()
            msg_erro = f"Erro crítico durante o treinamento: {e}"
            print(msg_erro)
            return False, msg_erro
            
    def iniciar_treinamento_raso_handler(self):
        confirmacao = QMessageBox.question(self, 
            "Confirmação de Treinamento (Classificador Raso)", 
            "Este processo irá treinar os modelos rasos (LR e XGBoost) com base nas features extraídas e nas divisões de dados existentes.\n\n"
            "Deseja continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if confirmacao == QMessageBox.StandardButton.No:
            self.barra_status.showMessage("Treinamento raso cancelado.", 5000)
            return

        self.barra_status.showMessage("Iniciando treinamento Raso... (verifique o console)")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents() 
        
        
        sucesso, mensagem = iniciar_treinamento_raso() 
        
        QApplication.restoreOverrideCursor()
        
        if sucesso:
            self.barra_status.showMessage("Treinamento Raso finalizado!", 10000)
            QMessageBox.information(self, "Treinamento Concluído", mensagem)
            self.visualizar_performance_shallow()
        else:
            QMessageBox.critical(self, "Erro no Treinamento Raso", mensagem)
            self.barra_status.showMessage(f"Erro no treinamento raso: {mensagem}", 5000)

    
    def exec_segmentacao(self, image_slice, n_clusters=N_CLUSTERS, min_area_brain=MIN_AREA_BRAIN, min_area_ventricle=MIN_AREA_VENTRICLE, max_area_ventricle=MAX_AREA_VENTRICLE, center_tolerance_ratio=CENTER_TOLERANCE_RATIO):
        original_shape = image_slice.shape
        img_norm = rescale_intensity(image_slice, out_range=(0, 1))
        img_clahe = equalize_adapthist(img_norm)
        img_smooth = gaussian(img_clahe, sigma=1)
        t = threshold_otsu(img_smooth)
        mask_cerebro = img_smooth > t
        mask_cerebro = binary_opening(mask_cerebro, disk(3))
        mask_cerebro = remove_small_objects(mask_cerebro, min_size=min_area_brain) 
        mask_cerebro = binary_fill_holes(mask_cerebro)
        labels_cerebro = label(mask_cerebro)
        if labels_cerebro.max() == 0:
            return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_smooth}
        maior_comp_label = np.argmax([region.area for region in regionprops(labels_cerebro)]) + 1
        mask_cerebro = (labels_cerebro == maior_comp_label)
        img_sem_cranio = img_smooth * mask_cerebro
        pixels_cerebro = img_sem_cranio[mask_cerebro].reshape(-1, 1)
        if pixels_cerebro.shape[0] < n_clusters:
            return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_sem_cranio}
        kmeans = KMeans(n_clusters=n_clusters, random_state=64, n_init=10).fit(pixels_cerebro)
        centers = kmeans.cluster_centers_.flatten()
        labels_flat = kmeans.labels_
        sorted_indices = np.argsort(centers)
        indice_lcr = sorted_indices[0] 
        labels_kmeans = np.zeros(original_shape, dtype=int)
        labels_kmeans[mask_cerebro] = labels_flat + 1
        mask_lcr_total = (labels_kmeans == (indice_lcr + 1))
        dist_transform = distance_transform_edt(mask_lcr_total)
        labels_lcr = label(dist_transform)
        regioes_lcr = regionprops(labels_lcr)
        mask_ventriculos_proc = np.zeros(original_shape, dtype=bool)
        if not regioes_lcr:
            return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_sem_cranio}
        center_r, center_c = np.array(original_shape) / 2
        max_dist_r = original_shape[0] * center_tolerance_ratio
        max_dist_c = original_shape[1] * center_tolerance_ratio
        for r in regioes_lcr:
            is_correct_size = r.area > min_area_ventricle and r.area < max_area_ventricle
            centroid_r, centroid_c = r.centroid
            is_central_r = abs(centroid_r - center_r) < max_dist_r
            is_central_c = abs(centroid_c - center_c) < max_dist_c
            is_central = is_central_r and is_central_c
            if is_correct_size and is_central:
                mask_ventriculos_proc[labels_lcr == r.label] = True
        mask_ventriculos_proc = binary_fill_holes(mask_ventriculos_proc)
        return {
            'ventriculos': mask_ventriculos_proc,
            'pre_processamento': img_sem_cranio,
        }

    def extract_features(self, ventricle_mask):
        default_features = {
            'Ventricle_Area': 0, 'Ventricle_Perimeter': 0, 'Ventricle_Circularity': 0,
            'Ventricle_Eccentricity': 0, 'Ventricle_Solidity': 0, 'Ventricle_MajorAxisLength': 0
        }
        labels = label(ventricle_mask)
        props = regionprops(labels)
        if not props:
            return default_features
        total_area = 0
        total_perimeter = 0
        metrics_list = {'Circularity': [], 'Eccentricity': [], 'Solidity': [], 'MajorAxisLength': []}
        for region in props:
            total_area += region.area
            total_perimeter += region.perimeter
            if region.perimeter > 0:
                circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
            else:
                circularity = 0
            metrics_list['Circularity'].append(circularity)
            metrics_list['Eccentricity'].append(region.eccentricity)
            metrics_list['Solidity'].append(region.solidity)
            metrics_list['MajorAxisLength'].append(region.major_axis_length)
        features = {
            'Ventricle_Area': total_area,
            'Ventricle_Perimeter': total_perimeter,
            'Ventricle_Circularity': np.mean(metrics_list['Circularity']),
            'Ventricle_Eccentricity': np.mean(metrics_list['Eccentricity']),
            'Ventricle_Solidity': np.mean(metrics_list['Solidity']),
            'Ventricle_MajorAxisLength': np.mean(metrics_list['MajorAxisLength'])
        }
        return features

    def get_id_img(self, nii_path):
        filename = os.path.basename(nii_path)
        mri_id_base = filename.split('.')[0]
        if mri_id_base.endswith('_axl'):
            mri_id = mri_id_base.removesuffix('_axl')
        else:
            mri_id = mri_id_base
        return mri_id

    def conversao_np_qpixmap(self, arr):
        if arr is None:
            return QPixmap()
        a = np.array(arr, copy=True)
        if a.dtype in [np.float32, np.float64]:
            if a.max() > a.min():
                a = (a - a.min()) / (a.max() - a.min()) * 255.0
            else:
                a = a * 255.0
        elif a.dtype == np.uint16:
            a = (a / 256)
        a = a.astype(np.uint8)
        pil_img = Image.fromarray(a, mode='L')
        qim = ImageQt.ImageQt(pil_img)
        return QPixmap.fromImage(qim)

    def update_abas(self):
        self.btn_abrir_imagem.setEnabled(True) 
        self.abas_centrais.setTabEnabled(1, self.imagem_carregada)

    def janela_erro(self, titulo, mensagem):
        QApplication.restoreOverrideCursor()
        self.barra_status.showMessage(f"Erro: {mensagem}", 10000)
        dialogo_erro = QMessageBox(self)
        dialogo_erro.setWindowTitle(titulo)
        dialogo_erro.setText(mensagem)
        dialogo_erro.setIcon(QMessageBox.Icon.Warning)
        dialogo_erro.exec()

    def limpar_resultados_antigos(self):
        self.item_pixmap_visualizador.setPixmap(QPixmap())
        self.item_pixmap_orig_proc.setPixmap(QPixmap())
        self.item_pixmap_preproc.setPixmap(QPixmap())
        self.item_pixmap_seg_final.setPixmap(QPixmap())
        self.tabela_caracteristicas.clear()
        self.tabela_caracteristicas.setRowCount(0)
        self.tabela_caracteristicas.setColumnCount(0)
        self.log_classificacao.clear()
        self.log_regressao.clear()
        self.log_classificacao_dl.clear()
        self.log_regressao_dl.clear()

    def preencher_tabela(self, df_tabela):
        if df_tabela is None or df_tabela.empty:
            self.tabela_caracteristicas.clear()
            return
            
        self.tabela_caracteristicas.setColumnCount(len(df_tabela.columns))
        self.tabela_caracteristicas.setHorizontalHeaderLabels(list(df_tabela.columns))
        self.tabela_caracteristicas.setRowCount(len(df_tabela))
        
        for i, (idx, row) in enumerate(df_tabela.iterrows()):
            for j, col in enumerate(df_tabela.columns):
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    item = f"{val:,.4f}"
                else:
                    item = str(val)
                self.tabela_caracteristicas.setItem(i, j, QTableWidgetItem(item))
                
        self.tabela_caracteristicas.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabela_caracteristicas.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    
    def iniciar_preparacao_base(self):
        confirmacao = QMessageBox.question(self, 
            "Confirmar Preparação da Base", 
            "Este processo irá:\n"
            "1. Segmentar todas as imagens na pasta 'axl'.\n"
            "2. Gerar novos arquivos 'features_full.csv'.\n"
            "3. Dividir a base em pastas 'treino', 'teste' e 'validacao'.\n\n"
            "Isso pode levar vários minutos e sobrescreverá arquivos existentes.\n"
            "Deseja continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if confirmacao == QMessageBox.StandardButton.No:
            return

        self.barra_status.showMessage("Preparando base de dados... Por favor aguarde.")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        try:
            sucesso, msg = self.executar_pipeline_dados()
            if sucesso:
                QMessageBox.information(self, "Sucesso", msg)
                self.barra_status.showMessage("Base de dados preparada com sucesso!", 10000)
            else:
                QMessageBox.critical(self, "Erro", msg)
                self.barra_status.showMessage("Erro ao preparar base.", 10000)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Erro Crítico", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def executar_pipeline_dados(self):
        
        print("Iniciando extração de features em lote...")
        
        import glob
        
        all_files = list(AXL_DIR.glob("*.nii.gz"))
        total_files = len(all_files)
        results_list = []
        
        if total_files == 0:
            return False, f"Nenhuma imagem encontrada em {AXL_DIR}"

        for i, file_path in enumerate(all_files):
            try:
                
                base_name = file_path.name.split('.')[0]
                mr_id = base_name.replace('_axl', '').strip()
                
                nii_img = nib.load(str(file_path))
                data = nii_img.get_fdata()
                
                if data.ndim == 3:
                    slice_z = data.shape[2] // 2
                    image_slice = data[:, :, slice_z]
                elif data.ndim == 2:
                    image_slice = data
                elif data.ndim == 4:
                    slice_z = data.shape[2] // 2
                    image_slice = data[:, :, slice_z, 0]
                else:
                    print(f"Ignorando {mr_id} (dimensão invalida).")
                    continue
                
                image_slice = np.rot90(image_slice) 

                
                seg_result = self.exec_segmentacao(image_slice)
                ventricle_mask = seg_result['ventriculos']
                
                
                features = self.extract_features(ventricle_mask)
                features['MRI ID'] = mr_id
                results_list.append(features)
                
                if i % 20 == 0:
                    self.barra_status.showMessage(f"Processando imagem {i}/{total_files}...")
                    QApplication.processEvents()

            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")

        if not results_list:
            return False, "Falha ao extrair features das imagens."

        df_features = pd.DataFrame(results_list)
        
        
        try:
            df_demographic = pd.read_csv(OASIS_CSV_PATH, sep=';', decimal=',')
            df_demographic['MRI ID'] = df_demographic['MRI ID'].str.strip()
        except Exception as e:
            return False, f"Erro ao ler CSV demográfico: {e}"

        
        df_final = pd.merge(df_demographic, df_features, on='MRI ID', how='left')
        
        
        def map_class_robusta(row):
            group = row['Group']
            cdr = row['CDR']
            if group == 'Converted':
                if cdr > 0: return 'Demented'
                else: return 'NonDemented'
            if group == 'Demented': return 'Demented'
            return 'NonDemented'

        df_final['Group'] = df_final.apply(map_class_robusta, axis=1)
        
        
        features_full_path = DB_ROOT / 'features_full.csv'
        features_ids_path = DB_ROOT / 'features_identifiers.csv'
        
        cols_full_requested = ['Subject ID', 'MRI ID', 'Group', 'Age', 'Ventricle_Area', 'Ventricle_Perimeter', 'Ventricle_Circularity', 'Ventricle_Eccentricity', 'Ventricle_Solidity', 'Ventricle_MajorAxisLength']
        df_final = df_final[df_final['Ventricle_Area'].notna()] 
        
        df_final[cols_full_requested].to_csv(features_full_path, index=False, sep=';', decimal=',')
        
        print("Extração concluída. Iniciando Divisão (Split)...")

        
        
        
        TREINO_DIR = DB_ROOT / 'treino'
        VAL_DIR = DB_ROOT / 'validacao'
        TESTE_DIR = DB_ROOT / 'teste'
        
        for d in [TREINO_DIR / 'imgs', VAL_DIR / 'imgs', TESTE_DIR / 'imgs']:
            os.makedirs(d, exist_ok=True)

        
        df_work = df_final.copy()
        df_work['Group_num'] = df_work['Group'].map({'Demented': 1, 'NonDemented': 0})
        
        
        patient_labels = df_work.groupby('Subject ID')['Group_num'].max()
        patient_ids = patient_labels.index
        labels = patient_labels.values
        
        
        train_val_patients, test_patients, train_val_labels, _ = train_test_split(patient_ids, labels, test_size=0.2, stratify=labels, random_state=42)
        train_patients, val_patients, _, _ = train_test_split(train_val_patients, train_val_labels, test_size=0.2, stratify=train_val_labels, random_state=42)

        
        def processar_subset(patients_list, output_folder):
            subset_df = df_work[df_work['Subject ID'].isin(patients_list)].copy()
            
            
            subset_df[cols_full_requested].to_csv(output_folder / 'features_full.csv', index=False, sep=';', decimal=',')
            
            
            dest_img_dir = output_folder / 'imgs'
            count = 0
            for _, row in subset_df.iterrows():
                mri_id = row['MRI ID']
                
                orig_file = list(AXL_DIR.glob(f"*{mri_id}*.nii.gz"))
                if orig_file:
                    shutil.copy(orig_file[0], dest_img_dir / orig_file[0].name)
                    count += 1
            return count

        c_train = processar_subset(train_patients, TREINO_DIR)
        c_val = processar_subset(val_patients, VAL_DIR)
        c_test = processar_subset(test_patients, TESTE_DIR)

        return True, (f"Processamento concluído!\n\n"
                      f"Imagens processadas: {len(df_final)}\n"
                      f"Treino: {c_train} imgs ({len(train_patients)} pacientes)\n"
                      f"Validação: {c_val} imgs ({len(val_patients)} pacientes)\n"
                      f"Teste: {c_test} imgs ({len(test_patients)} pacientes)\n\n"
                      f"Arquivos salvos em: {DB_ROOT}")

    def iniciar_treinamento(self):
        confirmacao = QMessageBox.question(self, 
            "Confirmação de Treinamento (Deep Learning)", 
            "Este processo irá treinar os modelos de Deep Learning (EfficientNet).\n"
            "Isso pode levar MUITOS minutos (ou horas) e irá sobrescrever os modelos .keras e gráficos atuais.\n\n"
            "**Requer uma GPU e TensorFlow configurado corretamente.**\n\n"
            "Deseja continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if confirmacao == QMessageBox.StandardButton.No:
            self.barra_status.showMessage("Treinamento cancelado.", 5000)
            return
        self.barra_status.showMessage("Iniciando treinamento Deep Learning... (verifique o console)")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents() 
        sucesso, mensagem = self.treinar_modelo()
        QApplication.restoreOverrideCursor()
        
        if sucesso:
            self.barra_status.showMessage("Treinamento DL finalizado!", 10000)
            QMessageBox.information(self, "Treinamento Concluído", mensagem)
            self.visualizar_performance()
        else:
            QMessageBox.critical(self, "Erro no Treinamento", mensagem)
            self.barra_status.showMessage(f"Erro no treinamento: {mensagem}", 5000)

    def visualizar_performance(self):
        dialogo = QDialog(self)
        dialogo.setWindowTitle("Performance do Modelo (Deep Learning)")
        dialogo.setMinimumSize(1000, 700)

        main_layout = QVBoxLayout(dialogo)
        abas = QTabWidget()
        main_layout.addWidget(abas, 1)

        
        tab_classificacao = QWidget()
        layout_class = QHBoxLayout(tab_classificacao)

        
        layout_class_left = QVBoxLayout()
        layout_class_left.addWidget(
            QLabel("Matriz de Confusão (Teste)"),
            0,
            Qt.AlignmentFlag.AlignCenter,
        )

        lbl_cm_dl = LabelComZoom()
        scroll_cm = QScrollArea()
        scroll_cm.setWidget(lbl_cm_dl)
        scroll_cm.setWidgetResizable(True)
        scroll_cm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_class_left.addWidget(scroll_cm, 1)

        
        controles_zoom_cm = QHBoxLayout()
        btn_cm_zoom_in = QPushButton("Zoom (+)")
        btn_cm_zoom_out = QPushButton("Zoom (-)")
        btn_cm_zoom_reset = QPushButton("Resetar zoom")
        btn_cm_zoom_in.clicked.connect(lambda: lbl_cm_dl.zoom_in())
        btn_cm_zoom_out.clicked.connect(lambda: lbl_cm_dl.zoom_out())
        btn_cm_zoom_reset.clicked.connect(lambda: lbl_cm_dl.reset_zoom())
        controles_zoom_cm.addWidget(btn_cm_zoom_in)
        controles_zoom_cm.addWidget(btn_cm_zoom_out)
        controles_zoom_cm.addWidget(btn_cm_zoom_reset)
        layout_class_left.addLayout(controles_zoom_cm)

        layout_class.addLayout(layout_class_left)

        
        layout_class_right = QVBoxLayout()
        layout_class_right.addWidget(
            QLabel("Curvas de Aprendizado (Treino/Val)"),
            0,
            Qt.AlignmentFlag.AlignCenter,
        )

        lbl_curves_dl = LabelComZoom()
        scroll_curves = QScrollArea()
        scroll_curves.setWidget(lbl_curves_dl)
        scroll_curves.setWidgetResizable(True)
        scroll_curves.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_class_right.addWidget(scroll_curves, 1)

        
        controles_zoom_curves = QHBoxLayout()
        btn_curves_zoom_in = QPushButton("Zoom (+)")
        btn_curves_zoom_out = QPushButton("Zoom (-)")
        btn_curves_zoom_reset = QPushButton("Resetar zoom")
        btn_curves_zoom_in.clicked.connect(lambda: lbl_curves_dl.zoom_in())
        btn_curves_zoom_out.clicked.connect(lambda: lbl_curves_dl.zoom_out())
        btn_curves_zoom_reset.clicked.connect(lambda: lbl_curves_dl.reset_zoom())
        controles_zoom_curves.addWidget(btn_curves_zoom_in)
        controles_zoom_curves.addWidget(btn_curves_zoom_out)
        controles_zoom_curves.addWidget(btn_curves_zoom_reset)
        layout_class_right.addLayout(controles_zoom_curves)

        layout_class.addLayout(layout_class_right)

        abas.addTab(tab_classificacao, "Resultados da Classificação de Grupo")

        
        tab_regressao = QWidget()
        layout_reg = QHBoxLayout(tab_regressao)

        
        layout_reg_left = QVBoxLayout()
        layout_reg_left.addWidget(
            QLabel("Dispersão Idade Real vs. Prevista (Teste)"),
            0,
            Qt.AlignmentFlag.AlignCenter,
        )

        lbl_scatter_dl = LabelComZoom()
        scroll_scatter_dl = QScrollArea()
        scroll_scatter_dl.setWidget(lbl_scatter_dl)
        scroll_scatter_dl.setWidgetResizable(True)
        scroll_scatter_dl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_reg_left.addWidget(scroll_scatter_dl, 1)

        controles_zoom_scatter = QHBoxLayout()
        btn_scatter_zoom_in = QPushButton("Zoom (+)")
        btn_scatter_zoom_out = QPushButton("Zoom (-)")
        btn_scatter_zoom_reset = QPushButton("Resetar zoom")
        btn_scatter_zoom_in.clicked.connect(lambda: lbl_scatter_dl.zoom_in())
        btn_scatter_zoom_out.clicked.connect(lambda: lbl_scatter_dl.zoom_out())
        btn_scatter_zoom_reset.clicked.connect(lambda: lbl_scatter_dl.reset_zoom())
        controles_zoom_scatter.addWidget(btn_scatter_zoom_in)
        controles_zoom_scatter.addWidget(btn_scatter_zoom_out)
        controles_zoom_scatter.addWidget(btn_scatter_zoom_reset)
        layout_reg_left.addLayout(controles_zoom_scatter)

        layout_reg.addLayout(layout_reg_left)

        
        layout_reg_right = QVBoxLayout()
        layout_reg_right.addWidget(
            QLabel("Curvas de Aprendizado (Treino/Val)"),
            0,
            Qt.AlignmentFlag.AlignCenter,
        )

        lbl_curves_reg_dl = LabelComZoom()
        scroll_curves_reg = QScrollArea()
        scroll_curves_reg.setWidget(lbl_curves_reg_dl)
        scroll_curves_reg.setWidgetResizable(True)
        scroll_curves_reg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_reg_right.addWidget(scroll_curves_reg, 1)

        controles_zoom_curves_reg = QHBoxLayout()
        btn_curves_reg_zoom_in = QPushButton("Zoom (+)")
        btn_curves_reg_zoom_out = QPushButton("Zoom (-)")
        btn_curves_reg_zoom_reset = QPushButton("Resetar zoom")
        btn_curves_reg_zoom_in.clicked.connect(lambda: lbl_curves_reg_dl.zoom_in())
        btn_curves_reg_zoom_out.clicked.connect(lambda: lbl_curves_reg_dl.zoom_out())
        btn_curves_reg_zoom_reset.clicked.connect(lambda: lbl_curves_reg_dl.reset_zoom())
        controles_zoom_curves_reg.addWidget(btn_curves_reg_zoom_in)
        controles_zoom_curves_reg.addWidget(btn_curves_reg_zoom_out)
        controles_zoom_curves_reg.addWidget(btn_curves_reg_zoom_reset)
        layout_reg_right.addLayout(controles_zoom_curves_reg)

        layout_reg.addLayout(layout_reg_right)

        abas.addTab(tab_regressao, "Resultados da Regressão (Idade)")

        btn_fechar = QPushButton("Fechar")
        btn_fechar.clicked.connect(dialogo.accept)
        main_layout.addWidget(btn_fechar, 0, Qt.AlignmentFlag.AlignCenter)

        
        self.load_image(lbl_cm_dl, CM_DL_DEMENCIA_PATH)
        self.load_image(lbl_curves_dl, CURVES_DL_DEMENCIA_PATH)
        self.load_image(lbl_scatter_dl, SCATTER_DL_IDADE_PATH)
        self.load_image(lbl_curves_reg_dl, CURVES_DL_IDADE_PATH)

        dialogo.exec()

    
    def visualizar_performance_shallow(self):

        dialogo = QDialog(self)
        dialogo.setWindowTitle("Performance do Modelo (Classificador Raso)")
        dialogo.setMinimumSize(1000, 700)

        main_layout = QVBoxLayout(dialogo)
        abas = QTabWidget()
        main_layout.addWidget(abas, 1)

        
        tab_classificacao = QWidget()
        layout_class = QVBoxLayout(tab_classificacao)

        lbl_cm_shallow = LabelComZoom()
        scroll_cm_shallow = QScrollArea()
        scroll_cm_shallow.setWidget(lbl_cm_shallow)
        scroll_cm_shallow.setWidgetResizable(True)
        scroll_cm_shallow.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout_class.addWidget(QLabel("Matriz de Confusão (Classificador Raso)"),
                            0,
                            Qt.AlignmentFlag.AlignCenter)
        layout_class.addWidget(scroll_cm_shallow, 1)

        controles_zoom_cm = QHBoxLayout()
        btn_cm_zoom_in = QPushButton("Zoom (+)")
        btn_cm_zoom_out = QPushButton("Zoom (-)")
        btn_cm_zoom_reset = QPushButton("Resetar zoom")
        btn_cm_zoom_in.clicked.connect(lambda: lbl_cm_shallow.zoom_in())
        btn_cm_zoom_out.clicked.connect(lambda: lbl_cm_shallow.zoom_out())
        btn_cm_zoom_reset.clicked.connect(lambda: lbl_cm_shallow.reset_zoom())
        controles_zoom_cm.addWidget(btn_cm_zoom_in)
        controles_zoom_cm.addWidget(btn_cm_zoom_out)
        controles_zoom_cm.addWidget(btn_cm_zoom_reset)
        layout_class.addLayout(controles_zoom_cm)

        abas.addTab(tab_classificacao, "Resultados da Classificação de Grupo")

        
        tab_regressao = QWidget()
        layout_reg = QVBoxLayout(tab_regressao)

        lbl_scatter_shallow = LabelComZoom()
        scroll_scatter_sh = QScrollArea()
        scroll_scatter_sh.setWidget(lbl_scatter_shallow)
        scroll_scatter_sh.setWidgetResizable(True)
        scroll_scatter_sh.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout_reg.addWidget(QLabel("Dispersão Idade Real vs. Prevista (XGBoost)"),
                            0,
                            Qt.AlignmentFlag.AlignCenter)
        layout_reg.addWidget(scroll_scatter_sh, 1)

        controles_zoom_scatter = QHBoxLayout()
        btn_scatter_zoom_in = QPushButton("Zoom (+)")
        btn_scatter_zoom_out = QPushButton("Zoom (-)")
        btn_scatter_zoom_reset = QPushButton("Resetar zoom")
        btn_scatter_zoom_in.clicked.connect(lambda: lbl_scatter_shallow.zoom_in())
        btn_scatter_zoom_out.clicked.connect(lambda: lbl_scatter_shallow.zoom_out())
        btn_scatter_zoom_reset.clicked.connect(lambda: lbl_scatter_shallow.reset_zoom())
        controles_zoom_scatter.addWidget(btn_scatter_zoom_in)
        controles_zoom_scatter.addWidget(btn_scatter_zoom_out)
        controles_zoom_scatter.addWidget(btn_scatter_zoom_reset)
        layout_reg.addLayout(controles_zoom_scatter)

        abas.addTab(tab_regressao, "Resultados da Regressão (Idade)")

        btn_fechar = QPushButton("Fechar")
        btn_fechar.clicked.connect(dialogo.accept)
        main_layout.addWidget(btn_fechar, 0, Qt.AlignmentFlag.AlignCenter)

        
        self.load_image(lbl_cm_shallow, CM_SHALLOW_PATH)
        self.load_image(lbl_scatter_shallow, SCATTER_SHALLOW_PATH)

        dialogo.exec()


    
    
    
    def ajustar_fonte_global(self, delta):

        app = QApplication.instance()
        if app is None:
            return

        fonte_atual = app.font()
        tamanho = fonte_atual.pointSize()
        if tamanho <= 0:
            tamanho = 10  

        novo_tam = tamanho + delta
        
        novo_tam = max(8, min(28, novo_tam))

        fonte_atual.setPointSize(novo_tam)
        app.setFont(fonte_atual)

        try:
            self.barra_status.showMessage(f"Tamanho da fonte: {novo_tam} pt", 3000)
        except Exception:
            
            pass

    def aumentar_fonte_global(self):
        self.ajustar_fonte_global(2)

    def diminuir_fonte_global(self):
        self.ajustar_fonte_global(-2)

    
    
    
    
    def visualizar_scatter_features(self):
        features_path = DB_ROOT / 'features_full.csv'
        if not features_path.exists():
            QMessageBox.warning(
                self,
                "Features não encontradas",
                "Não encontrei o arquivo features_full.csv.\n"
                "Rode primeiro o menu 'Preparar base de dados'."
            )
            return

        
        try:
            df = pd.read_csv(features_path, sep=';', decimal=',')
        except Exception as e:
            QMessageBox.critical(self, "Erro ao ler CSV", f"Erro lendo {features_path}:\n{e}")
            return

        feature_cols = [
            'Ventricle_Area',
            'Ventricle_Perimeter',
            'Ventricle_Circularity',
            'Ventricle_Eccentricity',
            'Ventricle_Solidity',
            'Ventricle_MajorAxisLength'
        ]

        
        colunas_validas = [c for c in feature_cols if c in df.columns]
        if len(colunas_validas) < 2:
            QMessageBox.warning(
                self,
                "Dados insuficientes",
                "Não tem features suficientes no CSV pra montar os gráficos."
            )
            return

        if 'MRI ID' not in df.columns:
            QMessageBox.critical(
                self,
                "CSV inválido",
                "O arquivo features_full.csv precisa ter a coluna 'MRI ID' "
                "para recuperar a classe original."
            )
            return

        
        
        
        try:
            df_oasis = pd.read_csv(OASIS_CSV_PATH, sep=';', decimal=',')
            df_oasis['MRI ID'] = df_oasis['MRI ID'].astype(str).str.strip()
            df_oasis['Group'] = df_oasis['Group'].astype(str).str.strip()

            df_merge = df.copy()
            df_merge['MRI ID'] = df_merge['MRI ID'].astype(str).str.strip()

            
            df_merge = df_merge.merge(
                df_oasis[['MRI ID', 'Group']],
                on='MRI ID',
                how='left',
                suffixes=('', '_orig')
            )

            if 'Group_orig' in df_merge.columns:
                
                df_merge['GroupPlot'] = df_merge['Group_orig'].fillna(df_merge.get('Group'))
            else:
                df_merge['GroupPlot'] = df_merge.get('Group')

        except Exception:
            
            df_merge = df.copy()
            df_merge['GroupPlot'] = df_merge.get('Group')

        if 'GroupPlot' not in df_merge.columns:
            QMessageBox.warning(
                self,
                "Sem informação de classe",
                "Não encontrei coluna de classe para colorir os pontos."
            )
            return

        base_df = df_merge.dropna(subset=colunas_validas + ['GroupPlot'])
        if base_df.empty:
            QMessageBox.warning(
                self,
                "Dados vazios",
                "Depois de remover valores ausentes, não sobrou dado para plotar."
            )
            return

        
        def pegar_cor(gr):
            g = str(gr).strip()
            if g == 'Converted':
                return 'black'
            if g in ['NonDemented', 'Nondemented']:
                return 'blue'
            if g == 'Demented':
                return 'red'
            return 'gray'

        cores = base_df['GroupPlot'].apply(pegar_cor)

        
        
        
        n_features = len(colunas_validas)
        n_pairs = math.comb(n_features, 2)
        n_cols = 4
        n_rows = math.ceil(n_pairs / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        idx_plot = 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                x_name = colunas_validas[i]
                y_name = colunas_validas[j]

                ax = axes[idx_plot]
                ax.scatter(
                    base_df[x_name],
                    base_df[y_name],
                    c=cores,
                    alpha=0.7,
                    s=10
                )
                ax.set_xlabel(x_name)
                ax.set_ylabel(y_name)

                idx_plot += 1

        
        for k in range(idx_plot, len(axes)):
            axes[k].axis('off')

        plt.suptitle("Scatterplots das features dos ventrículos por classe", fontsize=14)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(SCATTER_FEATURES_PATH, dpi=300, bbox_inches='tight')
        plt.close(fig)

        
        
        
        dialogo = QDialog(self)
        dialogo.setWindowTitle("Scatterplots das features (ventrículos)")
        dialogo.setMinimumSize(1000, 700)
        layout = QVBoxLayout(dialogo)

        lbl_scatter = LabelComZoom()
        scroll_scatter = QScrollArea()
        scroll_scatter.setWidget(lbl_scatter)
        scroll_scatter.setWidgetResizable(True)
        scroll_scatter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(scroll_scatter, 1)

        
        controles_zoom = QHBoxLayout()
        btn_zoom_in = QPushButton("Zoom (+)")
        btn_zoom_out = QPushButton("Zoom (-)")
        btn_zoom_reset = QPushButton("Resetar zoom")
        btn_zoom_in.clicked.connect(lambda: lbl_scatter.zoom_in())
        btn_zoom_out.clicked.connect(lambda: lbl_scatter.zoom_out())
        btn_zoom_reset.clicked.connect(lambda: lbl_scatter.reset_zoom())
        controles_zoom.addWidget(btn_zoom_in)
        controles_zoom.addWidget(btn_zoom_out)
        controles_zoom.addWidget(btn_zoom_reset)
        layout.addLayout(controles_zoom)

        
        self.load_image(lbl_scatter, SCATTER_FEATURES_PATH)

        btn_fechar = QPushButton("Fechar")
        btn_fechar.clicked.connect(dialogo.accept)
        layout.addWidget(btn_fechar, 0, Qt.AlignmentFlag.AlignCenter)

        dialogo.exec()

    def load_image(self, label_widget, path_imgs):
        path_str = str(path_imgs)
        pixmap = QPixmap(path_str)
        if pixmap.isNull():
            pixmap = self.criar_box_img_nao_encontrada(path_str)

        
        if hasattr(label_widget, "set_pixmap_original"):
            label_widget.set_pixmap_original(pixmap)
        else:
            
            label_widget.setPixmap(pixmap)


    def criar_box_img_nao_encontrada(self, path_str):
        pixmap = QPixmap(600, 500)
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        painter.setPen(QColor(200, 0, 0))
        painter.setFont(QFont("Arial", 10))
        text = f"Imagem não encontrada.\n\nExecute o treinamento para gerar o gráfico:\n{os.path.basename(path_str)}"
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
        return pixmap

    def abrir_processar_imagem(self):
        filtro = "Arquivos NIfTI (*.nii *.nii.gz);;Imagens Padrão (*.png *.jpg *.jpeg)"
        caminho, _ = QFileDialog.getOpenFileName(self, "Abrir imagem", "", filtro)
        if not caminho:
            return
        self.barra_status.showMessage("Realizando o carregamento e processamento...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()  
        self.limpar_resultados_antigos()
        
        resultados = self.processar_img_input(caminho)
        QApplication.restoreOverrideCursor()

        if resultados.get("error"):
            self.janela_erro("Erro no processamento", resultados["error"])
            self.imagem_carregada = False
            self.update_abas()
            return

        
        pix_orig = resultados["pixmap_original"]
        self.item_pixmap_visualizador.setPixmap(pix_orig)
        self.cena_visualizador.setSceneRect(pix_orig.rect())
        self.view_visualizador.fitInView(pix_orig.rect(), Qt.AspectRatioMode.KeepAspectRatio)    
        self.item_pixmap_orig_proc.setPixmap(pix_orig)
        self.cena_orig_proc.setSceneRect(pix_orig.rect())
        self.view_orig_proc.fitInView(pix_orig.rect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.item_pixmap_preproc.setPixmap(resultados["pixmap_preproc"])
        self.cena_preproc.setSceneRect(resultados["pixmap_preproc"].rect())
        self.view_preproc.fitInView(resultados["pixmap_preproc"].rect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.item_pixmap_seg_final.setPixmap(resultados["pixmap_segmentada"])
        self.cena_seg_final.setSceneRect(resultados["pixmap_segmentada"].rect())
        self.view_seg_final.fitInView(resultados["pixmap_segmentada"].rect(), Qt.AspectRatioMode.KeepAspectRatio)

        
        self.preencher_tabela(resultados["features_tabela"])
        
        
        self.log_classificacao.setPlainText(resultados["reporte_classificacao"])
        self.log_regressao.setPlainText(resultados["reporte_regressao"])
        self.log_classificacao.moveCursor(QTextCursor.MoveOperation.Start)
        self.log_regressao.moveCursor(QTextCursor.MoveOperation.Start)
        
        
        self.log_classificacao_dl.setPlainText(resultados.get("reporte_classificacao_dl", "N/A"))
        self.log_regressao_dl.setPlainText(resultados.get("reporte_regressao_dl", "N/A"))
        self.log_classificacao_dl.moveCursor(QTextCursor.MoveOperation.Start)
        self.log_regressao_dl.moveCursor(QTextCursor.MoveOperation.Start)
        self.imagem_carregada = True
        self.update_abas()       
        self.abas_centrais.setCurrentIndex(1)
        self.abas_processamento.setCurrentIndex(2)
        self.barra_status.showMessage(f"Processamento finalizado: {caminho}", 10000)
        self.btn_zoom_in.setEnabled(True)
        self.btn_zoom_out.setEnabled(True)
        self.btn_reset_zoom.setEnabled(True)

    def aplicar_zoom_in(self):
        view = self.get_current_view()
        if view:
            view.scale(1.2, 1.2)
            self.barra_status.showMessage("Zoom aumentado")
        else:
            self.barra_status.showMessage("Imagem não carregada")

    def aplicar_zoom_out(self):
        view = self.get_current_view()
        if view:
            view.scale(0.8, 0.8)
            self.barra_status.showMessage("Zoom reduzido")
        else:
            self.barra_status.showMessage("Imagem não carregada")

    def resetar_zoom(self):
        view = self.get_current_view()
        if view:
            view.fitInView(view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.barra_status.showMessage("Zoom resetado")
        else:
            self.barra_status.showMessage("Imagem não carregada")

    def ao_trocar_aba(self, index):
        nome = self.abas_centrais.tabText(index)
        self.barra_status.showMessage(f"Aba ativa: {nome}")

    def get_current_view(self):
        widget_aba_ativa = self.abas_centrais.currentWidget()
        
        if widget_aba_ativa == self.aba_visualizador:
            self.view_visualizador.setFocus()
            return self.view_visualizador
        
        if widget_aba_ativa == self.aba_segmentacao:
            indice_sub_aba = self.abas_processamento.currentIndex()
            if indice_sub_aba == 0:
                self.view_orig_proc.setFocus()
                return self.view_orig_proc
            elif indice_sub_aba == 1:
                self.view_preproc.setFocus()
                return self.view_preproc
            elif indice_sub_aba == 2:
                self.view_seg_final.setFocus()
                return self.view_seg_final
        
        return None


if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    
    try:
        import seaborn as sns 
    except ImportError:
        print("Aviso: seaborn não instalado. O treinamento raso pode falhar.")

    app = QApplication(sys.argv)
    janela = InterfaceGrafica()
    janela.show()
    sys.exit(app.exec())