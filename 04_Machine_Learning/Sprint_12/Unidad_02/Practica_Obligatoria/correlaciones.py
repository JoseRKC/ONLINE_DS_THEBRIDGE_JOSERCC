import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import LabelEncoder

# Función para clasificar las variables
def clasificar_variables(df, umbral_continua=0.8, umbral_categorica=10):
    clasificacion = {}
    for col in df.columns:
        if df[col].dtype == 'object' or len(df[col].unique()) <= umbral_categorica:
            clasificacion[col] = 'Categórica'
        elif df[col].dtype in ['int64', 'float64']:
            if df[col].nunique() > umbral_continua:
                clasificacion[col] = 'Numérica Continua'
            else:
                clasificacion[col] = 'Numérica Discreta'
        else:
            clasificacion[col] = 'Binaria'  # Si es una variable binaria
    return clasificacion

# Función para calcular correlación numérica continua vs target
def correlacion_continua_target(df, target, umbral=0.3):
    correlaciones = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != target:
            corr = df[col].corr(df[target])
            if abs(corr) >= umbral:
                correlaciones.append((col, corr))
    return correlaciones

# Función para calcular correlación categórica vs target
def correlacion_categorica_target(df, target, umbral=0.05):
    correlaciones = []
    for col in df.select_dtypes(include=['object']).columns:
        contingency_table = pd.crosstab(df[col], df[target])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value <= umbral:
            correlaciones.append((col, p_value))
    return correlaciones

# Función para calcular correlación binaria vs target
def correlacion_binaria_target(df, target, umbral=0.3):
    correlaciones = []
    for col in df.select_dtypes(include=['bool', 'int64']).columns:
        if len(df[col].unique()) == 2:
            corr, _ = pointbiserialr(df[col], df[target])
            if abs(corr) >= umbral:
                correlaciones.append((col, corr))
    return correlaciones

# Función para calcular correlación entre variables numéricas continuas
def correlacion_continua(df, umbral=0.8):
    correlaciones = []
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2 and abs(corr_matrix[col1][col2]) >= umbral:
                correlaciones.append((col1, col2, corr_matrix[col1][col2]))
    return correlaciones
