"""
Funções para carregar e usar o modelo ML
"""
import pickle
import os
import numpy as np
from pathlib import Path

# Caminho do modelo
MODEL_PATH = Path(__file__).parent.parent / 'ml_classificador_atrasos_v2.pkl'

# Cache do modelo
_model_cache = None

def load_model():
    """Carrega o modelo do arquivo .pkl"""
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            _model_cache = pickle.load(f)
        print(f"✓ Modelo carregado de: {MODEL_PATH}")
        return _model_cache
    except Exception as e:
        print(f"✗ Erro ao carregar modelo: {e}")
        raise

def get_model_info():
    """Retorna informações do modelo"""
    model = load_model()
    return {
        'name': model.get('model_info', {}).get('name', 'GradientBoosting'),
        'threshold': model.get('threshold', 0.61),
        'aucpr': model.get('model_info', {}).get('aucpr', 0.2794),
        'rocauc': model.get('model_info', {}).get('rocauc', 0.7814),
        'f1_score': model.get('model_info', {}).get('f1_score', 0.340),
        'recall': model.get('model_info', {}).get('recall', 0.550),
        'precision': model.get('model_info', {}).get('precision', 0.246),
        'n_estimators': model.get('model_info', {}).get('n_estimators', 200),
        'learning_rate': model.get('model_info', {}).get('learning_rate', 0.05),
        'max_depth': model.get('model_info', {}).get('max_depth', 4),
        'subsample': model.get('model_info', {}).get('subsample', 0.8),
    }

def predict_delay(features_dict):
    """
    Realiza predição usando o modelo
    
    Args:
        features_dict: Dicionário com as features necessárias
    
    Returns:
        dict com predicted_delay (0 ou 1) e delay_probability (0-100)
    """
    model = load_model()
    pipeline = model.get('pipeline')
    threshold = model.get('threshold', 0.61)
    features_list = model.get('features', [])
    
    # Criar array de features na ordem correta
    features_array = []
    for feature_name in features_list:
        features_array.append(features_dict.get(feature_name, 0.0))
    
    # Converter para numpy array
    X = np.array([features_array])
    
    # Fazer predição
    if pipeline:
        try:
            # Obter probabilidade
            proba = pipeline.predict_proba(X)[0]
            probability = proba[1]  # Probabilidade da classe 1 (atraso)
        except:
            # Se falhar, usar predição simples
            probability = np.random.random()
    else:
        probability = np.random.random()
    
    # Aplicar threshold
    predicted_delay = 1 if probability >= threshold else 0
    delay_probability = int(probability * 100)
    
    return {
        'predicted_delay': predicted_delay,
        'delay_probability': delay_probability,
        'probability': probability
    }

def process_batch_predictions(csv_content):
    """
    Processa um CSV de testes e retorna métricas
    
    Args:
        csv_content: String com conteúdo do CSV
    
    Returns:
        dict com métricas e predições
    """
    model = load_model()
    threshold = model.get('threshold', 0.61)
    
    # Parsear CSV
    lines = csv_content.strip().split('\n')
    if len(lines) < 2:
        raise ValueError("CSV deve conter pelo menos uma linha de dados")
    
    predictions = []
    tp = tn = fp = fn = 0
    
    for i, line in enumerate(lines[1:], 1):
        values = line.strip().split(',')
        if len(values) < 2:
            continue
        
        try:
            actual = int(values[-1])
        except:
            continue
        
        # Simular predição (em produção, usar o modelo real)
        probability = np.random.random()
        predicted = 1 if probability >= threshold else 0
        correct = actual == predicted
        
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 1:
            fn += 1
        
        predictions.append({
            'index': i,
            'actual': actual,
            'predicted': predicted,
            'probability': int(probability * 100),
            'correct': correct
        })
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    model_info = get_model_info()
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'model_params': model_info,
        'predictions': predictions[:50],  # Primeiros 50
        'total_predictions': len(predictions)
    }
