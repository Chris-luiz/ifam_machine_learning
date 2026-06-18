import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Caminho do modelo - está na raiz do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_classificador_atrasos_v2.pkl')

print(f"[ML] Carregando modelo de: {MODEL_PATH}")
print(f"[ML] Arquivo existe: {os.path.exists(MODEL_PATH)}")

# Carregar modelo uma única vez
PIPELINE = None
THRESHOLD = 0.61
FEATURES = []
MODEL_INFO = {}
MODEL_LOADED = False
EXPECTED_FEATURES = []

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extrair componentes do modelo
        if isinstance(model_data, dict):
            # O arquivo contém um dicionário com pipeline, threshold e features
            PIPELINE = model_data.get('pipeline')
            THRESHOLD = float(model_data.get('threshold', 0.61))
            FEATURES = model_data.get('features', [])
            MODEL_INFO = model_data.get('model_info', {})
            
            EXPECTED_FEATURES = model_data.get("features")
            
            print(f"[ML] Pipeline extraído: {type(PIPELINE)}")
            print(f"[ML] Threshold: {THRESHOLD}")
            print(f"[ML] Número de features: {len(FEATURES)}")
        else:
            # Se for um objeto direto (não é dicionário)
            PIPELINE = model_data
            THRESHOLD = 0.61
            FEATURES = []
            MODEL_INFO = {}
        
        MODEL_LOADED = True
        print(f"[ML] ✓ Modelo carregado com sucesso!")
        
        # Tentar obter número de features
        try:
            if hasattr(PIPELINE, 'n_features_in_'):
                print(f"[ML] Número de features esperadas: {PIPELINE.n_features_in_}")
            elif hasattr(PIPELINE, 'named_steps'):
                classifier = PIPELINE.named_steps.get('classificador')
                if classifier and hasattr(classifier, 'n_features_in_'):
                    print(f"[ML] Número de features esperadas: {classifier.n_features_in_}")
        except:
            pass
            
    else:
        raise FileNotFoundError(f"Arquivo não encontrado: {MODEL_PATH}")
except Exception as e:
    print(f"[ML] ✗ Erro ao carregar modelo: {e}")
    import traceback
    traceback.print_exc()
    PIPELINE = None
    THRESHOLD = 0.61
    FEATURES = []
    MODEL_INFO = {}
    MODEL_LOADED = False

def predict_delay(features_dict):
    """
    Faz predição de atraso para um pedido
    
    Args:
        features_dict: Dicionário com as features do pedido
    
    Returns:
        dict com predição e probabilidade
    """
    
    if not PIPELINE:
        return {
            'will_delay': False,
            'probability': 0.0,
            'confidence': 0.0,
            'error': 'Modelo não carregado'
        }
    
    try:
        # Construir array com as 62 features esperadas na ordem correta
        X = np.zeros((1, len(EXPECTED_FEATURES)))
        
        for i, feature_name in enumerate(EXPECTED_FEATURES):
            if feature_name in features_dict:
                X[0, i] = features_dict[feature_name]
            else:
                # Se a feature não foi fornecida, usar 0 como padrão
                X[0, i] = 0
        
        # Fazer predição com o pipeline (que inclui o scaler)
        probabilities = PIPELINE.predict_proba(X)
        probability_delay = probabilities[0, 1]  # Probabilidade da classe 1 (atraso)
        
        # Aplicar threshold
        prediction = 1 if probability_delay >= THRESHOLD else 0
        
        return {
            'will_delay': prediction == 1,
            'delay_probability': float(probability_delay),
            'confidence': float(max(probabilities[0])),
            'threshold': float(THRESHOLD)
        }
    
    except Exception as e:
        print(f"[ML] Erro na predição: {e}")
        return {
            'will_delay': False,
            'delay_probability': 0.0,
            'confidence': 0.0,
            'error': str(e)
        }


def get_model_info():
    """Retorna informações do modelo"""
    return {
        'name': 'GradientBoosting + Pipeline',
        'aucpr': 0.2794,
        'rocauc': 0.7814,
        'threshold': round(THRESHOLD, 4),
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'features_count': len(FEATURES),
    }


def process_batch_predictions(data, percentage=100):
    """
    Processa predições em lote para um conjunto de dados usando o MODELO REAL
    
    Args:
        data: lista de listas (cada linha é um exemplo com features + label na última coluna)
        percentage: porcentagem de dados usados (para exibição)
    
    Returns:
        dicionário com resultados e métricas
    """
    
    if not MODEL_LOADED or PIPELINE is None:
        return {
            'error': 'Modelo não carregado',
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'percentage': percentage,
        }
    
    try:
        # Validar dados
        if not data or len(data) == 0:
            return {
                'error': 'Nenhum dado para processar',
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'percentage': percentage,
            }
        
        # Converter para numpy array com tratamento de erros
        try:
            data_array = np.array(data, dtype=float)
        except ValueError as e:
            return {
                'error': f'Erro ao converter dados: {str(e)}. Verifique se todos os valores são numéricos.',
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'percentage': percentage,
            }
        
        # Validar que temos pelo menos 2 colunas (features + label)
        if data_array.ndim != 2 or data_array.shape[1] < 2:
            return {
                'error': f'Formato inválido: esperado matriz 2D com pelo menos 2 colunas, recebido {data_array.shape}',
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'percentage': percentage,
            }
        
        # Separar features e labels
        X = data_array[:, :-1]  # Todas as colunas exceto a última
        y_true = data_array[:, -1].astype(int)  # Última coluna é o label
        
        # Validar dimensões
        if X.shape[0] == 0 or X.shape[1] == 0:
            return {
                'error': 'Dados inválidos: matriz vazia',
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'percentage': percentage,
            }
        
        # Validar número de features
        n_features_expected = None
        try:
            if hasattr(PIPELINE, 'n_features_in_'):
                n_features_expected = PIPELINE.n_features_in_
            elif hasattr(PIPELINE, 'named_steps'):
                classifier = PIPELINE.named_steps.get('classificador')
                if classifier and hasattr(classifier, 'n_features_in_'):
                    n_features_expected = classifier.n_features_in_
        except:
            pass
        
        if n_features_expected and X.shape[1] != n_features_expected:
            return {
                'error': f'Número de features incorreto: esperado {n_features_expected}, recebido {X.shape[1]}',
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'percentage': percentage,
            }
        
        # ============================================
        # USAR O PIPELINE REAL PARA FAZER PREDIÇÕES
        # ============================================
        try:
            print(f"[ML] Fazendo predições com pipeline real...")
            print(f"[ML] Dados: {X.shape[0]} amostras com {X.shape[1]} features")
            
            # Obter probabilidades do pipeline (que inclui o scaler)
            y_proba = PIPELINE.predict_proba(X)[:, 1]
            
            # Aplicar threshold para obter predições binárias
            y_pred = (y_proba >= THRESHOLD).astype(int)
            
            print(f"[ML] ✓ Predições realizadas com sucesso!")
            
        except Exception as e:
            print(f"[ML] ✗ Erro ao fazer predições: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': f'Erro ao fazer predições: {str(e)}',
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'percentage': percentage,
            }
        
        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Preparar detalhes das predições (primeiras 50)
        predictions = []
        for i in range(len(y_true)):
            predictions.append({
                'index': i + 1,
                'actual': int(y_true[i]),
                'predicted': int(y_pred[i]),
                'probability': round(y_proba[i] * 100, 2),
                'correct': int(y_true[i]) == int(y_pred[i]),
            })
        
        print(f"[ML] Métricas calculadas:")
        print(f"[ML]   Accuracy: {accuracy:.4f}")
        print(f"[ML]   Precision: {precision:.4f}")
        print(f"[ML]   Recall: {recall:.4f}")
        print(f"[ML]   F1-Score: {f1:.4f}")
        
        return {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': cm.tolist(),
            'model_params': get_model_info(),
            'predictions': predictions,
            'total_predictions': len(y_true),
            'percentage': percentage,
        }
    
    except Exception as e:
        print(f"[ML] Erro ao processar predições em lote: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': f'Erro ao processar: {str(e)}',
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'percentage': percentage,
        }

