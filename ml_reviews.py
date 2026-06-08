import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score

# Ativa o tracking automático do MLflow
mlflow.sklearn.autolog()

# ==========================================
# 1. DOWNLOAD DE RECURSOS E CONFIGURAÇÕES
# ==========================================
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

stopwords_pt = stopwords.words('portuguese')
stopwords_pt.extend([
    'produto', 'comprei', 'veio', 'loja', 'prazo', 
    'dia', 'dias', 'entregue', 'entrega', 'recebi'
])

# ==========================================
# 2. CARGA E ENGENHARIA DE ATRIBUTOS HÍBRIDA
# ==========================================
print("Carregando bases de dados do Olist...")
orders_df = pd.read_csv('datasets/olist_orders_dataset.csv')
reviews_df = pd.read_csv('datasets/olist_order_reviews_dataset.csv')
items_df = pd.read_csv("datasets/olist_order_items_dataset.csv")

# Tratamento e unificação do Texto das Avaliações
print("Processando dados textuais (NLP)...")
reviews_df['review_comment_title'] = reviews_df['review_comment_title'].fillna("")
reviews_df['review_comment_message'] = reviews_df['review_comment_message'].fillna("")
reviews_df['avaliacao_completa'] = (reviews_df['review_comment_title'] + " " + reviews_df['review_comment_message']).str.strip()

# Deduplicação rigorosa para evitar vazamento de dados por pedido
reviews_df = reviews_df.drop_duplicates(subset=['order_id'], keep='last')

# Agregação financeira dos itens do pedido
items_resumo = items_df.groupby('order_id').agg({
    'price': 'sum',
    'freight_value': 'sum'
}).reset_index()

# Cruzamento da Tabela Mestra
df_master = orders_df.merge(reviews_df, on="order_id", how="inner")
df_master = df_master.merge(items_resumo, on="order_id", how="inner")

# Engenharia Logística (O que causa a experiência do cliente)
print("Gerando variáveis temporais e de atraso...")
df_master['order_delivered_customer_date'] = pd.to_datetime(df_master['order_delivered_customer_date'])
df_master['order_estimated_delivery_date'] = pd.to_datetime(df_master['order_estimated_delivery_date'])
df_master['order_purchase_timestamp'] = pd.to_datetime(df_master['order_purchase_timestamp'])

# Dias de atraso real (valores positivos = atrasou)
df_master['dias_atraso'] = (df_master['order_delivered_customer_date'] - df_master['order_estimated_delivery_date']).dt.days
df_master['dias_atraso'] = df_master['dias_atraso'].fillna(0)

# Tempo total que levou para chegar na casa do cliente
df_master['tempo_transito_real'] = (df_master['order_delivered_customer_date'] - df_master['order_purchase_timestamp']).dt.days
df_master['tempo_transito_real'] = df_master['tempo_transito_real'].fillna(df_master['tempo_transito_real'].median())

# Definição do Target Binário (Padrão Acadêmico)
df_master['target_sentimento'] = df_master['review_score'].apply(lambda nota: 1 if nota > 3 else 0)

# AMOSTRAGEM SEGURA E CONTROLADA (Evita estouro de Memória / Killed)
print("Aplicando amostragem estratificada para controle de hardware...")
df_master = df_master.sample(n=35000, random_state=42).copy()

# ==========================================
# 3. DIVISÃO DOS DATASETS (TREINO, VALIDAÇÃO, TESTE)
# ==========================================
features_numericas = ['price', 'freight_value', 'dias_atraso', 'tempo_transito_real']
feature_textual = 'avaliacao_completa'

X = df_master[features_numericas + [feature_textual]]
y = df_master['target_sentimento']

# Divisão em Treino (80%) e Teste Holdout (20%) com estratificação do target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 4. CONSTRUÇÃO DO PIPELINE E TRANSFORMADORES
# ==========================================
# O ColumnTransformer processa o texto e os números de forma isolada e os une no final
processador_colunas = ColumnTransformer(
    transformers=[
        ('nlp_texto', TfidfVectorizer(stop_words=stopwords_pt, max_features=600, ngram_range=(1, 2)), feature_textual),
        ('scaler_num', StandardScaler(), features_numericas)
    ]
)

pipeline_final = Pipeline([
    ('pre_processamento', processador_colunas),
    ('classificador', LogisticRegression(random_state=42, max_iter=1000))
])

# Grade Multimodelo para o GridSearch
param_grid = [
    {
        'classificador': [LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')],
        'classificador__C': [0.1, 1.0]
    },
    {
        'classificador': [GradientBoostingClassifier(random_state=42)],
        'classificador__n_estimators': [100],
        'classificador__learning_rate': [0.1]
    }
]

# ==========================================
# 5. TREINAMENTO COM VALIDAÇÃO CRUZADA E MLFLOW
# ==========================================
print("Iniciando a busca pelo melhor modelo de satisfação no GridSearch...")
with mlflow.start_run(run_name="Experimento_Reviews_Olist"):
    
    grid_search = GridSearchCV(
        estimator=pipeline_final,
        param_grid=param_grid,
        cv=4, # Validação Cruzada de 4 Folds
        scoring='f1_macro',
        n_jobs=2, # Paralelismo seguro para não travar a máquina
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    melhor_pipeline = grid_search.best_estimator_
    print(f"\nMelhores parâmetros encontrados: {grid_search.best_params_}\n")
    
    # Avaliação Externa (Dataset de Teste)
    y_pred = melhor_pipeline.predict(X_test)
    
    # Métricas
    f1_final = f1_score(y_test, y_pred, average='macro')
    print("--- RELATÓRIO DE CLASSIFICAÇÃO DA BANCA ---")
    print(classification_report(y_test, y_pred, target_names=['Ruim (1-3)', 'Bom (4-5)']))
    
    mlflow.log_metric("holdout_f1_macro", f1_final)
    
    # Salva o arquivo binário do pipeline treinado
    os.makedirs("machinelearning/ml_models", exist_ok=True)
    with open("machinelearning/ml_models/classificador_reviews.pkl", "wb") as f:
        pickle.dump(melhor_pipeline, f)
    print("Modelo salvo com sucesso!")

# ==========================================
# 6. EXPORTAÇÃO DOS GRÁFICOS DO TRABALHO
# ==========================================
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ruim (1-3)', 'Bom (4-5)'])
disp.plot(cmap='YlGnBu', values_format='d')
plt.title('Matriz de Confusão - Modelo de Sentimento/Reviews')
plt.savefig('matriz_confusao_reviews.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico 'matriz_confusao_reviews.png' gerado para os slides da apresentação.")