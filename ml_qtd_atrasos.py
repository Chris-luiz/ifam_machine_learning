import os
import datetime
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from geopy.distance import geodesic

# Ativa o monitoramento automatizado do MLflow para modelos de Regressão
mlflow.sklearn.autolog()

# ==========================================
# 1. CARGA E PREPARAÇÃO DOS DADOS (PADRÃO OLIST)
# ==========================================

order_items_columns = ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"]
customers_cols = ["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"]
sellers_cols = ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]
products_cols = ["product_id", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm", "product_category_name"]
payment_cols = ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"]
gelocation_cols = ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]

print("Carregando os datasets originais...")
orders_df = pd.read_csv('datasets/olist_orders_dataset.csv')
items_df = pd.read_csv("datasets/olist_order_items_dataset.csv", usecols=order_items_columns)
customer_df = pd.read_csv("datasets/olist_customers_dataset.csv", usecols=customers_cols)
payment_df = pd.read_csv("datasets/olist_order_payments_dataset.csv", usecols=payment_cols)
products_df = pd.read_csv("datasets/olist_products_dataset.csv", usecols=products_cols)
sellers_df = pd.read_csv("datasets/olist_sellers_dataset.csv", usecols=sellers_cols)
geolocation_df = pd.read_csv("datasets/olist_geolocation_dataset.csv", usecols=gelocation_cols)

# Otimização de memória para Geolocalização (Agrupamento por Prefixo de CEP)
print("Processando geolocalização...")
geolocation_df = geolocation_df.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()

# Cruzamentos Estruturais (Agregações por Pedido)
items_prod_df = items_df.merge(products_df, on="product_id", how="left")
items_features = items_prod_df.groupby("order_id").agg({
    "price": "sum",
    "freight_value": "sum",
    "order_item_id": "count",
    "product_weight_g": "sum",
    "product_length_cm": "sum",
    "product_height_cm": "sum",
    "product_width_cm": "sum",
    "seller_id": "nunique",
    "product_id": "nunique",
    "product_category_name": "first"
}).reset_index()

items_features = items_features.rename(columns={
    "order_item_id": "qtde_itens",
    "seller_id": "qtde_vendedores",
    "product_id": "qtde_produtos"
})

seller_features = items_df.merge(sellers_df, on="seller_id", how="left").groupby("order_id").agg({ 
    "seller_state": "first",
    "seller_zip_code_prefix": "first"
}).reset_index()

# Unificação da Tabela Mestra (Apenas Pedidos Entregues para analisar histórico)
orders_df = orders_df[orders_df['order_status'] == 'delivered'].copy()
orders_df = orders_df.sample(n=30000, random_state=42).copy()
orders_df = orders_df.merge(items_features, on="order_id", how="left")
orders_df = orders_df.merge(seller_features, on="order_id", how="left")
orders_df = orders_df.merge(customer_df, on='customer_id', how='inner')

# Cálculo de Coordenadas Geográficas
orders_df = orders_df.merge(geolocation_df, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
orders_df = orders_df.rename(columns={'geolocation_lat': 'lat_customer', 'geolocation_lng': 'lng_customer'}).drop(columns=['geolocation_zip_code_prefix'])

orders_df = orders_df.merge(geolocation_df, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
orders_df = orders_df.rename(columns={'geolocation_lat': 'lat_seller', 'geolocation_lng': 'lng_seller'}).drop(columns=['geolocation_zip_code_prefix'])

# Imputação pela Média para evitar perdas de linhas
orders_df['lat_customer'] = orders_df['lat_customer'].fillna(orders_df['lat_customer'].mean())
orders_df['lng_customer'] = orders_df['lng_customer'].fillna(orders_df['lng_customer'].mean())
orders_df['lat_seller'] = orders_df['lat_seller'].fillna(orders_df['lat_seller'].mean())
orders_df['lng_seller'] = orders_df['lng_seller'].fillna(orders_df['lng_seller'].mean())

print("Calculando distâncias geodésicas (KM) entre clientes e vendedores...")
def calcular_distancia_km(row):
    return geodesic((row['lat_customer'], row['lng_customer']), (row['lat_seller'], row['lng_seller'])).km

orders_df['distancia_km'] = orders_df.apply(calcular_distancia_km, axis=1)

# ==========================================
# 2. ENGENHARIA DE ATRIBUTOS PARA REGRESSÃO (KAGGLE HIGHEST SCORES)
# ==========================================

orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])
orders_df["order_purchase_timestamp"] = pd.to_datetime(orders_df["order_purchase_timestamp"])

# DEFINIÇÃO DA VARIÁVEL ALVO (TARGET CONTÍNUO): Dias reais de desvio do prazo.
# Valores positivos significam atraso real. Valores negativos/zero significam entrega adiantada.
orders_df['dias_desvio_entrega'] = (orders_df['order_delivered_customer_date'] - orders_df['order_estimated_delivery_date']).dt.days

# Feature de apoio categórica apenas para gerar o Target Encoding estável
orders_df["atrasou_binario"] = (orders_df["order_delivered_customer_date"] > orders_df["order_estimated_delivery_date"]).astype(int)

orders_df["dias_estimados_logistica"] = (orders_df["order_estimated_delivery_date"] - orders_df["order_purchase_timestamp"]).dt.days
orders_df["dia_semana_estimado"] = orders_df["order_estimated_delivery_date"].dt.dayofweek
orders_df["mes_compra"] = orders_df["order_purchase_timestamp"].dt.month
orders_df["dia_semana_compra"] = orders_df["order_purchase_timestamp"].dt.dayofweek
orders_df["hora_compra"] = orders_df["order_purchase_timestamp"].dt.hour
orders_df['mesmo_estado'] = (orders_df['customer_state'] == orders_df['seller_state']).astype(int)

# Tratamento dos Dados de Pagamento
pagamentos_num = payment_df.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_installments': 'max',
    'payment_sequential': 'max'
}).reset_index()
pagamentos_tipo = pd.crosstab(payment_df['order_id'], payment_df['payment_type']).reset_index()
pagamentos_resumo = pagamentos_num.merge(pagamentos_tipo, on='order_id')
orders_df = orders_df.merge(pagamentos_resumo, on='order_id', how='left')

# Preenchimento de nulos estruturais
orders_df = orders_df.fillna(0)

# Target Encoding estável baseado no comportamento histórico das categorias de produtos
categoria_atraso_taxa = orders_df.groupby('product_category_name')['atrasou_binario'].mean().reset_index()
categoria_atraso_taxa = categoria_atraso_taxa.rename(columns={'atrasou_binario': 'taxa_atraso_categoria'})
orders_df = orders_df.merge(categoria_atraso_taxa, on='product_category_name', how='left')
orders_df['taxa_atraso_categoria'] = orders_df['taxa_atraso_categoria'].fillna(orders_df['taxa_atraso_categoria'].median())

# One-Hot Encoding das Variáveis Categóricas e Fluxos Geográficos
orders_df['fluxo_logistico'] = orders_df['seller_state'] + "_" + orders_df['customer_state']
orders_df = pd.get_dummies(orders_df, columns=['fluxo_logistico'], dtype='int8', drop_first=True)
orders_df = pd.get_dummies(orders_df, columns=['customer_state'], dtype='int8')
orders_df = pd.get_dummies(orders_df, columns=['seller_state'], dtype='int8')

# Filtro das listas dinâmicas de colunas binárias
colunas_estados_customer = [col for col in orders_df.columns if col.startswith('customer_state_')]
colunas_estados_seller = [col for col in orders_df.columns if col.startswith('seller_state_')]
colunas_fluxo = [col for col in orders_df.columns if col.startswith('fluxo_logistico_')]

# Mapeamento do Vetor Final de Features
features = [
    "taxa_atraso_categoria", "distancia_km", "dias_estimados_logistica", "dia_semana_estimado",
    "mes_compra", "dia_semana_compra", "mesmo_estado", "hora_compra", "price", "freight_value",
    "qtde_itens", "qtde_produtos", "qtde_vendedores", "product_weight_g", "product_length_cm",
    "product_height_cm", "product_width_cm", "payment_value", "payment_installments", "payment_sequential"
]
# Garante a existência das colunas de pagamento padrões no vetor
for col_pay in ['boleto', 'credit_card', 'voucher', 'debit_card']:
    if col_pay in orders_df.columns:
        features.append(col_pay)

features += colunas_estados_customer + colunas_estados_seller + colunas_fluxo

# Separação das Matrizes
X = orders_df[features]
y = orders_df['dias_desvio_entrega']

# Divisão em Dataset de Treino e Dataset de Validação Externa (Holdout Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# ==========================================
# 3. PIPELINE E MODELAGEM MULTI-REGRESSÃO COM CROSS-VALIDATION
# ==========================================

# Pipeline Robusto: Escalonamento + Regressor
pipeline_regressao = Pipeline([
    ('escalonador', StandardScaler()),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Grade de Hiperparâmetros para os Melhores Modelos de Regressão do Mercado
param_grid_pipeline = [
    {
        'regressor': [GradientBoostingRegressor(random_state=42)],
        'regressor__n_estimators': [100, 150],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [4, 6]
    },
    {
        'regressor': [RandomForestRegressor(random_state=42)],
        'regressor__n_estimators': [100],
        'regressor__max_depth': [8, 12]
    },
    {
        'regressor': [Ridge(random_state=42)],
        'regressor__alpha': [1.0, 10.0]
    }
]

print("Iniciando bateria de testes multi-regressão gerenciada pelo MLflow...")
with mlflow.start_run(run_name="Experimento_Regressao_Olist"):
    
    grid_search_reg = GridSearchCV(
        estimator=pipeline_regressao,
        param_grid=param_grid_pipeline,
        cv=5, # Validação Cruzada K-Fold em 5 partes
        scoring='neg_mean_absolute_error', # Foco em mitigar o Erro Absoluto em Dias
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_reg.fit(X_train, y_train)
    
    melhor_modelo_pipeline = grid_search_reg.best_estimator_
    print(f"\nMelhor Configuração Encontrada:\n{grid_search_reg.best_params_}\n")
    
    # Execução das predições no conjunto de teste isolado
    y_pred = melhor_modelo_pipeline.predict(X_test)
    
    # Cálculo das Métricas de Regressão para a Banca
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("--- Relatório Estatístico Final (Holdout) ---")
    print(f"Erro Médio Absoluto (MAE): {mae:.2f} dias")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f} dias")
    print(f"Coeficiente de Determinação (R² Score): {r2:.4f}")
    
    # Salvando as métricas customizadas explicitamente no Tracking do MLflow
    mlflow.log_metric("holdout_mae", mae)
    mlflow.log_metric("holdout_rmse", rmse)
    mlflow.log_metric("holdout_r2", r2)
    
    # Persistência Nativa do Modelo Completo de Regressão via Pickle
    with open("machinelearning/ml_models/regressor_dias_atraso.pkl", "wb") as f:
        pickle.dump(melhor_modelo_pipeline, f)
    print("Pipeline de Regressão persistido com sucesso em ml_models!")

# ==========================================
# 4. CRIAÇÃO DOS GRÁFICOS ACADÊMICOS DE AVALIAÇÃO
# ==========================================

# Gráfico 1: Predição vs. Realidade (Scatter Plot de Resíduos)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Predição Perfeita")
plt.title('Validação do Regressor: Dias Reais vs. Dias Previstos')
plt.xlabel('Dias de Desvio Reais (Histórico Olist)')
plt.ylabel('Dias de Desvio Previstos pelo Modelo')
plt.legend()
plt.grid(True, linestyle=':')
plt.savefig('grafico_regressao_residuos.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico de análise de resíduos exportado como 'grafico_regressao_residuos.png'")

# Gráfico 2: Importância de Atributos no Erro do Prazo
regressor_final = melhor_modelo_pipeline.named_steps['regressor']
if hasattr(regressor_final, 'feature_importances_'):
    importancias_reg = regressor_final.feature_importances_
    df_imp_reg = pd.DataFrame({
        'Feature': features,
        'Importancia': importancias_reg
    }).sort_values(by='Importancia', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Feature', data=df_imp_reg, hue='Feature', palette='flare', legend=False)
    plt.title('Top 10 Atributos Determinantes no Tempo de Entrega')
    plt.xlabel('Grau de Importância Relativa')
    plt.ylabel('Atributo')
    plt.savefig('importancia_features_regressao.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico de importância de variáveis exportado como 'importancia_features_regressao.png'")