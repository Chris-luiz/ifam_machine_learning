import mlflow
import mlflow.sklearn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from geopy.distance import geodesic
import pickle

mlflow.sklearn.autolog()


order_items_columns = [
    "order_id", 
    "order_item_id",
    "product_id",
    "seller_id",
    "price",
    "freight_value"
]

customers_cols = [
    "customer_id",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state"
]

sellers_cols = [
    "seller_id",
    "seller_zip_code_prefix",
    "seller_city",
    "seller_state"
]

products_cols = [
    "product_id",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "product_category_name",
]

payment_cols = [
    "order_id",
    "payment_sequential",
    "payment_type",
    "payment_installments",
    "payment_value"
]

gelocation_cols = [
    "geolocation_zip_code_prefix",
    "geolocation_lat",
    "geolocation_lng",
    "geolocation_city",
    "geolocation_state"
]


orders_df = pd.read_csv('datasets/olist_orders_dataset.csv')
items_df = pd.read_csv("datasets/olist_order_items_dataset.csv", usecols=order_items_columns)
customer_df = pd.read_csv("datasets/olist_customers_dataset.csv", usecols=customers_cols)
payment_df = pd.read_csv("datasets/olist_order_payments_dataset.csv", usecols=payment_cols)
products_df = pd.read_csv("datasets/olist_products_dataset.csv", usecols=products_cols)
sellers_df = pd.read_csv("datasets/olist_sellers_dataset.csv", usecols=sellers_cols)
geolocation_df = pd.read_csv("datasets/olist_geolocation_dataset.csv", usecols=gelocation_cols)

geolocation_df = geolocation_df.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()


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

items_seller_df = items_df.merge(sellers_df, on="seller_id", how="left")
seller_features = items_seller_df.groupby("order_id").agg({ 
    "seller_state": "first",
    "seller_zip_code_prefix": "first"
}).reset_index()


orders_df = orders_df.merge(items_features, on="order_id", how="left")
orders_df = orders_df.merge(seller_features, on="order_id", how="left")
orders_df = orders_df.merge(customer_df, on='customer_id', how='inner')




########## TRABALHANDO COM GEOLOCATION ####################

orders_df = orders_df.merge(geolocation_df, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
orders_df = orders_df.rename(columns={'geolocation_lat': 'lat_customer', 'geolocation_lng': 'lng_customer'}).drop(columns=['geolocation_zip_code_prefix'])

orders_df = orders_df.merge(geolocation_df, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
orders_df = orders_df.rename(columns={'geolocation_lat': 'lat_seller', 'geolocation_lng': 'lng_seller'}).drop(columns=['geolocation_zip_code_prefix'])

orders_df['lat_customer'] = orders_df['lat_customer'].fillna(orders_df['lat_customer'].mean())
orders_df['lng_customer'] = orders_df['lng_customer'].fillna(orders_df['lng_customer'].mean())
orders_df['lat_seller'] = orders_df['lat_seller'].fillna(orders_df['lat_seller'].mean())
orders_df['lng_seller'] = orders_df['lng_seller'].fillna(orders_df['lng_seller'].mean())

def calcular_distancia_km(row):
    ponto_A = (row['lat_customer'], row['lng_customer'])
    ponto_B = (row['lat_seller'], row['lng_seller'])
    return geodesic(ponto_A, ponto_B).km

orders_df['distancia_km'] = orders_df.apply(calcular_distancia_km, axis=1)

###########################################################







orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])
orders_df["order_purchase_timestamp"] = pd.to_datetime(orders_df["order_purchase_timestamp"])


orders_df["atrasou"] = (orders_df["order_delivered_customer_date"] > orders_df["order_estimated_delivery_date"]).astype(int)

orders_df["dias_estimados_logistica"] = (orders_df["order_estimated_delivery_date"] - orders_df["order_purchase_timestamp"]).dt.days

# 2. Em qual dia da semana era para ser entregue? (Problemas com fim de semana)
orders_df["dia_semana_estimado"] = orders_df["order_estimated_delivery_date"].dt.dayofweek


orders_df["order_purchase_timestamp"] = pd.to_datetime(orders_df["order_purchase_timestamp"])
orders_df["mes_compra"] = orders_df["order_purchase_timestamp"].dt.month
orders_df["dia_semana_compra"] = orders_df["order_purchase_timestamp"].dt.dayofweek
orders_df["hora_compra"] = orders_df["order_purchase_timestamp"].dt.hour

pagamentos_num = payment_df.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_installments': 'max',
    'payment_sequential': 'max'
}).reset_index()

pagamentos_tipo = pd.crosstab(payment_df['order_id'], payment_df['payment_type']).reset_index()

pagamentos_resumo = pagamentos_num.merge(pagamentos_tipo, on='order_id')

orders_df = orders_df.merge(pagamentos_resumo, on='order_id', how='left')

# orders_df['dias_atraso'] = (orders_df['order_delivered_customer_date'] - orders_df['order_estimated_delivery_date']).dt.days
# orders_df['dias_atraso'] = orders_df['dias_atraso'].fillna(orders_df['dias_atraso'].median())

orders_df = orders_df[orders_df['order_status'] == 'delivered'].copy()
orders_df = orders_df.fillna(0)
orders_df['mesmo_estado'] = (orders_df['customer_state'] == orders_df['seller_state']).astype(int)






categoria_atraso_taxa = orders_df.groupby('product_category_name')['atrasou'].mean().reset_index()
categoria_atraso_taxa = categoria_atraso_taxa.rename(columns={'atrasou': 'taxa_atraso_categoria'})

orders_df = orders_df.merge(categoria_atraso_taxa, on='product_category_name', how='left')
orders_df['taxa_atraso_categoria'] = orders_df['taxa_atraso_categoria'].fillna(orders_df['taxa_atraso_categoria'].median())

orders_df['fluxo_logistico'] = orders_df['seller_state'] + "_" + orders_df['customer_state']
orders_df = pd.get_dummies(orders_df, columns=['fluxo_logistico'], dtype=int, drop_first=True)
orders_df = pd.get_dummies(orders_df, columns=['customer_state'], dtype=int)
orders_df = pd.get_dummies(orders_df, columns=['seller_state'], dtype=int)

colunas_estados_customer = [col for col in orders_df.columns if col.startswith('customer_state_')]
colunas_estados_seller = [col for col in orders_df.columns if col.startswith('seller_state_')]
colunas_fluxo = [col for col in orders_df.columns if col.startswith('fluxo_logistico_')]

features = [
    "taxa_atraso_categoria",
    "distancia_km",
    "dias_estimados_logistica", # Nova!
    "dia_semana_estimado",
    "mes_compra",
    "dia_semana_compra",
    "mesmo_estado",
    "hora_compra",
    "price",
    "freight_value",
    "qtde_itens",
    "qtde_produtos",
    "qtde_vendedores",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "payment_value",
    "payment_installments",
    "payment_sequential",
    "boleto",
    "credit_card",
    "voucher",
    "debit_card"
] + colunas_estados_customer + colunas_estados_seller + colunas_fluxo


# tamanho_total = len(orders_df)
# corte = int(tamanho_total * 0.85)

# X_train = X.iloc[:corte]
# y_train = y.iloc[:corte]

# X_test = X.iloc[corte:]
# y_test = y.iloc[corte:]

# df_treino_export = X_train.copy()
# df_treino_export['alvo_real_atrasou'] = y_train
# df_treino_export.to_csv("datasets/dados_treinamento_IA.csv", index=False)
# print("-> CSV de Treinamento exportado (dados_treinamento_IA.csv).")

# df_holdout_export = X_test.copy()
# df_holdout_export['alvo_real_atrasou'] = y_test
# df_holdout_export.to_csv("datasets/holdout_test_django.csv", index=False)
# print("-> CSV de Teste Holdout exportado (holdout_test_django.csv).")


PROD_TREINO_PATH = '/datasets/prod/treino.csv'
PROD_TEST_PATH = '/datasets/prod/test.csv'

prod_dataset = pd.read_csv(PROD_TREINO_PATH)
# y_validation = pd.read_csv(PROD_TEST_PATH)
X = prod_dataset[features]
y = prod_dataset['atrasou']



X_train, X_test, y_train, y_test = train_test_split(prod_dataset, y, random_state=42)

pipeline = Pipeline([
    ('under_sampling', RandomUnderSampler(random_state=42)),
    ('classificador', GradientBoostingClassifier(random_state=42))
])

param_grid_pipeline = [
    {
        'classificador': [GradientBoostingClassifier(random_state=42)],
        'classificador__n_estimators': [100, 200],
        'classificador__learning_rate': [0.05, 0.1],
        'classificador__max_depth': [3, 5]
    },
    {
        'classificador': [RandomForestClassifier(random_state=42)],
        'classificador__n_estimators': [100, 200],
        'classificador__max_depth': [5, 10],
        'classificador__criterion': ['gini', 'entropy']
    }
]

with mlflow.start_run(run_name="Experimento_Multimodelo"):
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid_pipeline,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )

    print("Treinando o GridSearch com Pipeline protegido de vazamento de dados...")
    grid_search.fit(X_train, y_train) # Passamos o X_train ORIGINAL. O Pipeline cuida do UnderSampler internamente!

    melhor_pipeline = grid_search.best_estimator_
    print(f"\nMelhores parâmetros encontrados:\n{grid_search.best_params_}\n")

    y_probabilidades = melhor_pipeline.predict_proba(X_test)[:, 1]
    threshold = 0.65 
    y_pred = (y_probabilidades >= threshold).astype(int)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(conf_matrix)
    
    with open("ml_classifcador_atrasos.pkl", "wb") as f:
        pickle.dump(melhor_pipeline, f)
    
    mlflow.log_metric("teste_f1_score", classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'])



# ==========================================
# CÓDIGO PARA GERAR OS GRÁFICOS DO TRABALHO
# ==========================================

# 1. Gráfico da Matriz de Confusão
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Prazo', 'Atrasou'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão - Modelo Final (Pipeline GB)')
plt.savefig('matriz_confusao_final.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Gráfico da Matriz de Confusão salvo como 'matriz_confusao_final.png'")

# 2. Gráfico de Importância das Features
# Como o classificador está dentro do pipeline, acessamos ele pelo nome do passo
classificador_gb = melhor_pipeline.named_steps['classificador']
importancias = classificador_gb.feature_importances_

# Criamos um DataFrame para ordenar as 10 colunas mais importantes
df_importancias = pd.DataFrame({
    'Feature': features,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Feature', data=df_importancias, palette='viridis')
plt.title('Top 10 Variáveis Mais Importantes para Prever Atrasos')
plt.xlabel('Grau de Importância')
plt.ylabel('Variável')
plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Gráfico de Importância das Variáveis salvo como 'importancia_features.png'")

# {'classificador__learning_rate': 0.05, 'classificador__max_depth': 5, 'classificador__n_estimators': 100}

#               precision    recall  f1-score   support

#            0       0.95      0.88      0.91     22212
#            1       0.25      0.47      0.33      1908

#     accuracy                           0.85     24120
#    macro avg       0.60      0.68      0.62     24120
# weighted avg       0.90      0.85      0.87     24120

# [[19559  2653]
#  [ 1006   902]]
# -> Gráfico da Matriz de Confusão salvo como 'matriz_confusao_final.png'
# /var/www/html/projeto_ifam/atrasos.py:297: FutureWarning: 

# Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

#   sns.barplot(x='Importancia', y='Feature', data=df_importancias, palette='viridis')
# -> Gráfico de Importância das Variáveis salvo como 'importancia_features.png'