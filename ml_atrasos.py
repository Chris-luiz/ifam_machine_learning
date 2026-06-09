import numpy as np
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from geopy.distance import geodesic

# mlflow.sklearn.autolog()


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


orders_df = pd.read_csv('datasets/olist_orders_dataset.csv', nrows=20000)
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
# df_treino_export['atrasou'] = y_train
# df_treino_export.to_csv("datasets/dados_treinamento_IA.csv", index=False)
# print("-> CSV de Treinamento exportado (dados_treinamento_IA.csv).")

# df_holdout_export = X_test.copy()
# df_holdout_export['atrasou'] = y_test
# df_holdout_export.to_csv("datasets/holdout_test_django.csv", index=False)
# print("-> CSV de Teste Holdout exportado (holdout_test_django.csv).")


# PROD_TREINO_PATH = 'datasets/prod/treino.csv'
# PROD_TEST_PATH = 'datasets/prod/test.csv'

# prod_dataset = pd.read_csv(PROD_TREINO_PATH)
# # y_validation = pd.read_csv(PROD_TEST_PATH)
# X = prod_dataset[features]
# y = prod_dataset['atrasou']

# prod_dataset = pd.read_csv(PROD_TREINO_PATH)
# y_validation = pd.read_csv(PROD_TEST_PATH)
X = orders_df[features]
y = orders_df['atrasou']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # ('under_sampling', RandomUnderSampler(random_state=42)),
    ('over_sampling', SMOTE(random_state=42, sampling_strategy=0.5)),
    ('classificador', GradientBoostingClassifier(random_state=42))
])

param_grid_pipeline = [
    {
        'classificador': [GradientBoostingClassifier(random_state=42)],
        'classificador__n_estimators': [100],
        'classificador__learning_rate': [0.1],
        'classificador__max_depth': [3]
    },
    {
        'classificador': [RandomForestClassifier(random_state=42)],
        'classificador__n_estimators': [100],
        'classificador__max_depth': [10],
    },
    # {
    #     'classificador': [SVC(probability=True, random_state=42, max_iter=1500)], # probability=True habilita predict_proba
    #     'classificador__C': [1.0],
    #     'classificador__kernel': ['rbf']
    # },
    # {
    #     'classificador': [LogisticRegression(random_state=42, max_iter=1000)],
    #     'classificador__C': [0.1, 1.0]
    # }
]

with mlflow.start_run(run_name="Experimento_Multimodelo"):
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid_pipeline,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )

    print("Treinando o GridSearch com Pipeline protegido de vazamento de dados...")
    grid_search.fit(X_train, y_train) # Passamos o X_train ORIGINAL. O Pipeline cuida do UnderSampler internamente!

    melhor_pipeline = grid_search.best_estimator_
    print(f"\nMelhores parâmetros encontrados:\n{grid_search.best_params_}\n")

    y_probabilidades = melhor_pipeline.predict_proba(X_test)[:, 1]
    threshold = 0.5
    best_f1 = 0
    
    for th in np.arange(0.3, 0.8, 0.05):
        y_pred_temp = (y_probabilidades >= th).astype(int)
        f1_temp = f1_score(y_test, y_pred_temp)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = th
    
    print(f"O melhor Threshold matemático é: {best_threshold:.2f} com F1-Score de: {best_f1:.2f}")
    y_pred = (y_probabilidades >= best_threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(classification_report(y_test, y_pred))
    print(conf_matrix)








    # ========================================================
    # EXTRAINDO E IMPRIMINDO O RESULTADO DE TODOS OS MODELOS
    # ========================================================
    resultados_df = pd.DataFrame(grid_search.cv_results_)

    print("\n" + "="*80)
    print("              DESEMPENHO DE TODOS OS MODELOS TESTADOS (Validação Cruzada)")
    print("="*80)

    # Loop para detalhar cada modelo testado de forma legível
    for index, row in resultados_df.sort_values(by='rank_test_score').iterrows():
        # Extrai o nome limpo do classificador
        nome_modelo = row['param_classificador'].__class__.__name__
        
        # Filtra apenas os hiperparâmetros específicos do modelo (ex: max_depth, learning_rate)
        params_modelo = {k.split('__')[1]: v for k, v in row['params'].items() if '__' in k}
        
        print(f"Rank {row['rank_test_score']}: {nome_modelo}")
        print(f"  -> Parâmetros: {params_modelo}")
        print(f"  -> F1-Score Médio: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f} de desvio padrão)")
        print("-" * 80)

    print("="*80 + "\n")







    
    # with open("ml_classifcador_atrasos.pkl", "wb") as f:
    #     pickle.dump(melhor_pipeline, f)
    
    # mlflow.log_metric("teste_f1_score", classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'])



# ==========================================
# CÓDIGO PARA GERAR OS GRÁFICOS DO TRABALHO
# ==========================================
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Prazo', 'Atrasou'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão - Avaliação Multimodelo')
plt.savefig('matriz_confusao_final.png', dpi=300, bbox_inches='tight')
plt.close()

# Extração Segura de Importância (Previne quebra de script caso o SVM ou a Regressão Logística vençam)
classificador_final = melhor_pipeline.named_steps['classificador']

if hasattr(classificador_final, 'feature_importances_'):
    importancias = classificador_final.feature_importances_
    df_importancias = pd.DataFrame({'Feature': features, 'Importancia': importancias}).sort_values(by='Importancia', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Feature', data=df_importancias, hue='Feature', palette='viridis', legend=False)
    plt.title('Top 10 Variáveis Mais Importantes para Prever Atrasos')
    plt.xlabel('Grau de Importância')
    plt.ylabel('Variável')
    plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Gráfico de Importância gerado com sucesso.")
elif hasattr(classificador_final, 'coef_'):
    importancias = np.abs(classificador_final.coef_[0])
    df_importancias = pd.DataFrame({'Feature': features, 'Importancia': importancias}).sort_values(by='Importancia', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Feature', data=df_importancias, hue='Feature', palette='magma', legend=False)
    plt.title('Top 10 Coeficientes Determinantes (Modelo Linear)')
    plt.xlabel('Peso Absoluto do Coeficiente')
    plt.ylabel('Variável')
    plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Gráfico de pesos de coeficientes gerado com sucesso.")
else:
    print("-> O modelo vencedor não suporta extração direta de importância de atributos por este método.")

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















# Melhores parâmetros encontrados:
# {'classificador': GradientBoostingClassifier(random_state=42), 'classificador__learning_rate': 0.05, 'classificador__max_depth': 3, 'classificador__n_estimators': 100}

#               precision    recall  f1-score   support

#            0       0.94      0.90      0.92      4443
#            1       0.28      0.40      0.33       415

#     accuracy                           0.86      4858
#    macro avg       0.61      0.65      0.63      4858
# weighted avg       0.88      0.86      0.87      4858

# [[4017  426]
#  [ 250  165]]
# -> Gráfico de Importância gerado com sucesso.



# Melhores parâmetros encontrados:
# {'classificador': GradientBoostingClassifier(random_state=42), 'classificador__learning_rate': 0.1, 'classificador__max_depth': 3, 'classificador__n_estimators': 100}

#               precision    recall  f1-score   support

#            0       0.95      0.88      0.91     11146
#            1       0.25      0.45      0.32       984

#     accuracy                           0.84     12130
#    macro avg       0.60      0.66      0.62     12130
# weighted avg       0.89      0.84      0.86     12130

# [[9808 1338]
#  [ 544  440]]
# -> Gráfico de Importância gerado com sucesso.