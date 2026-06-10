import numpy as np
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    f1_score, precision_recall_curve, auc, roc_auc_score
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from geopy.distance import geodesic
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_STRATEGY = 0.4   # minoritária chegará a 40% da majoritária
CV_FOLDS = 5           # StratifiedKFold — mantém proporção em cada fold

# ============================================================
# 1. COLUNAS SELECIONADAS
# ============================================================
order_items_columns = [
    "order_id", "order_item_id", "product_id",
    "seller_id", "price", "freight_value"
]
customers_cols = [
    "customer_id", "customer_unique_id",
    "customer_zip_code_prefix", "customer_city", "customer_state"
]
sellers_cols = ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]
products_cols = [
    "product_id", "product_weight_g", "product_length_cm",
    "product_height_cm", "product_width_cm", "product_category_name",
]
payment_cols = [
    "order_id", "payment_sequential", "payment_type",
    "payment_installments", "payment_value"
]
gelocation_cols = [
    "geolocation_zip_code_prefix", "geolocation_lat",
    "geolocation_lng", "geolocation_city", "geolocation_state"
]

# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================
print("Carregando datasets...")
orders_df    = pd.read_csv('datasets/olist_orders_dataset.csv')
items_df     = pd.read_csv("datasets/olist_order_items_dataset.csv", usecols=order_items_columns)
customer_df  = pd.read_csv("datasets/olist_customers_dataset.csv",   usecols=customers_cols)
payment_df   = pd.read_csv("datasets/olist_order_payments_dataset.csv", usecols=payment_cols)
products_df  = pd.read_csv("datasets/olist_products_dataset.csv",    usecols=products_cols)
sellers_df   = pd.read_csv("datasets/olist_sellers_dataset.csv",     usecols=sellers_cols)
geolocation_df = pd.read_csv("datasets/olist_geolocation_dataset.csv", usecols=gelocation_cols)

# ============================================================
# 3. FEATURE ENGINEERING (sem target encoding ainda)
# ============================================================
print("Construindo features...")

# --- Geolocalização: média por CEP ---
geolocation_df = geolocation_df.groupby('geolocation_zip_code_prefix').agg(
    geolocation_lat=('geolocation_lat', 'mean'),
    geolocation_lng=('geolocation_lng', 'mean')
).reset_index()

# --- Itens + Produtos ---
items_prod_df = items_df.merge(products_df, on="product_id", how="left")
items_features = items_prod_df.groupby("order_id").agg(
    price=("price", "sum"),
    freight_value=("freight_value", "sum"),
    qtde_itens=("order_item_id", "count"),
    product_weight_g=("product_weight_g", "sum"),
    product_length_cm=("product_length_cm", "sum"),
    product_height_cm=("product_height_cm", "sum"),
    product_width_cm=("product_width_cm", "sum"),
    qtde_vendedores=("seller_id", "nunique"),
    qtde_produtos=("product_id", "nunique"),
    product_category_name=("product_category_name", "first")
).reset_index()

# --- Vendedores ---
items_seller_df = items_df.merge(sellers_df, on="seller_id", how="left")
seller_features = items_seller_df.groupby("order_id").agg(
    seller_state=("seller_state", "first"),
    seller_zip_code_prefix=("seller_zip_code_prefix", "first")
).reset_index()

# --- Merge principal ---
orders_df = orders_df.merge(items_features,  on="order_id",    how="left")
orders_df = orders_df.merge(seller_features, on="order_id",    how="left")
orders_df = orders_df.merge(customer_df,     on='customer_id', how='inner')

# --- Geolocalização: distância comprador <-> vendedor ---
orders_df = orders_df.merge(
    geolocation_df, left_on='customer_zip_code_prefix',
    right_on='geolocation_zip_code_prefix', how='left'
).rename(columns={
    'geolocation_lat': 'lat_customer',
    'geolocation_lng': 'lng_customer'
}).drop(columns=['geolocation_zip_code_prefix'])

orders_df = orders_df.merge(
    geolocation_df, left_on='seller_zip_code_prefix',
    right_on='geolocation_zip_code_prefix', how='left'
).rename(columns={
    'geolocation_lat': 'lat_seller',
    'geolocation_lng': 'lng_seller'
}).drop(columns=['geolocation_zip_code_prefix'])

for col in ['lat_customer', 'lng_customer', 'lat_seller', 'lng_seller']:
    orders_df[col] = orders_df[col].fillna(orders_df[col].mean())

def calcular_distancia_km(row):
    return geodesic(
        (row['lat_customer'], row['lng_customer']),
        (row['lat_seller'],   row['lng_seller'])
    ).km

orders_df['distancia_km'] = orders_df.apply(calcular_distancia_km, axis=1)

# --- Datas e features temporais ---
orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])
orders_df['order_purchase_timestamp']      = pd.to_datetime(orders_df['order_purchase_timestamp'])

orders_df["atrasou"] = (
    orders_df["order_delivered_customer_date"] > orders_df["order_estimated_delivery_date"]
).astype(int)

orders_df["dias_estimados_logistica"] = (
    orders_df["order_estimated_delivery_date"] - orders_df["order_purchase_timestamp"]
).dt.days

orders_df["dia_semana_estimado"] = orders_df["order_estimated_delivery_date"].dt.dayofweek
orders_df["mes_compra"]          = orders_df["order_purchase_timestamp"].dt.month
orders_df["dia_semana_compra"]   = orders_df["order_purchase_timestamp"].dt.dayofweek
orders_df["hora_compra"]         = orders_df["order_purchase_timestamp"].dt.hour

# --- Pagamentos ---
pagamentos_num = payment_df.groupby('order_id').agg(
    payment_value=("payment_value", "sum"),
    payment_installments=("payment_installments", "max"),
    payment_sequential=("payment_sequential", "max")
).reset_index()

pagamentos_tipo = pd.crosstab(payment_df['order_id'], payment_df['payment_type']).reset_index()
pagamentos_resumo = pagamentos_num.merge(pagamentos_tipo, on='order_id')
orders_df = orders_df.merge(pagamentos_resumo, on='order_id', how='left')

# --- Filtro: apenas pedidos entregues ---
orders_df = orders_df[orders_df['order_status'] == 'delivered'].copy()
orders_df = orders_df.fillna(0)

# --- Feature: mesmo estado e fluxo logístico ---
orders_df['mesmo_estado'] = (
    orders_df['customer_state'] == orders_df['seller_state']
).astype(int)

orders_df['fluxo_logistico'] = orders_df['seller_state'] + "_" + orders_df['customer_state']

# --- One-hot encoding de estados e fluxo ---
orders_df = pd.get_dummies(orders_df, columns=['fluxo_logistico'], dtype=int, drop_first=True)
orders_df = pd.get_dummies(orders_df, columns=['customer_state'], dtype=int)
orders_df = pd.get_dummies(orders_df, columns=['seller_state'],   dtype=int)

colunas_estados_customer = [c for c in orders_df.columns if c.startswith('customer_state_')]
colunas_estados_seller   = [c for c in orders_df.columns if c.startswith('seller_state_')]

# NOTA: 'taxa_atraso_categoria' e colunas_fluxo são calculadas DEPOIS do split
#       para evitar data leakage.

features_base = [
    "distancia_km",
    "dias_estimados_logistica",
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
] + colunas_estados_customer + colunas_estados_seller

# ============================================================
# 4. SPLIT ESTRATIFICADO — ANTES de qualquer target encoding
# ============================================================
print("\nSplit estratificado treino/teste...")

X = orders_df[features_base + ['product_category_name']]
y = orders_df['atrasou']

print(f"  Total de registros: {len(y)}")
print(f"  Atrasos (1): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  No prazo (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y          # <-- garante mesma proporção em treino e teste
)

# ============================================================
# 5. TARGET ENCODING SEM LEAKAGE
#    A taxa de atraso por categoria é calculada APENAS com treino
#    e depois aplicada ao teste via merge — sem ver o futuro.
# ============================================================
print("Calculando target encoding (sem leakage)...")

taxa_treino = (
    X_train[['product_category_name']].join(y_train)
    .groupby('product_category_name')['atrasou']
    .mean()
    .reset_index()
    .rename(columns={'atrasou': 'taxa_atraso_categoria'})
)

mediana_global = taxa_treino['taxa_atraso_categoria'].median()

X_train = X_train.merge(taxa_treino, on='product_category_name', how='left')
X_test  = X_test.merge(taxa_treino,  on='product_category_name', how='left')

X_train['taxa_atraso_categoria'] = X_train['taxa_atraso_categoria'].fillna(mediana_global)
X_test['taxa_atraso_categoria']  = X_test['taxa_atraso_categoria'].fillna(mediana_global)

# Remove coluna de texto (não entra no modelo)
X_train = X_train.drop(columns=['product_category_name'])
X_test  = X_test.drop(columns=['product_category_name'])

features_final = features_base + ['taxa_atraso_categoria']

X_train = X_train[features_final]
X_test  = X_test[features_final]

# ============================================================
# 6. RATIO DE DESBALANCEAMENTO (usado no XGBoost e sample_weight)
# ============================================================
ratio_classes = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Ratio negativo/positivo: {ratio_classes:.1f}x")

# ============================================================
# 7. MODELOS
#    Cada modelo tem sua estratégia de balanceamento mais adequada:
#    - GradientBoosting: sample_weight (nativo, sem SMOTE)
#    - RandomForest: class_weight='balanced'
#    - LogisticRegression: class_weight='balanced'
#    - XGBoost (se disponível): scale_pos_weight
# ============================================================

# --- Scaler compartilhado ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# sample_weight para GradientBoosting
sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)

modelos = {}

print("\nTreinando modelos...")

# ---- 7a. GradientBoostingClassifier com sample_weight ----
print("  [1/4] GradientBoostingClassifier...")
gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,        # reduz overfitting
    random_state=RANDOM_STATE
)
gbm.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
modelos['GradientBoosting'] = gbm

# ---- 7b. RandomForestClassifier com class_weight ----
print("  [2/4] RandomForestClassifier...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',   # pesa a minoritária automaticamente
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf.fit(X_train_scaled, y_train)
modelos['RandomForest'] = rf

# ---- 7c. LogisticRegression com class_weight ----
print("  [3/4] LogisticRegression...")
lr = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=2000,
    random_state=RANDOM_STATE
)
lr.fit(X_train_scaled, y_train)
modelos['LogisticRegression'] = lr

# ---- 7d. XGBoost (opcional — instala com: pip install xgboost) ----
try:
    from xgboost import XGBClassifier
    print("  [4/4] XGBoostClassifier...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=ratio_classes,  # equivalente ao class_weight para XGB
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',             # AUC da curva PR — ideal para desbalanceado
        random_state=RANDOM_STATE,
        verbosity=0
    )
    xgb.fit(X_train_scaled, y_train)
    modelos['XGBoost'] = xgb
    print("     XGBoost carregado com sucesso.")
except ImportError:
    print("  [4/4] XGBoost não instalado (pip install xgboost). Pulando.")

# ============================================================
# 8. AVALIAÇÃO COM THRESHOLD ÓTIMO VIA CURVA PRECISION-RECALL
#    Critério de negócio: recall >= 0.60 com maior precision possível.
#    Isso significa: "prefiro identificar 60% dos atrasos mesmo que
#    com algum falso positivo, a perder muitos atrasos reais."
# ============================================================
RECALL_MINIMO = 0.55   # ajuste conforme o custo de negócio

print("\n" + "="*80)
print("       AVALIAÇÃO FINAL — CURVA PRECISION-RECALL + THRESHOLD ÓTIMO")
print("="*80)

resultados = {}

for nome, modelo in modelos.items():
    probs = modelo.predict_proba(X_test_scaled)[:, 1]

    # AUC-PR: métrica mais confiável para desbalanceamento
    precisoes, recalls, thresholds = precision_recall_curve(y_test, probs)
    aucpr = auc(recalls, precisoes)
    rocauc = roc_auc_score(y_test, probs)

    # Encontra o threshold com recall >= RECALL_MINIMO e maior precision
    candidatos = [
        (p, r, t)
        for p, r, t in zip(precisoes[:-1], recalls[:-1], thresholds)
        if r >= RECALL_MINIMO
    ]

    if candidatos:
        # Dentre os candidatos com recall suficiente, pega o de maior precision
        best_precision, best_recall, best_threshold = max(candidatos, key=lambda x: x[0])
    else:
        # Fallback: threshold que maximiza F1 puro
        f1_scores = 2 * precisoes[:-1] * recalls[:-1] / (precisoes[:-1] + recalls[:-1] + 1e-9)
        idx = np.argmax(f1_scores)
        best_threshold = thresholds[idx]
        best_precision = precisoes[idx]
        best_recall    = recalls[idx]

    y_pred = (probs >= best_threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    resultados[nome] = {
        'modelo': modelo,
        'probs': probs,
        'y_pred': y_pred,
        'threshold': best_threshold,
        'f1': f1,
        'aucpr': aucpr,
        'rocauc': rocauc,
        'precision': best_precision,
        'recall': best_recall
    }

    print(f"\n{'─'*60}")
    print(f"  Modelo: {nome}")
    print(f"  AUC-PR : {aucpr:.4f}  |  ROC-AUC: {rocauc:.4f}")
    print(f"  Threshold escolhido: {best_threshold:.2f}")
    print(f"  Precision: {best_precision:.3f}  |  Recall: {best_recall:.3f}  |  F1: {f1:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Prazo', 'Atrasou'])}")

# ============================================================
# 9. MELHOR MODELO
# ============================================================
melhor_nome = max(resultados, key=lambda k: resultados[k]['aucpr'])
melhor = resultados[melhor_nome]

print("="*80)
print(f"  MELHOR MODELO (por AUC-PR): {melhor_nome}")
print(f"  AUC-PR = {melhor['aucpr']:.4f}  |  F1 = {melhor['f1']:.4f}")
print("="*80 + "\n")

# ============================================================
# 10. SALVAR O MELHOR PIPELINE COMPLETO
# ============================================================
from sklearn.pipeline import Pipeline as SKPipeline

pipeline_final = SKPipeline([
    ('scaler', scaler),
    ('classificador', melhor['modelo'])
])

with open("ml_classificador_atrasos_v2.pkl", "wb") as f:
    pickle.dump({
        'pipeline': pipeline_final,
        'threshold': melhor['threshold'],
        'features': features_final
    }, f)

print("-> Pipeline salvo em 'ml_classificador_atrasos_v2.pkl'")
print(f"   Features esperadas ({len(features_final)}): {features_final[:5]} ...")
print(f"   Threshold de decisão: {melhor['threshold']:.2f}")

# ============================================================
# 11. GRÁFICOS
# ============================================================

# --- 11a. Matriz de Confusão do melhor modelo ---
conf_matrix = confusion_matrix(y_test, melhor['y_pred'])
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=['No Prazo', 'Atrasou']
).plot(cmap='Blues', values_format='d')
plt.title(f'Matriz de Confusão — {melhor_nome}\n(Threshold = {melhor["threshold"]:.2f})')
plt.savefig('matriz_confusao_final.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Matriz de confusão salva.")

# --- 11b. Curva Precision-Recall de todos os modelos ---
plt.figure(figsize=(8, 6))
cores = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
for (nome, res), cor in zip(resultados.items(), cores):
    precisoes, recalls, _ = precision_recall_curve(y_test, res['probs'])
    plt.plot(recalls, precisoes, label=f"{nome} (AUC-PR={res['aucpr']:.3f})", color=cor, lw=2)
    plt.scatter([res['recall']], [res['precision']], color=cor, marker='*', s=200, zorder=5)

baseline = y_test.mean()
plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline aleatório ({baseline:.2f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Curva Precision-Recall — Comparação de Modelos\n(★ = threshold escolhido)', fontsize=13)
plt.legend(loc='upper right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('curva_precision_recall.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Curva Precision-Recall salva.")

# --- 11c. Feature Importance do melhor modelo (se disponível) ---
classificador_final = melhor['modelo']

if hasattr(classificador_final, 'feature_importances_'):
    importancias = classificador_final.feature_importances_
    df_imp = (
        pd.DataFrame({'Feature': features_final, 'Importancia': importancias})
        .sort_values('Importancia', ascending=False)
        .head(15)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Feature', data=df_imp,
                hue='Feature', palette='viridis', legend=False)
    plt.title(f'Top 15 Features Mais Importantes — {melhor_nome}')
    plt.xlabel('Grau de Importância')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Gráfico de feature importance salvo.")

elif hasattr(classificador_final, 'coef_'):
    importancias = np.abs(classificador_final.coef_[0])
    df_imp = (
        pd.DataFrame({'Feature': features_final, 'Importancia': importancias})
        .sort_values('Importancia', ascending=False)
        .head(15)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Feature', data=df_imp,
                hue='Feature', palette='magma', legend=False)
    plt.title(f'Top 15 Coeficientes — {melhor_nome}')
    plt.xlabel('Peso Absoluto')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Gráfico de coeficientes salvo.")

# --- 11d. Comparativo de AUC-PR entre modelos ---
nomes  = list(resultados.keys())
aucprs = [resultados[n]['aucpr'] for n in nomes]
f1s    = [resultados[n]['f1']    for n in nomes]

x = np.arange(len(nomes))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, aucprs, width, label='AUC-PR', color='#1976D2')
bars2 = ax.bar(x + width/2, f1s,    width, label='F1-Score', color='#388E3C')

ax.set_title('Comparação de Modelos — AUC-PR vs F1-Score', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(nomes, fontsize=11)
ax.set_ylim(0, 0.75)
ax.legend()
ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)
ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparativo_modelos.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Gráfico comparativo de modelos salvo.")

print("\nTodos os artefatos gerados com sucesso.")

# ============================================================
# COMO USAR O MODELO SALVO EM PRODUÇÃO (ex: Django)
# ============================================================
# import pickle
# import pandas as pd
#
# with open("ml_classificador_atrasos_v2.pkl", "rb") as f:
#     artefato = pickle.load(f)
#
# pipeline   = artefato['pipeline']
# threshold  = artefato['threshold']
# features   = artefato['features']
#
# # Monte um DataFrame com as mesmas features e na mesma ordem:
# X_novo = pd.DataFrame([{
#     'distancia_km': 450.0,
#     'dias_estimados_logistica': 12,
#     ... # todas as features
# }])[features]
#
# prob_atraso = pipeline.predict_proba(X_novo)[0, 1]
# vai_atrasar = int(prob_atraso >= threshold)
# print(f"Probabilidade de atraso: {prob_atraso:.2%} | Vai atrasar: {bool(vai_atrasar)}")