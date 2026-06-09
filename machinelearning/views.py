import os
import pandas as pd
import pickle
from django.shortcuts import render, redirect
from django.conf import settings
from geopy.distance import geodesic
import datetime
from django.http import JsonResponse
import json
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, roc_auc_score

MODEL_PATH = os.path.join(settings.BASE_DIR, 'machinelearning', 'ml_models', 'ml_classifcador_atrasos.pkl')
DATASET_DIR = os.path.join(settings.BASE_DIR, 'datasets')
DB_VENDAS_PATH = os.path.join(settings.BASE_DIR, 'datasets', 'vendas_realizadas.csv')

with open(MODEL_PATH, 'rb') as f:
    ML_CLASSIFICADOR_ATRASOS = pickle.load(f)

def vitrine_view(request):
    produtos = carregar_produtos_vitrine()
    return render(request, 'vitrine.html', {
        'produtos': produtos
    })

def carregar_produtos_vitrine():
    try:
        items_df = pd.read_csv(os.path.join(DATASET_DIR, 'olist_order_items_dataset.csv'), nrows=100)
        products_df = pd.read_csv(os.path.join(DATASET_DIR, 'olist_products_dataset.csv'))
        sellers_df = pd.read_csv(os.path.join(DATASET_DIR, 'olist_sellers_dataset.csv'))
        
        df_completo = items_df.merge(products_df, on='product_id', how='inner')
        df_completo = df_completo.merge(sellers_df, on='seller_id', how='inner')
        
        df_vitrine = df_completo.drop_duplicates(subset=['product_id']).head(15)
        df_vitrine['product_category_name'] = df_vitrine['product_category_name'].fillna('utilidades_domesticas')
        
        return df_vitrine.to_dict(orient='records')
    except Exception as e:
        print(f"Erro ao carregar dados da vitrine: {e}")
        return []

def buscar_coordenadas_cep(cep_prefixo):
    """Busca a lat/lng média do prefixo do CEP no dataset de geolocalização do Olist"""
    try:
        geo_df = pd.read_csv(os.path.join(DATASET_DIR, 'olist_geolocation_dataset.csv'), nrows=5000)
       
        regiao = geo_df[geo_df['geolocation_zip_code_prefix'] == int(cep_prefixo)]
        if not regiao.empty:
            return regiao['geolocation_lat'].mean(), regiao['geolocation_lng'].mean(), regiao['geolocation_state'].iloc[0]
        return -3.1190, -60.0217, "AM" # Fallback: Manaus/IFAM caso não encontre
    except:
        return -3.1190, -60.0217, "AM"
    
def predizer_satisfacao(dados_feature):
    print(dados_feature)
    try:
        # Quando seu modelo estiver salvo, descomente as linhas abaixo:
        # modelo = joblib.load('caminho/do/seu/melhor_modelo.pkl')
        # scaler = joblib.load('caminho/do/seu/scaler.pkl')
        # dados_escalados = scaler.transform([dados_features])
        # predicao = modelo.predict(dados_escalados)[0]
        # return predicao
        pass
    except:
        if dados_feature[0] > 5 or dados_feature[2] == 1:
            return 1
        return 0
        
def checkout_view(request):
    product_id = request.GET.get('product_id')
    produtos = carregar_produtos_vitrine()
    produto_escolhido = next((p for p in produtos if p['product_id'] == product_id), None)
    
    if not produto_escolhido:
        return redirect('vitrine')
        
    # Inicialização de variáveis de frete dinâmico via sessões ou POST
    cep_cliente = request.POST.get('cep_cliente', '')
    frete_calculado = 0.0
    prazo_calculado = 0
    uf_cliente = "AM"
    distancia_km = 0.0
    
    # Se o usuário informou o CEP (estilo Amazon: cálculo antes do fechamento)
    if cep_cliente and len(cep_cliente) >= 5:
        prefixo = cep_cliente[:5]
        lat_c, lng_c, uf_cliente = buscar_coordenadas_cep(prefixo)
        
        lat_v, lng_v = -23.5505, -46.6333 # Padrão SP
        
        distancia_km = geodesic((lat_c, lng_c), (lat_v, lng_v)).km
        
        prazo_calculado = max(5, int(distancia_km / 250) + 3)
        frete_calculado = round(15.0 + (distancia_km * 0.015) + (float(produto_escolhido['product_weight_g']) / 1000), 2)
    
    preco_produto = float(produto_escolhido['price'])
    total_pedido = round(preco_produto + frete_calculado, 2)
    
    contexto = {
        'produto': produto_escolhido,
        'cep_cliente': cep_cliente,
        'frete_calculado': frete_calculado,
        'prazo_calculado': prazo_calculado,
        'total_pedido': total_pedido,
        'uf_cliente': uf_cliente
    }
    
    # Confirmação Final do Pedido (POST definitivo)
    if request.method == "POST" and "confirmar_pedido" in request.POST:
        forma_pagto = request.POST.get("forma_pagamento", "credit_card")
        endereco = request.POST.get("endereco_completo", "Av. Sete de Setembro, 1975 - Centro")
        
        peso = float(produto_escolhido['product_weight_g'])
        uf_vendedor = produto_escolhido['seller_state']
        
        agora = datetime.datetime.now()
        mesmo_estado = 1 if uf_vendedor == uf_cliente else 0
        fluxo_logistico_atual = f"fluxo_logistico_{uf_vendedor}_{uf_cliente}"
        
        # Montagem do vetor de entrada alinhado com o scikit-learn
        dados_usuario = {
            "taxa_atraso_categoria": [0.08],
            "distancia_km": [distancia_km if distancia_km > 0 else 1500.0],
            "dias_estimados_logistica": [prazo_calculado if prazo_calculado > 0 else 15], 
            "dia_semana_estimado": [(agora + datetime.timedelta(days=prazo_calculado)).weekday()],
            "mes_compra": [agora.month],
            "dia_semana_compra": [agora.weekday()],
            "mesmo_estado": [mesmo_estado],
            "hora_compra": [agora.hour],
            "price": [preco_produto],
            "freight_value": [frete_calculado],
            "qtde_itens": [1],
            "qtde_produtos": [1],
            "qtde_vendedores": [1],
            "product_weight_g": [peso],
            "product_length_cm": [float(produto_escolhido.get('product_length_cm', 20))],
            "product_height_cm": [float(produto_escolhido.get('product_height_cm', 15))],
            "product_width_cm": [float(produto_escolhido.get('product_width_cm', 20))],
            "payment_value": [total_pedido],
            "payment_installments": [1],
            "payment_sequential": [1],
            "boleto": [1 if forma_pagto == 'boleto' else 0],
            "credit_card": [1 if forma_pagto == 'credit_card' else 0],
            "voucher": [1 if forma_pagto == 'voucher' else 0],
            "debit_card": [1 if forma_pagto in ['debit_card', 'pix'] else 0] # Mapeia PIX no padrão debito
        }
        
        colunas_do_modelo = ML_CLASSIFICADOR_ATRASOS.feature_names_in_
        for col in colunas_do_modelo:
            if col not in dados_usuario:
                dados_usuario[col] = [0]
                
        if f"customer_state_{uf_cliente}" in dados_usuario:
            dados_usuario[f"customer_state_{uf_cliente}"] = [1]
        if f"seller_state_{uf_vendedor}" in dados_usuario:
            dados_usuario[f"seller_state_{uf_vendedor}"] = [1]
        if fluxo_logistico_atual in dados_usuario:
            dados_usuario[fluxo_logistico_atual] = [1]
            
        df_input = pd.DataFrame(dados_usuario)[colunas_do_modelo]
        
        # INFERÊNCIA DA IA
        probabilidade = ML_CLASSIFICADOR_ATRASOS.predict_proba(df_input)[:, 1][0]
        alerta_atraso = 1 if probabilidade >= 0.65 else 0
        
        # SALVAMENTO NO BANCO DE DADOS (CSV)
        nova_venda = {
            'data_venda': agora.strftime('%Y-%m-%d %H:%M:%S'),
            'product_id': product_id,
            'vendedor_origem': uf_vendedor,
            'cliente_destino': uf_cliente,
            'cep': cep_cliente,
            'endereco': endereco,
            'forma_pagamento': forma_pagto,
            'preco_produto': preco_produto,
            'freight_value': frete_calculado,
            'total_pago': total_pedido,
            'dias_estimados_logistica': prazo_calculado,
            'probabilidade_atraso_ia': round(probabilidade * 100, 2),
            'risco_de_atraso': alerta_atraso
        }
        
        df_nova_venda = pd.DataFrame([nova_venda])
        if not os.path.exists(DB_VENDAS_PATH):
            df_nova_venda.to_csv(DB_VENDAS_PATH, index=False)
        else:
            df_nova_venda.to_csv(DB_VENDAS_PATH, mode='a', header=False, index=False)
            
        contexto['sucesso_compra'] = True
        contexto['alerta_atraso'] = alerta_atraso
        contexto['probabilidade'] = round(probabilidade * 100, 1)
        
    return render(request, 'checkout.html', contexto)


def pipeline_engenharia_atributos(df_base):
    """
    Recria o pipeline de tratamento de dados do modelo para o CSV submetido.
    Transforma dados brutos nas features exatas que a IA espera.
    """
    df = df_base.copy()
    
    # Tratamentos básicos de tipos temporais
    if 'order_delivered_customer_date' in df.columns:
        df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')
    if 'order_estimated_delivery_date' in df.columns:
        df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'], errors='coerce')
    if 'order_purchase_timestamp' in df.columns:
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors='coerce')
    
    # Preenchimento de lat/lng ausentes
    for col in ['lat_customer', 'lng_customer', 'lat_seller', 'lng_seller']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean() if pd.notna(df[col].mean()) else 0)
            
    # Cálculo de distância geodésica
    if all(c in df.columns for c in ['lat_customer', 'lng_customer', 'lat_seller', 'lng_seller']) and 'distancia_km' not in df.columns:
        def calcular_distancia_km(row):
            try:
                return geodesic((row['lat_customer'], row['lng_customer']), (row['lat_seller'], row['lng_seller'])).km
            except:
                return 0.0
        df['distancia_km'] = df.apply(calcular_distancia_km, axis=1)
    elif 'distancia_km' not in df.columns:
        df['distancia_km'] = 0.0

    # Criação das features temporais
    if 'atrasou' not in df.columns and 'order_delivered_customer_date' in df.columns:
        df["atrasou"] = (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(int)
    
    if "order_estimated_delivery_date" in df.columns and "order_purchase_timestamp" in df.columns:
        df["dias_estimados_logistica"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
        df["dia_semana_estimado"] = df["order_estimated_delivery_date"].dt.dayofweek
        df["mes_compra"] = df["order_purchase_timestamp"].dt.month
        df["dia_semana_compra"] = df["order_purchase_timestamp"].dt.dayofweek
        df["hora_compra"] = df["order_purchase_timestamp"].dt.hour
    
    if 'customer_state' in df.columns and 'seller_state' in df.columns:
        df['mesmo_estado'] = (df['customer_state'] == df['seller_state']).astype(int)
        df['fluxo_logistico'] = df['seller_state'] + "_" + df['customer_state']
        df = pd.get_dummies(df, columns=['fluxo_logistico', 'customer_state', 'seller_state'], dtype=int)
    
    # Alinha as colunas do DataFrame processado com as colunas reais que o modelo aprendeu
    colunas_esperadas = ML_CLASSIFICADOR_ATRASOS.feature_names_in_
    
    for col in colunas_esperadas:
        if col not in df.columns:
            df[col] = 0 # Preenche dummies ou features ausentes com 0
            
    X_final = df[colunas_esperadas]
    y_final = df['atrasou'] if 'atrasou' in df.columns else None
    
    return X_final, y_final

# ==============================================================================
# RENDERIZAÇÃO DA PÁGINA DO DASHBOARD
# ==============================================================================
def dashboard_view(request):
    """Renderiza aquele arquivo index.html (Olist MVP) que criamos com Bootstrap."""
    return render(request, 'dashboard.html')

# ==============================================================================
# API 1: SIMULAÇÃO DINÂMICA (AJUSTE DE PERCENTUAL)
# ==============================================================================
@csrf_exempt
def api_simulacao(request):
    if request.method == "POST":
        try:
            dados = json.loads(request.body)
            percentual = int(dados.get('percentual', 20))
            
            caminho_teste = os.path.join(DATASET_DIR, 'prod/test.csv')
            if not os.path.exists(caminho_teste):
                return JsonResponse({'erro': 'Arquivo não encontrado.'}, status=404)
            
            df_teste = pd.read_csv(caminho_teste)
            df_amostra = df_teste.sample(frac=percentual/100, random_state=None)
            
            X_amostra = df_amostra.drop(columns=['alvo_real_atrasou', 'atrasou'], errors='ignore')
            y_amostra = df_amostra['alvo_real_atrasou'] if 'alvo_real_atrasou' in df_amostra.columns else None
            
            colunas_esperadas = ML_CLASSIFICADOR_ATRASOS.feature_names_in_
            for col in colunas_esperadas:
                if col not in X_amostra.columns:
                    X_amostra[col] = 0
            X_amostra = X_amostra[colunas_esperadas]
            
            y_probabilidades = ML_CLASSIFICADOR_ATRASOS.predict_proba(X_amostra)[:, 1]
            y_pred = (y_probabilidades >= 0.65).astype(int)
            
            volume = len(df_amostra)
            alertas = int(np.sum(y_pred))
            
            y_pred_list = y_pred.tolist()
            y_amostra_list = y_amostra.tolist() if y_amostra is not None else []
            
            tabela_detalhes = []
            metrics = {} # Dicionário para as novas métricas
            
            if y_amostra is not None:
                acuracia = f"{(np.mean(y_pred == y_amostra) * 100):.2f}%"
                
                # ==========================================
                # CALCULO DAS MÉTRICAS DE DATA SCIENCE
                # ==========================================
                p_score = precision_score(y_amostra, y_pred, zero_division=0) * 100
                r_score = recall_score(y_amostra, y_pred, zero_division=0) * 100
                f1 = f1_score(y_amostra, y_pred, zero_division=0) * 100
                
                try:
                    auc = roc_auc_score(y_amostra, y_probabilidades) * 100
                except ValueError:
                    auc = 0.0 # Caso a amostra aleatória venha sem uma das classes
                
                # Matriz de Confusão (Verdadeiro Negativo, Falso Positivo, Falso Negativo, Verdadeiro Positivo)
                cm = confusion_matrix(y_amostra, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = map(int, cm.ravel())
                else:
                    tn, fp, fn, tp = volume, 0, 0, 0 # Fallback de segurança
                
                metrics = {
                    'precision': f"{p_score:.2f}%",
                    'recall': f"{r_score:.2f}%",
                    'f1_score': f"{f1:.2f}%",
                    'auc_roc': f"{auc:.2f}%",
                    'cm': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
                }
                
                for pred, real in zip(y_pred_list, y_amostra_list):
                    tabela_detalhes.append({
                        'predicao': int(pred),   
                        'real': int(real),       
                        'acertou': bool(pred == real)
                    })
            else:
                acuracia = "N/A"
            
            return JsonResponse({
                'sucesso': True,
                'volume': volume,
                'alertas': alertas,
                'acuracia': acuracia,
                'metrics': metrics, # Enviando o pacote de métricas avançadas
                'detalhes': tabela_detalhes,
            })
            
        except Exception as e:
            return JsonResponse({'erro': str(e)}, status=500)
    
    return JsonResponse({'erro': 'Método não permitido.'}, status=405)

# ==============================================================================
# API 2: UPLOAD E STRESS TEST DE CSV AD-HOC
# ==============================================================================
@csrf_exempt
def api_upload_csv(request):
    print("ENtriou aqui")
    if request.method == 'POST' and request.FILES.get('file'):
        arquivo_csv = request.FILES['file']
        
        try:
            df_submetido = pd.read_csv(arquivo_csv)
            
            # Processa o dataframe bruto transformando nas variáveis do modelo
            X_submetido, y_submetido = pipeline_engenharia_atributos(df_submetido)
            
            # Executa as predições
            y_probabilidades = ML_CLASSIFICADOR_ATRASOS.predict_proba(X_submetido)[:, 1]
            y_pred = (y_probabilidades >= 0.65).astype(int)
            
            volume = len(df_submetido)
            alertas = int(np.sum(y_pred))
            
            resultado = {
                'sucesso': True,
                'volume': volume,
                'alertas': alertas,
                'acuracia': 'N/A'
            }
            
            # Se o CSV tinha gabarito ('atrasou' calculado pelas datas), calcula Acurácia e Matriz
            if y_submetido is not None and not y_submetido.isnull().all():
                acuracia = np.mean(y_pred == y_submetido)
                resultado['acuracia'] = f"{(acuracia * 100):.2f}%"
                
                # Opcional: extrair a matriz de confusão para o front-end
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_submetido, y_pred)
                resultado['matriz_confusao'] = cm.tolist()
                
            return JsonResponse(resultado)
            
        except Exception as e:
            return JsonResponse({'erro': f'Erro ao processar arquivo: {str(e)}'}, status=500)
            
    return JsonResponse({'erro': 'Nenhum arquivo enviado ou método inválido.'}, status=400)