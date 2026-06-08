import os
import pandas as pd
import pickle
from django.shortcuts import render, redirect
from django.conf import settings
from geopy.distance import geodesic
import datetime

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