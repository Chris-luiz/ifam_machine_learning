import pickle
import os
import csv
import json
import random
import numpy as np
import math
from datetime import datetime
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .data import PRODUCTS, SELLERS, DEFAULT_USER, ESTADOS_BR
from .ml_model import predict_delay, get_model_info, process_batch_predictions

# Carregar modelo uma única vez
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml_classificador_atrasos_v2.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        MODEL = pickle.load(f)
    MODEL_LOADED = True
except Exception as e:
    MODEL = None
    MODEL_LOADED = False

# Simulação de sessão do usuário
def get_user_profile(request):
    """Obtém o perfil do usuário da sessão"""
    if 'user_profile' not in request.session:
        request.session['user_profile'] = DEFAULT_USER.copy()
    return request.session['user_profile']

def get_cart(request):
    """Obtém o carrinho da sessão"""
    if 'cart' not in request.session:
        request.session['cart'] = []
    return request.session['cart']

def save_cart(request, cart):
    """Salva o carrinho na sessão"""
    request.session['cart'] = cart
    request.session.modified = True

def save_user_profile(request, profile):
    """Salva o perfil do usuário na sessão"""
    request.session['user_profile'] = profile
    request.session.modified = True


def index(request):
    cart_count = len(get_cart(request))
    model_info = get_model_info()
    
    return render(request, 'store/index.html', {
        'cart_count': cart_count,
        'model_info': model_info
    })


def catalog(request):
    """Página de catálogo"""
    cart_count = len(get_cart(request))
    
    # Filtros
    category = request.GET.get('category', '')
    search = request.GET.get('search', '').lower()
    
    # Aplicar filtros
    products = PRODUCTS
    if category:
        products = [p for p in products if p['category'] == category]
    if search:
        products = [p for p in products if search in p['name'].lower()]
    
    # Obter categorias únicas
    categories = sorted(set(p['category'] for p in PRODUCTS))
    
    # Mapa de vendedores
    sellers_map = {s['id']: s for s in SELLERS}
    
    return render(request, 'store/catalog.html', {
        'products': products,
        'categories': categories,
        'sellers_map': sellers_map,
        'cart_count': cart_count,
        'selected_category': category,
        'search_term': search
    })


def profile(request):
    """Página de perfil do usuário"""
    cart_count = len(get_cart(request))
    user = get_user_profile(request)
    
    if request.method == 'POST':
        user['name'] = request.POST.get('name', user['name'])
        user['state'] = request.POST.get('state', user['state'])
        user['city'] = request.POST.get('city', user['city'])
        user['zip'] = request.POST.get('zip', user['zip'])
        save_user_profile(request, user)
        return redirect('profile')
    
    return render(request, 'store/profile.html', {
        'user': user,
        'states': ESTADOS_BR,
        'cart_count': cart_count
    })


def cart(request):
    """Página do carrinho"""
    cart_items = get_cart(request)
    
    # Enriquecer itens com informações do produto
    sellers_map = {s['id']: s for s in SELLERS}
    products_map = {p['id']: p for p in PRODUCTS}
    
    for item in cart_items:
        product = products_map.get(item['product_id'], {})
        item['name'] = product.get('name', '')
        item['category'] = product.get('category', '')
        seller = sellers_map.get(product.get('seller_id', ''), {})
        item['seller_state'] = seller.get('state', '')
    
    total = sum(item['price'] * item['quantity'] for item in cart_items)
    
    return render(request, 'store/cart.html', {
        'cart_items': cart_items,
        'total': total,
        'cart_count': len(cart_items)
    })


def checkout(request):
    """Página de checkout"""
    cart_items = get_cart(request)
    user = get_user_profile(request)
    
    if not cart_items:
        return redirect('cart')
    
    # Enriquecer dados
    sellers_map = {s['id']: s for s in SELLERS}
    products_map = {p['id']: p for p in PRODUCTS}
    
    for item in cart_items:
        product = products_map.get(item['product_id'], {})
        item['name'] = product.get('name', '')
        seller = sellers_map.get(product.get('seller_id', ''), {})
        item['seller_state'] = seller.get('state', '')
    
    total = sum(item['price'] * item['quantity'] for item in cart_items)
    freight = 15.00
    
    if request.method == 'POST':
        # Preparar features para predição - TODAS AS 62 FEATURES
        seller_state = cart_items[0].get('seller_state', 'SP') if cart_items else 'SP'
        
        # Inicializar features com todas as 62 esperadas
        features = {
            'distancia_km': math.sqrt((len(user['state']) + len(seller_state)) * 1000),
            'dias_estimados_logistica': 5,
            'dia_semana_estimado': 2,
            'mes_compra': datetime.today().month,
            'dia_semana_compra': datetime.today().day,
            'mesmo_estado': 1 if user['state'] == seller_state else 0,
            'hora_compra': datetime.now().second,
            'price': total,
            'freight_value': freight,
            'qtde_itens': len(cart_items),
            'qtde_produtos': len(cart_items),
            'qtde_vendedores': 1,
            'taxa_atraso_categoria': 0.08,
        }
        
        # Adicionar features de estado do cliente (27 estados)
        for estado in ESTADOS_BR:
            features[f'customer_state_{estado}'] = 1 if user['state'] == estado else 0
        
        # Adicionar features de estado do vendedor (27 estados)
        for estado in ESTADOS_BR:
            features[f'seller_state_{estado}'] = 1 if seller_state == estado else 0
        
        # Fazer predição com todas as features
        prediction = predict_delay(features)
        
        # Salvar resultado na sessão
        request.session['last_order'] = {
            'order': {
                'order_id': f"ORD_{len(get_cart(request))}_{hash(str(cart_items)) % 10000}",
                'total': total + freight,
                'prediction': prediction,
                'user': user,
                'seller_state': seller_state
            }
        }
        
    
        request.session.modified = True
        
        # Limpar carrinho
        # save_cart(request, [])
        return redirect('order_confirmation')
    
    return render(request, 'store/checkout.html', {
        'cart_items': cart_items,
        'total': total,
        'freight': freight,
        'user': user,
        'cart_count': len(cart_items)
    })


def order_confirmation(request):
    """Confirmação de pedido com predição"""
    
    
    user = get_user_profile(request)
    cart_items = get_cart(request)
    
    if not cart_items or not user:
        return redirect('checkout')
    
    request.session['cart'] = []
    request.session.modified = True
    
    prediction = (request.session['last_order'])
    print("Last Order")
    print(prediction)
    
    return render(request, 'store/order_confirmation.html', context=prediction)


def batch_test(request):
    """Testes em lote com upload de CSV e porcentagem"""
    results = None
    error = None
    
    if request.method == 'POST':
        try:
            csv_content = None
            percentage = int(request.POST.get('percentage', 100))
            # Validar porcentagem
            if not (1 <= percentage <= 100):
                error = "Porcentagem deve estar entre 1% e 100%"
                return render(request, 'store/batch_test.html', {'error': error})
            
            # Obter dados do CSV (arquivo ou texto)
            if 'csv_file' in request.FILES:
                csv_file = request.FILES['csv_file']
                
                try:
                    csv_content = csv_file.read().decode('utf-8')
                except UnicodeDecodeError:
                    error = "Arquivo deve estar em formato UTF-8"
                    return render(request, 'store/batch_test.html', {'error': error})
            else:
                csv_content = request.POST.get('csv_content', '').strip()
            
            if not csv_content:
                error = "Por favor, forneça um arquivo CSV ou cole o conteúdo"
                return render(request, 'store/batch_test.html', {'error': error})
            
            # Parsear CSV - remover linhas vazias
            if isinstance(csv_content, str):
                lines = [line.strip() for line in csv_content.strip().split('\n') if line.strip()]
            else:
                lines = csv_content
            
            if len(lines) < 2:
                error = "CSV deve ter pelo menos 2 linhas (cabeçalho + dados)"
                return render(request, 'store/batch_test.html', {'error': error})
            
            # Extrair dados
            try:
                reader = csv.reader(lines)
                header = next(reader)
                data = list(reader)
            except Exception as e:
                error = f"Erro ao ler CSV: {str(e)}"
                return render(request, 'store/batch_test.html', {'error': error})
            
            # Remover linhas vazias ou incompletas
            data = [row for row in data if row and len(row) > 0]
            
            if not data:
                error = "Nenhum dado válido encontrado no CSV"
                return render(request, 'store/batch_test.html', {'error': error})
            
            # Aplicar porcentagem
            if percentage < 100:
                sample_size = max(1, int(len(data) * percentage / 100))
                data = random.sample(data, sample_size)
            
            if not data:
                error = "Nenhum dado para processar após aplicar a porcentagem"
                return render(request, 'store/batch_test.html', {'error': error})
            
            # Processar predições
            
            results = process_batch_predictions(data, percentage)
            
        except ValueError as e:
            error = f"Erro ao processar CSV: {str(e)}"
        except Exception as e:
            error = f"Erro inesperado: {str(e)}"
            import traceback
            traceback.print_exc()

    context = {
        'results': results,
        'error': error,
    }
    return render(request, 'store/batch_test.html', context)


@require_http_methods(["POST"])
def add_to_cart(request):
    """API para adicionar ao carrinho"""
    
    data = json.loads(request.body)
    product_id = data.get('product_id')
    quantity = int(data.get('quantity', 1))

    
    product = next((p for p in PRODUCTS if p['id'] == product_id), None)
    if not product:
        return JsonResponse({'success': False, 'error': 'Produto não encontrado'})
    
    cart = request.session.get('cart', [])
    
    # Verificar se já existe no carrinho
    existing = next((item for item in cart if item['product_id'] == product_id), None)
    if existing:
        existing['quantity'] += quantity
    else:
        cart.append({
            'product_id': product_id,
            'name': product['name'],
            'category': product['category'],
            'price': product['price'],
            'seller_id': product['seller_id'],
            'quantity': quantity,
        })
    
    request.session['cart'] = cart
    request.session.modified = True
    
    return JsonResponse({
        'success': True, 
        'cart_count': len(cart), 
        'message': f'{product["name"]} adicionado ao carrinho!',
    })


@require_http_methods(["POST"])
def remove_from_cart(request):
    """Remove produto do carrinho (AJAX)"""
    try:
        data = json.loads(request.body)
        product_id = data.get('product_id')
        
        cart = get_cart(request)
        cart = [item for item in cart if item['product_id'] != product_id]
        save_cart(request, cart)
        
        return JsonResponse({
            'success': True,
            'cart_count': len(cart)
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@require_http_methods(["POST"])
def update_cart_quantity(request):
    """Atualiza quantidade no carrinho (AJAX)"""
    try:
        data = json.loads(request.body)
        product_id = data.get('product_id')
        quantity = int(data.get('quantity', 1))
        
        cart = get_cart(request)
        item = next((item for item in cart if item['product_id'] == product_id), None)
        
        if item:
            if quantity <= 0:
                cart = [i for i in cart if i['product_id'] != product_id]
            else:
                item['quantity'] = quantity
        
        save_cart(request, cart)
        
        return JsonResponse({
            'success': True,
            'cart_count': len(cart)
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
