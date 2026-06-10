"""
Views da aplicação Olist E-Commerce
"""
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from .data import PRODUCTS, SELLERS, ESTADOS_BR, DEFAULT_USER
from .ml_model import predict_delay, get_model_info, process_batch_predictions
import math

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

# Views
def index(request):
    """Página inicial"""
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
        # Preparar features para predição
        seller_state = cart_items[0].get('seller_state', 'SP') if cart_items else 'SP'
        
        features = {
            'distancia_km': math.sqrt((len(user['state']) + len(seller_state)) * 1000),
            'dias_estimados_logistica': 5,
            'dia_semana_estimado': 2,
            'mes_compra': 6,
            'dia_semana_compra': 3,
            'mesmo_estado': 1 if user['state'] == seller_state else 0,
            'hora_compra': 10,
            'price': total,
            'freight_value': freight,
            'qtde_itens': len(cart_items),
            'qtde_produtos': len(cart_items),
            'qtde_vendedores': 1,
            'taxa_atraso_categoria': 0.08,
        }
        
        # Adicionar features de estado do cliente
        for estado in ESTADOS_BR:
            features[f'customer_state_{estado}'] = 1 if user['state'] == estado else 0
        
        # Adicionar features de estado do vendedor
        for estado in ESTADOS_BR:
            features[f'seller_state_{estado}'] = 1 if seller_state == estado else 0
        
        # Fazer predição
        prediction = predict_delay(features)
        
        # Salvar resultado na sessão
        request.session['last_order'] = {
            'order_id': f"ORD_{len(get_cart(request))}_{hash(str(cart_items)) % 10000}",
            'total': total + freight,
            'prediction': prediction,
            'user': user,
            'seller_state': seller_state
        }
        request.session.modified = True
        
        # Limpar carrinho
        save_cart(request, [])
        
        return redirect('order_confirmation')
    
    return render(request, 'store/checkout.html', {
        'cart_items': cart_items,
        'total': total,
        'freight': freight,
        'user': user,
        'cart_count': len(cart_items)
    })

def order_confirmation(request):
    """Página de confirmação do pedido"""
    order = request.session.get('last_order')
    
    if not order:
        return redirect('catalog')
    
    return render(request, 'store/order_confirmation.html', {
        'order': order,
        'cart_count': 0
    })

def batch_test(request):
    """Página de testes em lote"""
    cart_count = len(get_cart(request))
    results = None
    error = None
    
    if request.method == 'POST':
        csv_content = request.POST.get('csv_content', '')
        
        if not csv_content.strip():
            error = 'Por favor, forneça dados CSV'
        else:
            try:
                results = process_batch_predictions(csv_content)
            except Exception as e:
                error = f'Erro ao processar CSV: {str(e)}'
    
    model_info = get_model_info()
    
    return render(request, 'store/batch_test.html', {
        'results': results,
        'error': error,
        'model_info': model_info,
        'cart_count': cart_count
    })

# API endpoints
@require_http_methods(["POST"])
def add_to_cart(request):
    """Adiciona produto ao carrinho (AJAX)"""
    try:
        data = json.loads(request.body)
        product_id = data.get('product_id')
        quantity = int(data.get('quantity', 1))
        
        # Encontrar produto
        product = next((p for p in PRODUCTS if p['id'] == product_id), None)
        if not product:
            return JsonResponse({'error': 'Produto não encontrado'}, status=404)
        
        # Adicionar ao carrinho
        cart = get_cart(request)
        existing = next((item for item in cart if item['product_id'] == product_id), None)
        
        if existing:
            existing['quantity'] += quantity
        else:
            cart.append({
                'product_id': product_id,
                'name': product['name'],
                'price': product['price'],
                'quantity': quantity
            })
        
        save_cart(request, cart)
        
        return JsonResponse({
            'success': True,
            'cart_count': len(cart),
            'message': f'{product["name"]} adicionado ao carrinho'
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

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
