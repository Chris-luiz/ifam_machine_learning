"""
Dados em arrays para popular a aplicação
"""

ESTADOS_BR = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']

# Vendedores
SELLERS = [
    {'id': 'seller_001', 'name': 'Tech Store', 'state': 'SP', 'city': 'São Paulo', 'zip': '01310100'},
    {'id': 'seller_002', 'name': 'Fashion Plus', 'state': 'RJ', 'city': 'Rio de Janeiro', 'zip': '20040020'},
    {'id': 'seller_003', 'name': 'Home Decor', 'state': 'MG', 'city': 'Belo Horizonte', 'zip': '30140071'},
    {'id': 'seller_004', 'name': 'Sports World', 'state': 'BA', 'city': 'Salvador', 'zip': '40010160'},
    {'id': 'seller_005', 'name': 'Beauty Box', 'state': 'SC', 'city': 'Florianópolis', 'zip': '88010100'},
]

# Produtos
PRODUCTS = [
    {'id': 'prod_001', 'name': 'Notebook Dell', 'category': 'Eletrônicos', 'price': 3500.00, 'seller_id': 'seller_001', 'weight': 2000},
    {'id': 'prod_002', 'name': 'Smartphone Samsung', 'category': 'Eletrônicos', 'price': 1800.00, 'seller_id': 'seller_001', 'weight': 180},
    {'id': 'prod_003', 'name': 'Fone Bluetooth', 'category': 'Eletrônicos', 'price': 250.00, 'seller_id': 'seller_001', 'weight': 150},
    {'id': 'prod_004', 'name': 'Câmera Digital', 'category': 'Eletrônicos', 'price': 1200.00, 'seller_id': 'seller_001', 'weight': 500},
    {'id': 'prod_005', 'name': 'Camiseta Premium', 'category': 'Roupas', 'price': 89.90, 'seller_id': 'seller_002', 'weight': 200},
    {'id': 'prod_006', 'name': 'Calça Jeans', 'category': 'Roupas', 'price': 129.90, 'seller_id': 'seller_002', 'weight': 400},
    {'id': 'prod_007', 'name': 'Jaqueta Inverno', 'category': 'Roupas', 'price': 299.90, 'seller_id': 'seller_002', 'weight': 800},
    {'id': 'prod_008', 'name': 'Sapato Social', 'category': 'Roupas', 'price': 199.90, 'seller_id': 'seller_002', 'weight': 500},
    {'id': 'prod_009', 'name': 'Luminária LED', 'category': 'Decoração', 'price': 79.90, 'seller_id': 'seller_003', 'weight': 300},
    {'id': 'prod_010', 'name': 'Tapete Persa', 'category': 'Decoração', 'price': 450.00, 'seller_id': 'seller_003', 'weight': 2000},
    {'id': 'prod_011', 'name': 'Quadro Moderno', 'category': 'Decoração', 'price': 120.00, 'seller_id': 'seller_003', 'weight': 400},
    {'id': 'prod_012', 'name': 'Almofada Conforto', 'category': 'Decoração', 'price': 89.90, 'seller_id': 'seller_003', 'weight': 300},
    {'id': 'prod_013', 'name': 'Bola de Futebol', 'category': 'Esportes', 'price': 89.90, 'seller_id': 'seller_004', 'weight': 450},
    {'id': 'prod_014', 'name': 'Raquete Tênis', 'category': 'Esportes', 'price': 250.00, 'seller_id': 'seller_004', 'weight': 350},
    {'id': 'prod_015', 'name': 'Bicicleta Mountain', 'category': 'Esportes', 'price': 1200.00, 'seller_id': 'seller_004', 'weight': 15000},
    {'id': 'prod_016', 'name': 'Mochila Esportiva', 'category': 'Esportes', 'price': 150.00, 'seller_id': 'seller_004', 'weight': 600},
    {'id': 'prod_017', 'name': 'Creme Facial', 'category': 'Beleza', 'price': 79.90, 'seller_id': 'seller_005', 'weight': 100},
    {'id': 'prod_018', 'name': 'Perfume Premium', 'category': 'Beleza', 'price': 199.90, 'seller_id': 'seller_005', 'weight': 150},
    {'id': 'prod_019', 'name': 'Kit Maquiagem', 'category': 'Beleza', 'price': 120.00, 'seller_id': 'seller_005', 'weight': 300},
    {'id': 'prod_020', 'name': 'Sérum Vitamina C', 'category': 'Beleza', 'price': 89.90, 'seller_id': 'seller_005', 'weight': 80},
]

# Perfil padrão do usuário
DEFAULT_USER = {
    'name': 'João Silva',
    'state': 'SP',
    'city': 'São Paulo',
    'zip': '01310100'
}
