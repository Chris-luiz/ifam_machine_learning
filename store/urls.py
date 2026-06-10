"""
URLs da aplicação store
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('catalog/', views.catalog, name='catalog'),
    path('profile/', views.profile, name='profile'),
    path('cart/', views.cart, name='cart'),
    path('checkout/', views.checkout, name='checkout'),
    path('order-confirmation/', views.order_confirmation, name='order_confirmation'),
    path('batch-test/', views.batch_test, name='batch_test'),
    
    # API
    path('api/add-to-cart/', views.add_to_cart, name='add_to_cart'),
    path('api/remove-from-cart/', views.remove_from_cart, name='remove_from_cart'),
    path('api/update-cart-quantity/', views.update_cart_quantity, name='update_cart_quantity'),
]
