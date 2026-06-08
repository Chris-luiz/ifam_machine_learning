from django.urls import path
from .views import vitrine_view, checkout_view

urlpatterns = [
    path('', vitrine_view, name='vitrine'),
    path('checkout/', checkout_view, name='checkout'),
]