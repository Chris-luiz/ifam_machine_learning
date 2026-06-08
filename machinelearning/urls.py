from django.urls import path
from .views import vitrine_view, checkout_view, dashboard_view, api_simulacao, api_upload_csv

urlpatterns = [
    path('', vitrine_view, name='vitrine'),
    path('checkout/', checkout_view, name='checkout'),

    path('dashboard/', dashboard_view, name='dashboard'),
    path('api/simulacao/', api_simulacao, name='api_simulacao'),
    path('api/upload_csv/', api_upload_csv, name='api_upload_csv'),
]