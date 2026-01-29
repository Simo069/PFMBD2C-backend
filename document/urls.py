from django.urls import path
from . import views

urlpatterns = [
    # PDF Management
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('list/', views.list_user_pdfs, name='list_pdfs'),
    path('<int:pdf_id>/', views.get_pdf_status, name='get_pdf_status'),
    path('<int:pdf_id>/delete/', views.delete_pdf, name='delete_pdf'),
    
    # Chunk details
    path('chunks/<int:chunk_id>/', views.get_chunk_details, name='get_chunk_details'),
]