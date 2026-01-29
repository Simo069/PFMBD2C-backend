from django.urls import path
from . import views

urlpatterns = [
    # Sessions de chat
    path('sessions/create/', views.create_chat_session, name='create_session'),
    path('sessions/', views.list_chat_sessions, name='list_sessions'),
    path('sessions/<int:session_id>/', views.get_chat_history, name='get_chat_history'),
    path('sessions/<int:session_id>/delete/', views.delete_chat_session, name='delete_session'),
    
    # Opérations RAG
    path('ask/', views.ask_question, name='ask_question'),
    path('summary/', views.generate_summary, name='generate_summary'),
    path('mindmap/', views.generate_mindmap, name='generate_mindmap'),
]