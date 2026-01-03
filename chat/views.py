from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import ChatSession, Message
from document.models import PDFFile
from .services.rag_service import RAGService

rag_service = RAGService()

# ============================================
# Endpoints des Sessions de Chat
# ============================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_chat_session(request):
    """
    Créer une nouvelle session de chat
    """
    title = request.data.get('title', 'Nouveau Chat')
    pdf_ids = request.data.get('pdf_ids', [])
    
    session = ChatSession.objects.create(
        user=request.user,
        title=title,
        pdf_ids=pdf_ids
    )
    
    return Response({
        'id': session.id,
        'title': session.title,
        'created_at': session.created_at,
        'pdf_ids': session.pdf_ids
    }, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_chat_sessions(request):
    """
    Lister toutes les sessions de chat de l'utilisateur connecté
    """
    sessions = ChatSession.objects.filter(user=request.user, is_active=True)
    
    sessions_data = [{
        'id': session.id,
        'title': session.title,
        'created_at': session.created_at,
        'updated_at': session.updated_at,
        'pdf_ids': session.pdf_ids,
        'message_count': session.messages.count()
    } for session in sessions]
    
    return Response({'sessions': sessions_data})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chat_history(request, session_id):
    """
    Obtenir tous les messages d'une session de chat
    """
    try:
        session = ChatSession.objects.get(id=session_id, user=request.user)
    except ChatSession.DoesNotExist:
        return Response(
            {'error': 'Session non trouvée'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    messages = Message.objects.filter(session=session).order_by('created_at')
    
    messages_data = [{
        'id': msg.id,
        'role': msg.role,
        'content': msg.content,
        'created_at': msg.created_at,
        'chunk_ids': msg.chunk_ids
    } for msg in messages]
    
    return Response({
        'session_id': session.id,
        'messages': messages_data
    })


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_chat_session(request, session_id):
    """
    Supprimer une session de chat
    """
    try:
        session = ChatSession.objects.get(id=session_id, user=request.user)
        session.delete()
        return Response({'message': 'Session supprimée avec succès'})
    except ChatSession.DoesNotExist:
        return Response(
            {'error': 'Session non trouvée'},
            status=status.HTTP_404_NOT_FOUND
        )


# ============================================
# Endpoints RAG
# ============================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def ask_question(request):
    """
    Poser une question en utilisant RAG
    
    Payload attendu:
    {
        "question": "Qu'est-ce que...",
        "session_id": 123,  // optionnel
        "pdf_ids": [1, 2],  // optionnel
        "top_k": 5          // optionnel, défaut 5
    }
    """
    question = request.data.get('question')
    session_id = request.data.get('session_id')
    pdf_ids = request.data.get('pdf_ids')
    top_k = request.data.get('top_k', 5)
    
    if not question:
        return Response(
            {'error': 'Question requise'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Valider la propriété de la session si fournie
    if session_id:
        try:
            ChatSession.objects.get(id=session_id, user=request.user)
        except ChatSession.DoesNotExist:
            return Response(
                {'error': 'Session non trouvée'},
                status=status.HTTP_404_NOT_FOUND
            )
    
    # Valider la propriété des PDFs si fournis
    if pdf_ids:
        user_pdfs = PDFFile.objects.filter(
            id__in=pdf_ids, 
            user=request.user,
            processing_status='completed'
        ).values_list('id', flat=True)
        
        if len(user_pdfs) != len(pdf_ids):
            return Response(
                {'error': 'IDs de PDF invalides ou PDFs non prêts'},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    # Traiter la question
    result = rag_service.ask_question(
        user_id=request.user.id,
        question=question,
        session_id=session_id,
        pdf_ids=pdf_ids,
        top_k=top_k
    )
    
    return Response(result)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_summary(request):
    """
    Générer un résumé pour un PDF
    
    Payload attendu:
    {
        "pdf_id": 123
    }
    """
    pdf_id = request.data.get('pdf_id')
    
    if not pdf_id:
        return Response(
            {'error': 'ID de PDF requis'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Vérifier la propriété du PDF
    try:
        pdf = PDFFile.objects.get(id=pdf_id, user=request.user)
    except PDFFile.DoesNotExist:
        return Response(
            {'error': 'PDF non trouvé'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    if pdf.processing_status != 'completed':
        return Response(
            {'error': 'Le PDF est encore en cours de traitement'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Générer le résumé
    summary = rag_service.generate_summary(request.user.id, pdf_id)
    
    return Response({
        'pdf_id': pdf_id,
        'filename': pdf.original_filename,
        'summary': summary
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_mindmap(request):
    """
    Générer une carte mentale pour un PDF
    
    Payload attendu:
    {
        "pdf_id": 123
    }
    """
    pdf_id = request.data.get('pdf_id')
    
    if not pdf_id:
        return Response(
            {'error': 'ID de PDF requis'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Vérifier la propriété du PDF
    try:
        pdf = PDFFile.objects.get(id=pdf_id, user=request.user)
    except PDFFile.DoesNotExist:
        return Response(
            {'error': 'PDF non trouvé'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    if pdf.processing_status != 'completed':
        return Response(
            {'error': 'Le PDF est encore en cours de traitement'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Générer la carte mentale
    mindmap = rag_service.generate_mindmap(request.user.id, pdf_id)
    
    return Response({
        'pdf_id': pdf_id,
        'filename': pdf.original_filename,
        'mindmap': mindmap
    })