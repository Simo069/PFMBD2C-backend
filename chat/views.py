from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import ChatSession, Message
from document.models import PDFFile
from .services.rag_service import RAGService
from django.db.models import Q

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
    title = request.data.get('title', 'New Chat')
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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_session_title(request, session_id):
    """
    Mettre à jour le titre d'une session de chat
    """
    try:
        session = ChatSession.objects.get(id=session_id, user=request.user)
    except ChatSession.DoesNotExist:
        return Response(
            {'error': 'Session non trouvée'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    title = request.data.get('title')
    if not title:
        return Response(
            {'error': 'Titre requis'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    session.title = title
    session.save()
    
    return Response({
        'id': session.id,
        'title': session.title,
        'updated_at': session.updated_at
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_chat_sessions(request):
    """
    Lister toutes les sessions de chat de l'utilisateur connecté
    """
    sessions = ChatSession.objects.filter(user=request.user, is_active=True)
    
    sessions_data = []
    for session in sessions:
        # Récupérer les noms des PDFs
        pdf_names = []
        for pdf_id in session.pdf_ids:
            try:
                pdf = PDFFile.objects.get(id=pdf_id, user=request.user)
                pdf_names.append(pdf.original_filename)
            except PDFFile.DoesNotExist:
                pass
        
        sessions_data.append({
            'id': session.id,
            'title': session.title,
            'created_at': session.created_at,
            'updated_at': session.updated_at,
            'pdf_ids': session.pdf_ids,
            'pdf_names': pdf_names,
            'message_count': session.messages.count()
        })
    
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
        return Response({'error': 'Question requise'}, status=400)

    # récupérer la session
    if session_id:
        session = ChatSession.objects.get(id=session_id, user=request.user)
        # Vérifier si c'est la première question de la session
        if session.messages.count() == 0:
            # Mettre à jour le titre avec la première question
            title = question[:50]
            if len(question) > 50:
                title = title[:47] + "..."
            session.title = title
            session.save()
    else:
        # Créer une nouvelle session avec la première question comme titre
        title = question[:50]
        if len(question) > 50:
            title = title[:47] + "..."
        
        session = ChatSession.objects.create(
            user=request.user,
            title=title,
            pdf_ids=pdf_ids or []
        )

    Message.objects.create(
        user=request.user,
        session=session,
        role="user",
        content=question
    )

    result = rag_service.ask_question(
        user_id=request.user.id,
        question=question,
        session_id=session.id,
        pdf_ids=pdf_ids,
        top_k=top_k
    )

    Message.objects.create(
        user=request.user,
        session=session,
        role="assistant",
        content=result.get("answer", ""),
        chunk_ids=result.get("chunk_ids", [])
    )

    return Response({
        "session_id": session.id,
        "answer": result.get("answer"),
        "sources": result.get("sources", [])
    })

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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def search_conversations(request):
    query = request.data.get('query', '').strip()
    filter_type = request.data.get('filter', 'all')  
    sort_by = request.data.get('sort_by', 'relevance')  
    top_k = request.data.get('top_k', 10)

    if not query:
        return Response({'error': 'Query is required'}, status=400)

    sessions = ChatSession.objects.filter(user=request.user, is_active=True)

    if filter_type == 'recent':
        from datetime import datetime, timedelta
        week_ago = datetime.now() - timedelta(days=7)
        sessions = sessions.filter(updated_at__gte=week_ago)
    elif filter_type == 'withPDF':
        sessions = [session for session in sessions if session.pdf_ids]

    results = []
    query_lower = query.lower()

    for session in sessions:
        relevance_score = 0
        preview = ""

        # Recherche dans le titre
        if query_lower in session.title.lower():
            relevance_score += 0.5

        # Recherche dans les messages
        messages = Message.objects.filter(session=session)
        for message in messages:
            if query_lower in message.content.lower():
                relevance_score += 0.3
                preview = message.content[:150]  
                break

        if relevance_score > 0:
            pdf_names = []
            for pdf_id in session.pdf_ids:
                try:
                    pdf = PDFFile.objects.get(id=pdf_id, user=request.user)
                    pdf_names.append(pdf.original_filename)
                except PDFFile.DoesNotExist:
                    pass

            results.append({
                'id': session.id,
                'title': session.title,
                'preview': preview,
                'message_count': messages.count(),
                'pdfs_involved': pdf_names,
                'last_activity': session.updated_at,
                'relevance_score': min(relevance_score, 1.0),
                'created_at': session.created_at
            })

    if sort_by == 'relevance':
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
    elif sort_by == 'date':
        results.sort(key=lambda x: x['last_activity'], reverse=True)

    results = results[:top_k]

    return Response({
        'success': True,
        'conversations': results,
        'count': len(results)
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def search_conversations_fulltext(request):
    """
    Recherche full-text dans les conversations (titres et messages)
    
    Payload attendu:
    {
        "query": "texte à rechercher",
        "top_k": 20  // optionnel, nombre maximum de résultats
    }
    """
    query = request.data.get('query', '').strip()
    top_k = request.data.get('top_k', 20)

    if not query:
        return Response({
            'success': True,
            'conversations': [],
            'count': 0
        })

    sessions = ChatSession.objects.filter(user=request.user, is_active=True)

    results = []
    query_lower = query.lower()

    for session in sessions:
        matches = []
        found_in_title = False
        
        # Recherche dans le titre
        if query_lower in session.title.lower():
            found_in_title = True
        
        # Recherche dans les messages
        messages = Message.objects.filter(session=session)
        for message in messages:
            if query_lower in message.content.lower():
                matches.append({
                    'role': message.role,
                    'content': message.content,
                    'created_at': message.created_at
                })
        
        if found_in_title or len(matches) > 0:
            pdf_names = []
            for pdf_id in session.pdf_ids:
                try:
                    pdf = PDFFile.objects.get(id=pdf_id, user=request.user)
                    pdf_names.append(pdf.original_filename)
                except PDFFile.DoesNotExist:
                    pass
            
            results.append({
                'id': session.id,
                'title': session.title,
                'found_in_title': found_in_title,
                'matches': matches[:10],  
                'message_count': messages.count(),
                'pdf_names': pdf_names,
                'created_at': session.created_at,
                'updated_at': session.updated_at
            })

    # Trier par date de mise à jour (les plus récentes d'abord)
    results.sort(key=lambda x: x['updated_at'], reverse=True)
    
    # Limiter le nombre de résultats
    results = results[:top_k]

    return Response({
        'success': True,
        'conversations': results,
        'count': len(results)
    })