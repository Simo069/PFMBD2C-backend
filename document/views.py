import os
import uuid
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import PDFFile, Chunk
from .tasks import process_pdf_async

# Taille maximale: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_pdf(request):
    """
    Upload d'un fichier PDF
    """
    # Valider la présence du fichier
    if 'file' not in request.FILES:
        return Response(
            {'error': 'Aucun fichier fourni'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    uploaded_file = request.FILES['file']
    
    # Valider le type de fichier
    if not uploaded_file.name.endswith('.pdf'):
        return Response(
            {'error': 'Seuls les fichiers PDF sont acceptés'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Valider la taille du fichier
    if uploaded_file.size > MAX_FILE_SIZE:
        return Response(
            {'error': f'La taille du fichier dépasse la limite de {MAX_FILE_SIZE / (1024*1024)}MB'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Générer un nom de fichier unique
    file_extension = '.pdf'
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Déterminer le chemin de stockage
    user_folder = os.path.join(settings.MEDIA_ROOT, 'pdfs', str(request.user.id))
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, unique_filename)
    
    # Sauvegarder le fichier
    try:
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
    except Exception as e:
        return Response(
            {'error': f'Échec de la sauvegarde du fichier: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    # Créer l'enregistrement dans la base de données
    pdf_file = PDFFile.objects.create(
        user=request.user,
        filename=unique_filename,
        original_filename=uploaded_file.name,
        file_path=file_path,
        file_size=uploaded_file.size,
        processing_status='pending'
    )
    
    # Déclencher le traitement asynchrone
    process_pdf_async.delay(pdf_file.id)
    
    return Response({
        'id': pdf_file.id,
        'filename': pdf_file.original_filename,
        'size': pdf_file.file_size,
        'status': pdf_file.processing_status,
        'message': 'PDF uploadé avec succès. Traitement démarré.'
    }, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_pdf_status(request, pdf_id):
    """
    Vérifier le statut de traitement d'un PDF
    """
    try:
        pdf_file = PDFFile.objects.get(id=pdf_id, user=request.user)
    except PDFFile.DoesNotExist:
        return Response(
            {'error': 'PDF non trouvé'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    return Response({
        'id': pdf_file.id,
        'filename': pdf_file.original_filename,
        'status': pdf_file.processing_status,
        'page_count': pdf_file.page_count,
        'total_chunks': pdf_file.total_chunks,
        'upload_date': pdf_file.upload_date
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_user_pdfs(request):
    """
    Lister tous les PDFs de l'utilisateur connecté
    """
    pdfs = PDFFile.objects.filter(user=request.user)
    
    pdf_list = [{
        'id': pdf.id,
        'filename': pdf.original_filename,
        'size': pdf.file_size,
        'status': pdf.processing_status,
        'page_count': pdf.page_count,
        'total_chunks': pdf.total_chunks,
        'upload_date': pdf.upload_date
    } for pdf in pdfs]
    
    return Response({'pdfs': pdf_list})


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_pdf(request, pdf_id):
    """
    Supprimer un PDF et toutes ses données associées
    """
    try:
        pdf_file = PDFFile.objects.get(id=pdf_id, user=request.user)
    except PDFFile.DoesNotExist:
        return Response(
            {'error': 'PDF non trouvé'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Supprimer le fichier physique
    if os.path.exists(pdf_file.file_path):
        os.remove(pdf_file.file_path)
    
    # Supprimer de la base de données vectorielle
    from .services.vector_db_service import VectorDBService
    vector_db = VectorDBService()
    vector_db.delete_pdf_vectors(pdf_file.id, request.user.id)
    
    # Supprimer l'enregistrement de la base de données (cascade vers chunks)
    pdf_file.delete()
    
    return Response({
        'message': 'PDF supprimé avec succès'
    }, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chunk_details(request, chunk_id):
    """
    Obtenir les détails d'un chunk spécifique
    """
    try:
        chunk = Chunk.objects.get(id=chunk_id, user=request.user)
    except Chunk.DoesNotExist:
        return Response(
            {'error': 'Chunk non trouvé'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    return Response({
        'id': chunk.id,
        'text': chunk.chunk_text,
        'pdf_id': chunk.pdf.id,
        'pdf_filename': chunk.pdf.original_filename,
        'page_number': chunk.page_number,
        'chunk_index': chunk.chunk_index,
        'token_count': chunk.token_count
    })
    
    
    