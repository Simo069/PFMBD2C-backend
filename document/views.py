import os
import uuid
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

import PyPDF2


import json


from .models import PDFFile, Chunk
from .tasks import process_pdf_async
from .services.summary_service import SummaryService
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
    Obtenir le statut d'un PDF spécifique
    """
    try:
        pdf = PDFFile.objects.get(id=pdf_id, user=request.user)
        
        return Response({
            'id': pdf.id,
            'filename': pdf.original_filename,
            'status': pdf.processing_status,
            'page_count': pdf.page_count,
            'total_chunks': pdf.total_chunks,
            'upload_date': pdf.upload_date,
            'file_size': pdf.file_size
        })
        
    except PDFFile.DoesNotExist:
        return Response(
            {'error': 'PDF non trouvé'},
            status=status.HTTP_404_NOT_FOUND
        ) 



        

    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def summarize_pdf(request, pdf_id):
    """
    Génère un résumé d'un PDF et le retourne
    """
    try:
        # Vérifier que le PDF appartient à l'utilisateur
        pdf = PDFFile.objects.get(id=pdf_id, user=request.user)
        
        # Vérifier que le PDF est traité
        if pdf.processing_status != 'completed':
            return Response({  # Utiliser Response au lieu de JsonResponse
                'error': 'Le PDF n\'est pas encore traité',
                'success': False
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Récupérer les chunks du PDF
        chunks = Chunk.objects.filter(pdf=pdf).order_by('page_number', 'chunk_index')
        
        if not chunks.exists():
            return Response({  # Utiliser Response
                'error': 'Aucun contenu trouvé dans le PDF',
                'success': False
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Combiner le texte des chunks
        text_content = " ".join([chunk.chunk_text for chunk in chunks])
        
        if not text_content.strip():
            return Response({  # Utiliser Response
                'error': 'Le PDF ne contient pas de texte extractible',
                'success': False
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Générer le résumé
        try:
            summary_service = SummaryService()
            summary = summary_service.generate_summary(text_content)
        except Exception as e:
            print(f"Erreur dans le service de résumé: {e}")
            # Fallback: retourner un extrait
            words = text_content.split()[:150]
            summary = " ".join(words) + "..."
        
        return Response({  # Utiliser Response
            'summary': summary,
            'pdf_id': pdf.id,
            'filename': os.path.basename(pdf.file_path),
            'success': True
        })
        
    except PDFFile.DoesNotExist:
        return Response({  # Utiliser Response
            'error': 'PDF non trouvé',
            'success': False
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        print(f"Erreur générale dans summarize_pdf: {e}")
        return Response({  # Utiliser Response
            'error': f'Erreur lors du résumé: {str(e)}',
            'success': False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
    
    
    