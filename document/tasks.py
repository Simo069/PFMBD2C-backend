from celery import shared_task
from document.models import PDFFile
from document.services.pdf_processor import PDFProcessor
from document.services.embedding_service import get_embedding_service
from document.services.vector_db_service import VectorDBService

@shared_task
def process_pdf_async(pdf_id):
    """
    Tâche en arrière-plan pour traiter le PDF: extraire le texte, chunker, embedder et stocker les vecteurs
    
    Args:
        pdf_id: ID du PDF à traiter
    """
    try:
        print(f"[TASK] Démarrage du traitement du PDF {pdf_id}")
        
        # Obtenir l'enregistrement PDF
        pdf_file = PDFFile.objects.get(id=pdf_id)
        
        # Étape 1: Extraire le texte et créer les chunks
        print(f"[TASK] Étape 1/3: Extraction et chunking...")
        pdf_processor = PDFProcessor(chunk_size=800, chunk_overlap=100)
        chunks = pdf_processor.process_and_save_chunks(pdf_file)
        
        if not chunks:
            pdf_file.processing_status = 'failed'
            pdf_file.metadata = {'error': 'Aucun texte n\'a pu être extrait du PDF'}
            pdf_file.save()
            print(f"[TASK] ✗ Échec: Aucun texte extrait")
            return
        
        print(f"[TASK] ✓ {len(chunks)} chunks créés")
        
        # Étape 2: Générer les embeddings avec SentenceTransformers
        print(f"[TASK] Étape 2/3: Génération des embeddings (SentenceTransformers)...")
        embedding_service = get_embedding_service()
        chunk_embeddings = embedding_service.embed_chunks(chunks)
        
        print(f"[TASK] ✓ {len(chunk_embeddings)} embeddings générés")
        
        # Étape 3: Stocker dans la base de données vectorielle
        print(f"[TASK] Étape 3/3: Stockage dans FAISS...")
        vector_db_service = VectorDBService()
        vector_db_service.add_embeddings(pdf_file.user_id, chunk_embeddings)
        
        print(f"[TASK] ✓ Vecteurs stockés dans FAISS")
        
        # Mettre à jour le statut
        pdf_file.processing_status = 'completed'
        pdf_file.save()
        
        print(f"[TASK] ✓ Traitement terminé avec succès pour PDF {pdf_id}")
        
        return {
            'pdf_id': pdf_id,
            'status': 'completed',
            'total_chunks': len(chunks)
        }
        
    except PDFFile.DoesNotExist:
        print(f"[TASK] ✗ Erreur: PDF {pdf_id} introuvable")
        return {'error': f'PDF avec id {pdf_id} introuvable'}
    except Exception as e:
        print(f"[TASK] ✗ Erreur lors du traitement: {str(e)}")
        # Mettre à jour le statut en échec
        try:
            pdf_file = PDFFile.objects.get(id=pdf_id)
            pdf_file.processing_status = 'failed'
            pdf_file.metadata = {'error': str(e)}
            pdf_file.save()
        except:
            pass
        
        raise e


