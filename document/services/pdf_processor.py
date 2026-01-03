import re
import PyPDF2
from typing import List, Dict
from document.models import PDFFile, Chunk

class PDFProcessor:
    """
    Gère l'extraction de texte et le chunking des PDFs
    """
    
    def __init__(self, chunk_size=800, chunk_overlap=100):
        """
        Args:
            chunk_size: Taille cible de chaque chunk en caractères
            chunk_overlap: Nombre de caractères de chevauchement entre chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """
        Extrait le texte du PDF, organisé par page
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Dictionnaire associant les numéros de page au texte
        """
        page_texts = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Nettoyer le texte extrait
                    text = self.clean_text(text)
                    page_texts[page_num + 1] = text  # Numéros de page indexés à partir de 1
                
        except Exception as e:
            raise Exception(f"Échec de l'extraction de texte du PDF: {str(e)}")
        
        return page_texts
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte extrait en supprimant les espaces excessifs
        
        Args:
            text: Texte brut du PDF
            
        Returns:
            Texte nettoyé
        """
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les sauts de ligne multiples
        text = re.sub(r'\n+', '\n', text)
        
        # Supprimer les espaces en début et fin
        text = text.strip()
        
        return text
    
    def create_chunks(self, page_texts: Dict[int, str]) -> List[Dict]:
        """
        Divise le texte en chunks avec chevauchement
        
        Args:
            page_texts: Dictionnaire associant les numéros de page au texte
            
        Returns:
            Liste de dictionnaires de chunks avec métadonnées
        """
        chunks = []
        chunk_index = 0
        
        # Combiner toutes les pages en un seul texte pour le chunking
        full_text = ""
        page_char_ranges = {}  # Suivre quels caractères appartiennent à quelle page
        current_pos = 0
        
        for page_num, page_text in sorted(page_texts.items()):
            page_start = current_pos
            full_text += page_text + "\n"
            page_end = len(full_text)
            page_char_ranges[page_num] = (page_start, page_end)
            current_pos = page_end
        
        # Créer des chunks avec chevauchement
        start = 0
        while start < len(full_text):
            end = start + self.chunk_size
            
            # Essayer de couper à une limite de phrase
            if end < len(full_text):
                sentence_end = self._find_sentence_boundary(full_text, end)
                if sentence_end > start:
                    end = sentence_end
            
            chunk_text = full_text[start:end].strip()
            
            if chunk_text:  # Ajouter uniquement les chunks non vides
                # Déterminer à quelle page ce chunk appartient principalement
                chunk_middle = start + (end - start) // 2
                page_num = self._get_page_for_position(chunk_middle, page_char_ranges)
                
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'page_number': page_num,
                    'start_char': start,
                    'end_char': end,
                    'token_count': self._estimate_tokens(chunk_text)
                })
                chunk_index += 1
            
            # Déplacer la position de départ avec chevauchement
            start = end - self.chunk_overlap
            
            # S'assurer de progresser même avec de très petits chunks
            if start <= 0 or start >= len(full_text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, position: int, search_window: int = 100) -> int:
        """
        Trouve la limite de phrase la plus proche (., !, ?) près de la position donnée
        """
        search_start = max(0, position - search_window)
        search_end = min(len(text), position + search_window)
        search_text = text[search_start:search_end]
        
        # Chercher les fins de phrase
        sentence_endings = ['.', '!', '?', '\n']
        best_pos = position
        
        for i in range(len(search_text) - 1, -1, -1):
            if search_text[i] in sentence_endings:
                best_pos = search_start + i + 1
                break
        
        return best_pos
    
    def _get_page_for_position(self, position: int, page_char_ranges: Dict[int, tuple]) -> int:
        """
        Détermine à quelle page appartient une position de caractère
        """
        for page_num, (start, end) in page_char_ranges.items():
            if start <= position < end:
                return page_num
        return list(page_char_ranges.keys())[-1]  # Par défaut, dernière page
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimation approximative du nombre de tokens (1 token ≈ 4 caractères pour le français)
        """
        return len(text) // 4
    
    def process_and_save_chunks(self, pdf_file: PDFFile) -> List[Chunk]:
        """
        Pipeline complet de traitement: extraire, chunker et sauvegarder dans la base de données
        
        Args:
            pdf_file: Instance du modèle PDFFile
            
        Returns:
            Liste des instances Chunk créées
        """
        try:
            # Mettre à jour le statut
            pdf_file.processing_status = 'processing'
            pdf_file.save()
            
            # Extraire le texte
            page_texts = self.extract_text_from_pdf(pdf_file.file_path)
            pdf_file.page_count = len(page_texts)
            
            # Créer les chunks
            chunk_data = self.create_chunks(page_texts)
            
            # Sauvegarder les chunks dans la base de données
            chunk_objects = []
            for chunk_info in chunk_data:
                chunk = Chunk.objects.create(
                    pdf=pdf_file,
                    user=pdf_file.user,
                    chunk_text=chunk_info['text'],
                    chunk_index=chunk_info['chunk_index'],
                    page_number=chunk_info['page_number'],
                    start_char=chunk_info['start_char'],
                    end_char=chunk_info['end_char'],
                    token_count=chunk_info['token_count']
                )
                chunk_objects.append(chunk)
            
            # Mettre à jour l'enregistrement PDF
            pdf_file.total_chunks = len(chunk_objects)
            pdf_file.processing_status = 'completed'
            pdf_file.save()
            
            return chunk_objects
            
        except Exception as e:
            pdf_file.processing_status = 'failed'
            pdf_file.metadata = {'error': str(e)}
            pdf_file.save()
            raise e