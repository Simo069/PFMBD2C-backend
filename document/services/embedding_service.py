from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import torch

class EmbeddingService:
    """
    Service pour générer des embeddings avec SentenceTransformers
    Utilise un modèle multilingue performant
    """
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialise le modèle SentenceTransformers
        Args:
            model_name: Nom du modèle à utiliser
                Options recommandées:
                - 'paraphrase-multilingual-mpnet-base-v2' (768 dim, multilingue)
                - 'all-MiniLM-L6-v2' (384 dim, anglais uniquement, plus rapide)
                - 'distiluse-base-multilingual-cased-v2' (512 dim, multilingue)
        """
        print(f"Chargement du modèle SentenceTransformers: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Modèle chargé. Dimension des embeddings: {self.dimension}")
        
        # Utiliser GPU si disponible
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        print(f"Device utilisé: {device}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Génère un embedding pour un texte unique
        
        Args:
            text: Texte à transformer en embedding
            
        Returns:
            Liste de valeurs d'embedding (vecteur)
        """
        try:
            # Encoder le texte
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Échec de génération de l'embedding: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Génère des embeddings pour plusieurs textes par lots
        SentenceTransformers est optimisé pour le traitement par batch
        
        Args:
            texts: Liste de textes à transformer en embeddings
            batch_size: Nombre de textes à traiter en une seule fois
            
        Returns:
            Liste de vecteurs d'embedding
        """
        try:
            # Encoder tous les textes en une seule fois (optimisé)
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True if len(texts) > 100 else False
            )
            return embeddings.tolist()
        except Exception as e:
            raise Exception(f"Échec de génération des embeddings batch: {str(e)}")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Génère un embedding pour une requête de recherche
        
        Note: Avec SentenceTransformers, pas de différence entre
        document et query embeddings (contrairement à Gemini)
        
        Args:
            query: Requête de recherche de l'utilisateur
            
        Returns:
            Vecteur d'embedding de la requête
        """
        return self.generate_embedding(query)
    
    def embed_chunks(self, chunks) -> Dict[int, List[float]]:
        """
        Génère des embeddings pour une liste d'objets Chunk
        
        Args:
            chunks: Liste d'instances du modèle Chunk
            
        Returns:
            Dictionnaire associant les IDs de chunks aux embeddings
        """
        if not chunks:
            return {}
        
        # Extraire les textes et IDs
        texts = [chunk.chunk_text for chunk in chunks]
        chunk_ids = [chunk.id for chunk in chunks]
        
        print(f"Génération d'embeddings pour {len(texts)} chunks...")
        
        # Générer les embeddings (batch processing automatique)
        embeddings = self.generate_embeddings_batch(texts, batch_size=32)
        
        # Associer les IDs de chunks aux embeddings
        chunk_embeddings = {}
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            chunk_embeddings[chunk_id] = embedding
        
        print(f"✓ {len(chunk_embeddings)} embeddings générés")
        return chunk_embeddings
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs
        
        Args:
            vec1: Premier vecteur
            vec2: Deuxième vecteur
            
        Returns:
            Score de similarité (0 à 1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_embedding_dimension(self) -> int:
        """
        Retourne la dimension des embeddings du modèle
        
        Returns:
            Dimension (ex: 768 pour mpnet, 384 pour MiniLM)
        """
        return self.dimension


# Instance globale du service (singleton pattern)
# Le modèle est chargé une seule fois au démarrage
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """
    Récupère l'instance singleton du service d'embeddings
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service





# import google.generativeai as genai
# from typing import List, Dict
# import numpy as np
# from django.conf import settings

# class EmbeddingService:
#     """
#     Service pour générer des embeddings avec l'API Gemini
#     """
    
#     def __init__(self):
#         """
#         Initialise l'API Gemini avec les identifiants
#         """
#         genai.configure(api_key=settings.GEMINI_API_KEY)
#         self.model_name = "models/embedding-001"
    
#     def generate_embedding(self, text: str) -> List[float]:
#         """
#         Génère un embedding pour un texte unique
        
#         Args:
#             text: Texte à transformer en embedding
            
#         Returns:
#             Liste de valeurs d'embedding (vecteur 768-dimensions)
#         """
#         try:
#             result = genai.embed_content(
#                 model=self.model_name,
#                 content=text,
#                 task_type="retrieval_document"  # Optimisé pour la récupération de documents
#             )
#             return result['embedding']
#         except Exception as e:
#             raise Exception(f"Échec de génération de l'embedding: {str(e)}")
    
#     def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
#         """
#         Génère des embeddings pour plusieurs textes par lots
        
#         Args:
#             texts: Liste de textes à transformer en embeddings
#             batch_size: Nombre de textes à traiter en une seule fois
            
#         Returns:
#             Liste de vecteurs d'embedding
#         """
#         all_embeddings = []
        
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
            
#             try:
#                 # Traiter le lot
#                 batch_embeddings = []
#                 for text in batch:
#                     embedding = self.generate_embedding(text)
#                     batch_embeddings.append(embedding)
                
#                 all_embeddings.extend(batch_embeddings)
                
#             except Exception as e:
#                 print(f"Erreur lors du traitement du lot {i // batch_size}: {str(e)}")
#                 # Continuer avec le lot suivant même si un lot échoue
#                 continue
        
#         return all_embeddings
    
#     def generate_query_embedding(self, query: str) -> List[float]:
#         """
#         Génère un embedding pour une requête de recherche
        
#         Args:
#             query: Requête de recherche de l'utilisateur
            
#         Returns:
#             Vecteur d'embedding de la requête
#         """
#         try:
#             result = genai.embed_content(
#                 model=self.model_name,
#                 content=query,
#                 task_type="retrieval_query"  # Optimisé pour les requêtes
#             )
#             return result['embedding']
#         except Exception as e:
#             raise Exception(f"Échec de génération de l'embedding de requête: {str(e)}")
    
#     def embed_chunks(self, chunks) -> Dict[int, List[float]]:
#         """
#         Génère des embeddings pour une liste d'objets Chunk
        
#         Args:
#             chunks: Liste d'instances du modèle Chunk
            
#         Returns:
#             Dictionnaire associant les IDs de chunks aux embeddings
#         """
#         # Extraire les textes et IDs
#         texts = [chunk.chunk_text for chunk in chunks]
#         chunk_ids = [chunk.id for chunk in chunks]
        
#         # Générer les embeddings
#         embeddings = self.generate_embeddings_batch(texts)
        
#         # Associer les IDs de chunks aux embeddings
#         chunk_embeddings = {}
#         for chunk_id, embedding in zip(chunk_ids, embeddings):
#             chunk_embeddings[chunk_id] = embedding
        
#         return chunk_embeddings
    
#     @staticmethod
#     def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
#         """
#         Calcule la similarité cosinus entre deux vecteurs
        
#         Args:
#             vec1: Premier vecteur
#             vec2: Deuxième vecteur
            
#         Returns:
#             Score de similarité (0 à 1)
#         """
#         vec1 = np.array(vec1)
#         vec2 = np.array(vec2)
        
#         dot_product = np.dot(vec1, vec2)
#         norm1 = np.linalg.norm(vec1)
#         norm2 = np.linalg.norm(vec2)
        
#         if norm1 == 0 or norm2 == 0:
#             return 0.0
        
#         return dot_product / (norm1 * norm2)