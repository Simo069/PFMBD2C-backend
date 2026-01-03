import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from django.conf import settings
from document.models import Chunk

class VectorDBService:
    """
    Service pour gérer la base de données vectorielle FAISS avec isolation par utilisateur
    """
    
    def __init__(self, dimension: int = None):
        """
        Args:
            dimension: Dimension du vecteur d'embedding (auto-détecté depuis le modèle)
        """
        if dimension is None:
            # Auto-détecter la dimension depuis le service d'embeddings
            from document.services.embedding_service import get_embedding_service
            embedding_service = get_embedding_service()
            dimension = embedding_service.get_embedding_dimension()
        
        self.dimension = dimension
        print(f"VectorDBService initialisé avec dimension: {self.dimension}")
        self.index_dir = os.path.join(settings.MEDIA_ROOT, 'vector_indexes')
        os.makedirs(self.index_dir, exist_ok=True)
    
    def _get_user_index_path(self, user_id: int) -> str:
        """Obtenir le chemin du fichier pour l'index FAISS de l'utilisateur"""
        return os.path.join(self.index_dir, f"user_{user_id}.index")
    
    def _get_user_metadata_path(self, user_id: int) -> str:
        """Obtenir le chemin du fichier pour les métadonnées de l'utilisateur"""
        return os.path.join(self.index_dir, f"user_{user_id}_metadata.pkl")
    
    def _load_user_index(self, user_id: int) -> Tuple[faiss.Index, Dict]:
        """
        Charger l'index FAISS et les métadonnées de l'utilisateur
        
        Returns:
            Tuple de (index, metadata_dict)
        """
        index_path = self._get_user_index_path(user_id)
        metadata_path = self._get_user_metadata_path(user_id)
        
        if os.path.exists(index_path):
            # Charger l'index existant
            index = faiss.read_index(index_path)
            
            # Charger les métadonnées
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            else:
                metadata = {}
        else:
            # Créer un nouvel index
            # Utilisation de IndexFlatL2 pour une recherche exacte
            index = faiss.IndexFlatL2(self.dimension)
            metadata = {}
        
        return index, metadata
    
    def _save_user_index(self, user_id: int, index: faiss.Index, metadata: Dict):
        """
        Sauvegarder l'index FAISS et les métadonnées de l'utilisateur sur le disque
        """
        index_path = self._get_user_index_path(user_id)
        metadata_path = self._get_user_metadata_path(user_id)
        
        # Sauvegarder l'index
        faiss.write_index(index, index_path)
        
        # Sauvegarder les métadonnées
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def add_embeddings(self, user_id: int, chunk_embeddings: Dict[int, List[float]]):
        """
        Ajouter des embeddings à la base de données vectorielle de l'utilisateur
        
        Args:
            user_id: ID utilisateur pour l'isolation
            chunk_embeddings: Dictionnaire associant les IDs de chunks aux embeddings
        """
        # Charger l'index de l'utilisateur
        index, metadata = self._load_user_index(user_id)
        
        # Préparer les vecteurs
        chunk_ids = list(chunk_embeddings.keys())
        vectors = np.array([chunk_embeddings[cid] for cid in chunk_ids], dtype='float32')
        
        # Obtenir la taille actuelle de l'index (pour le mapping)
        current_size = index.ntotal
        
        # Ajouter à l'index FAISS
        index.add(vectors)
        
        # Mettre à jour les métadonnées (mapper les positions d'index FAISS aux IDs de chunks)
        for i, chunk_id in enumerate(chunk_ids):
            faiss_idx = current_size + i
            metadata[faiss_idx] = chunk_id
            
            # Mettre à jour l'enregistrement du chunk avec vector_id
            try:
                chunk = Chunk.objects.get(id=chunk_id)
                chunk.vector_id = str(faiss_idx)
                chunk.save()
            except Chunk.DoesNotExist:
                pass
        
        # Sauvegarder l'index mis à jour
        self._save_user_index(user_id, index, metadata)
    
    def search(self, user_id: int, query_embedding: List[float], top_k: int = 5) -> List[int]:
        """
        Rechercher des chunks similaires dans la base de données vectorielle de l'utilisateur
        
        Args:
            user_id: ID utilisateur pour l'isolation
            query_embedding: Vecteur de requête
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste d'IDs de chunks triés par similarité (plus similaire en premier)
        """
        # Charger l'index de l'utilisateur
        index, metadata = self._load_user_index(user_id)
        
        if index.ntotal == 0:
            return []
        
        # Préparer le vecteur de requête
        query_vector = np.array([query_embedding], dtype='float32')
        
        # Rechercher
        distances, indices = index.search(query_vector, min(top_k, index.ntotal))
        
        # Mapper les indices FAISS aux IDs de chunks
        chunk_ids = []
        for idx in indices[0]:
            if idx in metadata:
                chunk_ids.append(metadata[idx])
        
        return chunk_ids
    
    def delete_pdf_vectors(self, pdf_id: int, user_id: int):
        """
        Supprimer les vecteurs associés à un PDF
        
        Note: FAISS ne supporte pas la suppression efficace, donc nous reconstruisons l'index
        
        Args:
            pdf_id: ID du PDF à supprimer
            user_id: ID utilisateur pour l'isolation
        """
        # Obtenir tous les chunks pour ce PDF
        pdf_chunks = Chunk.objects.filter(pdf_id=pdf_id, user_id=user_id)
        chunk_ids_to_remove = set(chunk.id for chunk in pdf_chunks)
        
        if not chunk_ids_to_remove:
            return
        
        # Charger l'index actuel
        index, metadata = self._load_user_index(user_id)
        
        # Obtenir les chunks restants
        all_user_chunks = Chunk.objects.filter(user_id=user_id).exclude(pdf_id=pdf_id)
        
        if all_user_chunks.count() == 0:
            # Plus de chunks, supprimer les fichiers d'index
            index_path = self._get_user_index_path(user_id)
            metadata_path = self._get_user_metadata_path(user_id)
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            return
        
        # Reconstruire l'index avec les chunks restants
        from document.services.embedding_service import EmbeddingService
        
        # Créer un nouvel index
        new_index = faiss.IndexFlatL2(self.dimension)
        new_metadata = {}
        
        # Ré-embedder les chunks restants
        embedding_service = EmbeddingService()
        chunks_list = list(all_user_chunks)
        chunk_embeddings = embedding_service.embed_chunks(chunks_list)
        
        # Ajouter au nouvel index
        chunk_ids = list(chunk_embeddings.keys())
        vectors = np.array([chunk_embeddings[cid] for cid in chunk_ids], dtype='float32')
        new_index.add(vectors)
        
        # Construire les nouvelles métadonnées
        for i, chunk_id in enumerate(chunk_ids):
            new_metadata[i] = chunk_id
        
        # Sauvegarder le nouvel index
        self._save_user_index(user_id, new_index, new_metadata)
    
    def get_user_index_stats(self, user_id: int) -> Dict:
        """
        Obtenir des statistiques sur la base de données vectorielle de l'utilisateur
        
        Returns:
            Dictionnaire avec les statistiques de l'index
        """
        index, metadata = self._load_user_index(user_id)
        
        return {
            'total_vectors': index.ntotal,
            'dimension': self.dimension,
            'index_type': type(index).__name__
        }



# import faiss
# import numpy as np
# import pickle
# import os
# from typing import List, Dict, Tuple
# from django.conf import settings
# from document.models import Chunk

# class VectorDBService:
#     """
#     Service pour gérer la base de données vectorielle FAISS avec isolation par utilisateur
#     """
    
#     def __init__(self, dimension: int = 768):
#         """
#         Args:
#             dimension: Dimension du vecteur d'embedding (768 pour Gemini)
#         """
#         self.dimension = dimension
#         self.index_dir = os.path.join(settings.MEDIA_ROOT, 'vector_indexes')
#         os.makedirs(self.index_dir, exist_ok=True)
    
#     def _get_user_index_path(self, user_id: int) -> str:
#         """Obtenir le chemin du fichier pour l'index FAISS de l'utilisateur"""
#         return os.path.join(self.index_dir, f"user_{user_id}.index")
    
#     def _get_user_metadata_path(self, user_id: int) -> str:
#         """Obtenir le chemin du fichier pour les métadonnées de l'utilisateur"""
#         return os.path.join(self.index_dir, f"user_{user_id}_metadata.pkl")
    
#     def _load_user_index(self, user_id: int) -> Tuple[faiss.Index, Dict]:
#         """
#         Charger l'index FAISS et les métadonnées de l'utilisateur
        
#         Returns:
#             Tuple de (index, metadata_dict)
#         """
#         index_path = self._get_user_index_path(user_id)
#         metadata_path = self._get_user_metadata_path(user_id)
        
#         if os.path.exists(index_path):
#             # Charger l'index existant
#             index = faiss.read_index(index_path)
            
#             # Charger les métadonnées
#             if os.path.exists(metadata_path):
#                 with open(metadata_path, 'rb') as f:
#                     metadata = pickle.load(f)
#             else:
#                 metadata = {}
#         else:
#             # Créer un nouvel index
#             # Utilisation de IndexFlatL2 pour une recherche exacte
#             index = faiss.IndexFlatL2(self.dimension)
#             metadata = {}
        
#         return index, metadata
    
#     def _save_user_index(self, user_id: int, index: faiss.Index, metadata: Dict):
#         """
#         Sauvegarder l'index FAISS et les métadonnées de l'utilisateur sur le disque
#         """
#         index_path = self._get_user_index_path(user_id)
#         metadata_path = self._get_user_metadata_path(user_id)
        
#         # Sauvegarder l'index
#         faiss.write_index(index, index_path)
        
#         # Sauvegarder les métadonnées
#         with open(metadata_path, 'wb') as f:
#             pickle.dump(metadata, f)
    
#     def add_embeddings(self, user_id: int, chunk_embeddings: Dict[int, List[float]]):
#         """
#         Ajouter des embeddings à la base de données vectorielle de l'utilisateur
        
#         Args:
#             user_id: ID utilisateur pour l'isolation
#             chunk_embeddings: Dictionnaire associant les IDs de chunks aux embeddings
#         """
#         # Charger l'index de l'utilisateur
#         index, metadata = self._load_user_index(user_id)
        
#         # Préparer les vecteurs
#         chunk_ids = list(chunk_embeddings.keys())
#         vectors = np.array([chunk_embeddings[cid] for cid in chunk_ids], dtype='float32')
        
#         # Obtenir la taille actuelle de l'index (pour le mapping)
#         current_size = index.ntotal
        
#         # Ajouter à l'index FAISS
#         index.add(vectors)
        
#         # Mettre à jour les métadonnées (mapper les positions d'index FAISS aux IDs de chunks)
#         for i, chunk_id in enumerate(chunk_ids):
#             faiss_idx = current_size + i
#             metadata[faiss_idx] = chunk_id
            
#             # Mettre à jour l'enregistrement du chunk avec vector_id
#             try:
#                 chunk = Chunk.objects.get(id=chunk_id)
#                 chunk.vector_id = str(faiss_idx)
#                 chunk.save()
#             except Chunk.DoesNotExist:
#                 pass
        
#         # Sauvegarder l'index mis à jour
#         self._save_user_index(user_id, index, metadata)
    
#     def search(self, user_id: int, query_embedding: List[float], top_k: int = 5) -> List[int]:
#         """
#         Rechercher des chunks similaires dans la base de données vectorielle de l'utilisateur
        
#         Args:
#             user_id: ID utilisateur pour l'isolation
#             query_embedding: Vecteur de requête
#             top_k: Nombre de résultats à retourner
            
#         Returns:
#             Liste d'IDs de chunks triés par similarité (plus similaire en premier)
#         """
#         # Charger l'index de l'utilisateur
#         index, metadata = self._load_user_index(user_id)
        
#         if index.ntotal == 0:
#             return []
        
#         # Préparer le vecteur de requête
#         query_vector = np.array([query_embedding], dtype='float32')
        
#         # Rechercher
#         distances, indices = index.search(query_vector, min(top_k, index.ntotal))
        
#         # Mapper les indices FAISS aux IDs de chunks
#         chunk_ids = []
#         for idx in indices[0]:
#             if idx in metadata:
#                 chunk_ids.append(metadata[idx])
        
#         return chunk_ids
    
#     def delete_pdf_vectors(self, pdf_id: int, user_id: int):
#         """
#         Supprimer les vecteurs associés à un PDF
        
#         Note: FAISS ne supporte pas la suppression efficace, donc nous reconstruisons l'index
        
#         Args:
#             pdf_id: ID du PDF à supprimer
#             user_id: ID utilisateur pour l'isolation
#         """
#         # Obtenir tous les chunks pour ce PDF
#         pdf_chunks = Chunk.objects.filter(pdf_id=pdf_id, user_id=user_id)
#         chunk_ids_to_remove = set(chunk.id for chunk in pdf_chunks)
        
#         if not chunk_ids_to_remove:
#             return
        
#         # Charger l'index actuel
#         index, metadata = self._load_user_index(user_id)
        
#         # Obtenir les chunks restants
#         all_user_chunks = Chunk.objects.filter(user_id=user_id).exclude(pdf_id=pdf_id)
        
#         if all_user_chunks.count() == 0:
#             # Plus de chunks, supprimer les fichiers d'index
#             index_path = self._get_user_index_path(user_id)
#             metadata_path = self._get_user_metadata_path(user_id)
#             if os.path.exists(index_path):
#                 os.remove(index_path)
#             if os.path.exists(metadata_path):
#                 os.remove(metadata_path)
#             return
        
#         # Reconstruire l'index avec les chunks restants
#         from document.services.embedding_service import EmbeddingService
        
#         # Créer un nouvel index
#         new_index = faiss.IndexFlatL2(self.dimension)
#         new_metadata = {}
        
#         # Ré-embedder les chunks restants
#         embedding_service = EmbeddingService()
#         chunks_list = list(all_user_chunks)
#         chunk_embeddings = embedding_service.embed_chunks(chunks_list)
        
#         # Ajouter au nouvel index
#         chunk_ids = list(chunk_embeddings.keys())
#         vectors = np.array([chunk_embeddings[cid] for cid in chunk_ids], dtype='float32')
#         new_index.add(vectors)
        
#         # Construire les nouvelles métadonnées
#         for i, chunk_id in enumerate(chunk_ids):
#             new_metadata[i] = chunk_id
        
#         # Sauvegarder le nouvel index
#         self._save_user_index(user_id, new_index, new_metadata)
    
#     def get_user_index_stats(self, user_id: int) -> Dict:
#         """
#         Obtenir des statistiques sur la base de données vectorielle de l'utilisateur
        
#         Returns:
#             Dictionnaire avec les statistiques de l'index
#         """
#         index, metadata = self._load_user_index(user_id)
        
#         return {
#             'total_vectors': index.ntotal,
#             'dimension': self.dimension,
#             'index_type': type(index).__name__
#         }