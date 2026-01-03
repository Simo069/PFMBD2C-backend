# import google.generativeai as genai
from google import genai
from typing import List, Dict, Optional
from django.conf import settings
from document.models import Chunk
from chat.models import Message, ChatSession
from document.services.vector_db_service import VectorDBService

class RAGService:
    """
    Service pour la génération augmentée par récupération (RAG) avec Gemini
    """
    
    def __init__(self):
        self.client = genai.Client(api_key='AIzaSyDYW-fW8whxic6gE8qSqarP-9J-JqwZRBA')
        self.model_name = "gemini-2.5-flash"

        from document.services.embedding_service import get_embedding_service
        self.embedding_service = get_embedding_service()
        self.vector_db_service = VectorDBService()
    
    def ask_question(
        self, 
        user_id: int, 
        question: str, 
        session_id: Optional[int] = None,
        pdf_ids: Optional[List[int]] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Répondre à une question en utilisant RAG
        
        Args:
            user_id: ID utilisateur pour l'isolation des données
            question: Question de l'utilisateur
            session_id: ID de session de chat optionnel
            pdf_ids: Liste optionnelle d'IDs de PDF à rechercher
            top_k: Nombre de chunks à récupérer
            
        Returns:
            Dictionnaire avec la réponse, les sources et les métadonnées
        """
        # Étape 1: Générer l'embedding de la requête
        query_embedding = self.embedding_service.generate_query_embedding(question)
        
        # Étape 2: Récupérer les chunks pertinents
        chunk_ids = self.vector_db_service.search(user_id, query_embedding, top_k)
        
        if not chunk_ids:
            return {
                'answer': "Je n'ai pas pu trouver d'informations pertinentes dans vos documents pour répondre à cette question.",
                'sources': [],
                'chunk_ids': []
            }
        
        # Étape 3: Obtenir les détails des chunks
        chunks = Chunk.objects.filter(id__in=chunk_ids, user_id=user_id)
        
        # Filtrer par IDs de PDF si spécifié
        if pdf_ids:
            chunks = chunks.filter(pdf_id__in=pdf_ids)
        
        # Préserver l'ordre de la recherche de similarité
        chunks_dict = {chunk.id: chunk for chunk in chunks}
        ordered_chunks = [chunks_dict[cid] for cid in chunk_ids if cid in chunks_dict]
        
        if not ordered_chunks:
            return {
                'answer': "Aucune information pertinente trouvée dans les documents spécifiés.",
                'sources': [],
                'chunk_ids': []
            }
        
        # Étape 4: Construire le prompt avec le contexte
        context = self._build_context(ordered_chunks)
        prompt = self._build_prompt(question, context)
        
        # Étape 5: Générer la réponse avec Gemini
        try:
            # response = self.model.generate_content(prompt)
            # answer = response.text
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            answer = response.text
        except Exception as e:
            return {
                'answer': f"Erreur lors de la génération de la réponse: {str(e)}",
                'sources': [],
                'chunk_ids': []
            }
        
        # Étape 6: Préparer les informations sur les sources
        sources = self._prepare_sources(ordered_chunks)
        
        # Étape 7: Sauvegarder dans l'historique du chat si une session est fournie
        if session_id:
            self._save_to_chat_history(
                session_id=session_id,
                user_id=user_id,
                question=question,
                answer=answer,
                chunk_ids=[c.id for c in ordered_chunks]
            )
        
        return {
            'answer': answer,
            'sources': sources,
            'chunk_ids': [c.id for c in ordered_chunks]
        }
    
    def _build_context(self, chunks: List[Chunk]) -> str:
        """
        Construire une chaîne de contexte à partir des chunks récupérés
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i} - {chunk.pdf.original_filename}, Page {chunk.page_number}]\n"
                f"{chunk.chunk_text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Construire le prompt final pour Gemini
        """
        prompt = f"""Vous êtes un assistant IA utile qui répond aux questions basées UNIQUEMENT sur le contexte fourni à partir des documents PDF.

Contexte des documents:
{context}

Question de l'utilisateur: {question}

Instructions:
1. Répondez à la question en utilisant UNIQUEMENT les informations fournies dans le contexte ci-dessus
2. Si le contexte ne contient pas suffisamment d'informations pour répondre à la question, dites-le clairement
3. Soyez concis et précis
4. Si vous référencez des informations spécifiques, mentionnez de quelle source elles proviennent
5. Ne fabriquez pas et n'inférez pas d'informations qui ne sont pas explicitement mentionnées dans le contexte

Réponse:"""
        
        return prompt
    
    def _prepare_sources(self, chunks: List[Chunk]) -> List[Dict]:
        """
        Préparer les informations sur les sources pour la réponse
        """
        sources = []
        
        for chunk in chunks:
            sources.append({
                'pdf_id': chunk.pdf.id,
                'pdf_filename': chunk.pdf.original_filename,
                'page_number': chunk.page_number,
                'chunk_text': chunk.chunk_text[:200] + "...",  # Aperçu
                'chunk_id': chunk.id
            })
        
        return sources
    
    def _save_to_chat_history(
        self, 
        session_id: int, 
        user_id: int, 
        question: str, 
        answer: str, 
        chunk_ids: List[int]
    ):
        """
        Sauvegarder la question et la réponse dans l'historique du chat
        """
        # Sauvegarder le message de l'utilisateur
        Message.objects.create(
            session_id=session_id,
            user_id=user_id,
            role='user',
            content=question
        )
        
        # Sauvegarder le message de l'assistant
        Message.objects.create(
            session_id=session_id,
            user_id=user_id,
            role='assistant',
            content=answer,
            chunk_ids=chunk_ids
        )
        
        # Mettre à jour l'horodatage de la session
        try:
            session = ChatSession.objects.get(id=session_id)
            session.save()  # Cela met à jour le champ updated_at
        except ChatSession.DoesNotExist:
            pass
    
    def generate_summary(self, user_id: int, pdf_id: int) -> str:
        """
        Générer un résumé d'un document PDF
        
        Args:
            user_id: ID utilisateur pour l'isolation des données
            pdf_id: ID du PDF à résumer
            
        Returns:
            Texte du résumé
        """
        # Obtenir tous les chunks pour ce PDF
        chunks = Chunk.objects.filter(pdf_id=pdf_id, user_id=user_id).order_by('chunk_index')
        
        if not chunks:
            return "Aucun contenu trouvé à résumer."
        
        # Combiner les chunks (limiter pour éviter les limites de tokens)
        combined_text = ""
        for chunk in chunks[:20]:  # Limiter aux 20 premiers chunks
            combined_text += chunk.chunk_text + "\n\n"
        
        # Générer le résumé
        prompt = f"""Veuillez fournir un résumé complet du contenu du document suivant:

{combined_text}

Résumé:"""
        
        try:
            # response = self.model.generate_content(prompt)
            # return response.text
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Erreur lors de la génération du résumé: {str(e)}"
    
    def generate_mindmap(self, user_id: int, pdf_id: int) -> Dict:
        """
        Générer une structure de carte mentale à partir d'un PDF
        
        Args:
            user_id: ID utilisateur pour l'isolation des données
            pdf_id: ID du PDF pour créer la carte mentale
            
        Returns:
            Dictionnaire représentant la structure de la carte mentale
        """
        # Obtenir tous les chunks pour ce PDF
        chunks = Chunk.objects.filter(pdf_id=pdf_id, user_id=user_id).order_by('chunk_index')
        
        if not chunks:
            return {"error": "Aucun contenu trouvé"}
        
        # Combiner les chunks (limiter pour éviter les limites de tokens)
        combined_text = ""
        for chunk in chunks[:15]:
            combined_text += chunk.chunk_text + "\n\n"
        
        # Générer la structure de la carte mentale
        prompt = f"""Basé sur le contenu du document suivant, créez une structure de carte mentale hiérarchique.
Identifiez les sujets principaux et leurs sous-sujets.

Contenu du document:
{combined_text}

Veuillez fournir la carte mentale au format JSON suivant:
{{
    "central_topic": "Sujet principal du document",
    "branches": [
        {{
            "topic": "Branche 1",
            "subtopics": ["Sous-sujet 1.1", "Sous-sujet 1.2"]
        }},
        {{
            "topic": "Branche 2",
            "subtopics": ["Sous-sujet 2.1", "Sous-sujet 2.2"]
        }}
    ]
}}

JSON de la carte mentale:"""
        
        try:
            # response = self.model.generate_content(prompt)
            # return {"structure": response.text}
            response = self.client.models.generate_content(
                model=self.model_name,
                ontents=prompt
            )
            return {"structure": response.text}
        except Exception as e:
            return {"error": f"Erreur lors de la génération de la carte mentale: {str(e)}"}

