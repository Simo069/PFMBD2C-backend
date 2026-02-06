# import google.generativeai as genai
# from typing import Optional
# import os
# from django.conf import settings

# class SummaryService:
#     """
#     Service pour générer des résumés de documents avec Gemini
#     """
    
#     def __init__(self, model_name: str = "models/gemini-flash-latest"):
#         """
#         Initialise le service Gemini pour les résumés
        
#         Args:
#             model_name: Nom du modèle Gemini à utiliser
#         """
#         # Configurer l'API Gemini
#         api_key = os.getenv('GEMINI_API_KEY')
#         if not api_key:
#             raise ValueError("Clé API Gemini non configurée")
        
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel(model_name)
#         print(f"Service de résumé initialisé avec le modèle {model_name}")
    
#     def generate_summary(self, text: str, max_length: int = 500) -> str:
#         """
#         Génère un résumé concis d'un texte
        
#         Args:
#             text: Texte à résumer
#             max_length: Longueur maximale du résumé en mots
            
#         Returns:
#             Résumé du texte
#         """
#         try:
#             # Préparer le prompt pour le résumé
#             prompt = f"""
#             Résumez le texte suivant de manière concise et structurée.
#             Le résumé doit être en français et ne doit pas dépasser {max_length} mots.
            
#             Structure demandée :
#             1. Sujet principal (1-2 phrases)
#             2. Points clés (3-5 points maximum)
#             3. Conclusion principale
            
#             Texte à résumer :
#             {text[:10000]}  # Limiter la taille pour éviter les tokens excessifs
#             """
            
#             # Générer le résumé
#             response = self.model.generate_content(prompt)
            
#             # Extraire le texte de la réponse
#             if response.text:
#                 return response.text
#             else:
#                 return "Impossible de générer un résumé pour ce document."
                
#         except Exception as e:
#             print(f"Erreur lors de la génération du résumé: {e}")
#             return f"Erreur lors de la génération du résumé: {str(e)}"
    
#     def generate_detailed_summary(self, text: str, sections: list = None) -> str:
#         """
#         Génère un résumé détaillé avec sections spécifiques
        
#         Args:
#             text: Texte à résumer
#             sections: Liste des sections à inclure dans le résumé
            
#         Returns:
#             Résumé détaillé structuré
#         """
#         if sections is None:
#             sections = ["Introduction", "Méthodologie", "Résultats", "Conclusion"]
        
#         sections_str = "\n".join([f"- {section}" for section in sections])
        
#         prompt = f"""
#         Générer un résumé détaillé et structuré du texte suivant.
#         Organisez le résumé selon les sections suivantes :
        
#         {sections_str}
        
#         Pour chaque section, fournissez 2-3 points essentiels.
#         Le résumé doit être en français et complet.
        
#         Texte à résumer :
#         {text[:8000]}
#         """
        
#         try:
#             response = self.model.generate_content(prompt)
#             return response.text if response.text else "Résumé non disponible."
#         except Exception as e:
#             return f"Erreur : {str(e)}"









#test diali de nouveau package 
from google import genai
from typing import Optional
import os
from django.conf import settings


class SummaryService:
    """
    Service pour générer des résumés de documents avec Gemini
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialise le service Gemini pour les résumés

        Args:
            model_name: Nom du modèle Gemini à utiliser
        """
        api_key = os.getenv("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)

        if not api_key:
            raise ValueError("Clé API Gemini non configurée")

        # Initialisation du client Gemini (nouvelle API)
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        print(f"Service de résumé initialisé avec le modèle {model_name}")

    def generate_summary(self, text: str, max_length: int = 500) -> str:
        """
        Génère un résumé concis d'un texte
        """
        try:
            prompt = f"""
Résumez le texte suivant de manière concise et structurée.
Le résumé doit être en français et ne doit pas dépasser {max_length} mots.

Structure demandée :
1. Sujet principal (1-2 phrases)
2. Points clés (3-5 points maximum)
3. Conclusion principale

Texte à résumer :
{text[:10000]}
"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            return response.text or "Impossible de générer un résumé pour ce document."

        except Exception as e:
            print(f"Erreur lors de la génération du résumé: {e}")
            return f"Erreur lors de la génération du résumé: {str(e)}"

    def generate_detailed_summary(self, text: str, sections: list = None) -> str:
        """
        Génère un résumé détaillé avec sections spécifiques
        """
        if sections is None:
            sections = ["Introduction", "Méthodologie", "Résultats", "Conclusion"]

        sections_str = "\n".join([f"- {section}" for section in sections])

        prompt = f"""
Générer un résumé détaillé et structuré du texte suivant.
Organisez le résumé selon les sections suivantes :

{sections_str}

Pour chaque section, fournissez 2-3 points essentiels.
Le résumé doit être en français et complet.

Texte à résumer :
{text[:8000]}
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text or "Résumé non disponible."
        except Exception as e:
            return f"Erreur : {str(e)}"
