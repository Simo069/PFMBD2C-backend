from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    """
    Modèle utilisateur personnalisé
    """
    email = models.EmailField(unique=True, verbose_name="Email")
    full_name = models.CharField(max_length=255, blank=True, verbose_name="Nom complet")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Date de création")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Date de modification")
    
    # Préférences utilisateur pour le RAG
    chunk_size = models.IntegerField(default=800, verbose_name="Taille des chunks")
    chunk_overlap = models.IntegerField(default=100, verbose_name="Chevauchement des chunks")
    retrieval_k = models.IntegerField(default=5, verbose_name="Nombre de chunks à récupérer")
    
    class Meta:
        db_table = 'users'
        verbose_name = 'Utilisateur'
        verbose_name_plural = 'Utilisateurs'
    
    def __str__(self):
        return self.username