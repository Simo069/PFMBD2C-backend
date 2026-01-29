from django.db import models
from django.conf import settings

class PDFFile(models.Model):
    """
    Modèle pour stocker les métadonnées des fichiers PDF
    """
    STATUS_CHOICES = [
        ('pending', 'En attente'),
        ('processing', 'En traitement'),
        ('completed', 'Terminé'),
        ('failed', 'Échoué'),
    ]
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='pdf_files',
        verbose_name="Utilisateur"
    )
    filename = models.CharField(max_length=255, verbose_name="Nom du fichier")
    original_filename = models.CharField(max_length=255, verbose_name="Nom original")
    file_path = models.CharField(max_length=500, verbose_name="Chemin du fichier")
    file_size = models.BigIntegerField(verbose_name="Taille du fichier")
    upload_date = models.DateTimeField(auto_now_add=True, verbose_name="Date d'upload")
    processing_status = models.CharField(
        max_length=50, 
        choices=STATUS_CHOICES, 
        default='pending',
        verbose_name="Statut de traitement"
    )
    page_count = models.IntegerField(null=True, blank=True, verbose_name="Nombre de pages")
    total_chunks = models.IntegerField(default=0, verbose_name="Nombre total de chunks")
    metadata = models.JSONField(null=True, blank=True, verbose_name="Métadonnées")
    
    class Meta:
        db_table = 'pdf_files'
        ordering = ['-upload_date']
        verbose_name = 'Fichier PDF'
        verbose_name_plural = 'Fichiers PDF'
        indexes = [
            models.Index(fields=['user', 'processing_status']),
            models.Index(fields=['upload_date']),
        ]
    
    def __str__(self):
        return f"{self.original_filename} - {self.user.username}"


class Chunk(models.Model):
    """
    Modèle pour stocker les chunks de texte et leurs métadonnées
    """
    pdf = models.ForeignKey(
        PDFFile, 
        on_delete=models.CASCADE, 
        related_name='chunks',
        verbose_name="PDF"
    )
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='chunks',
        verbose_name="Utilisateur"
    )
    chunk_text = models.TextField(verbose_name="Texte du chunk")
    chunk_index = models.IntegerField(verbose_name="Index du chunk")
    page_number = models.IntegerField(null=True, blank=True, verbose_name="Numéro de page")
    start_char = models.IntegerField(null=True, blank=True, verbose_name="Caractère de début")
    end_char = models.IntegerField(null=True, blank=True, verbose_name="Caractère de fin")
    token_count = models.IntegerField(null=True, blank=True, verbose_name="Nombre de tokens")
    vector_id = models.CharField(
        max_length=255, 
        null=True, 
        blank=True,
        verbose_name="ID du vecteur"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Date de création")
    
    class Meta:
        db_table = 'chunks'
        ordering = ['pdf', 'chunk_index']
        verbose_name = 'Chunk'
        verbose_name_plural = 'Chunks'
        indexes = [
            models.Index(fields=['pdf', 'chunk_index']),
            models.Index(fields=['user']),
            models.Index(fields=['vector_id']),
        ]
    
    def __str__(self):
        return f"Chunk {self.chunk_index} de {self.pdf.original_filename}"