from django.db import models
from django.conf import settings
from django.contrib.postgres.fields import ArrayField


class ChatSession(models.Model):
    """
    Représente une session de conversation
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='chat_sessions',
        verbose_name="Utilisateur"
    )
    title = models.CharField(max_length=255, blank=True, verbose_name="Titre")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Date de création")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Dernière mise à jour")
    is_active = models.BooleanField(default=True, verbose_name="Active")
    pdf_ids = ArrayField(
        models.IntegerField(),
        blank=True,
        default=list,
        verbose_name="IDs des PDFs"
    )
    
    class Meta:
        db_table = 'chat_sessions'
        ordering = ['-updated_at']
        verbose_name = 'Session de chat'
        verbose_name_plural = 'Sessions de chat'
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['updated_at']),
        ]
    
    def __str__(self):
        return f"Session {self.id} - {self.user.username}"


class Message(models.Model):
    """
    Stocke les messages individuels dans une session de chat
    """
    ROLE_CHOICES = [
        ('user', 'Utilisateur'),
        ('assistant', 'Assistant'),
    ]
    
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name="Session"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name="Utilisateur"
    )
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        verbose_name="Rôle"
    )
    content = models.TextField(verbose_name="Contenu")
    chunk_ids = ArrayField(
        models.IntegerField(),
        blank=True,
        default=list,
        verbose_name="IDs des chunks utilisés"
    )
    tokens_used = models.IntegerField(
        null=True,
        blank=True,
        verbose_name="Tokens utilisés"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Date de création")
    metadata = models.JSONField(
        null=True,
        blank=True,
        verbose_name="Métadonnées"
    )
    
    class Meta:
        db_table = 'messages'
        ordering = ['created_at']
        verbose_name = 'Message'
        verbose_name_plural = 'Messages'
        indexes = [
            models.Index(fields=['session', 'created_at']),
            models.Index(fields=['user']),
        ]
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
    



class MessageFeedback(models.Model):
    """
    Représente un feedback utilisateur sur un message spécifique
    """
    FEEDBACK_CHOICES = [
        ('positive', 'Utile'),
        ('negative', 'Incorrecte'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='message_feedbacks',
        verbose_name="Utilisateur"
    )

    message = models.ForeignKey(
        'Message',
        on_delete=models.CASCADE,
        related_name='feedbacks',
        verbose_name="Message concerné"
    )

    feedback = models.CharField(
        max_length=20,
        choices=FEEDBACK_CHOICES,
        verbose_name="Type de feedback"
    )

    comment = models.TextField(
        blank=True,
        null=True,
        verbose_name="Commentaire optionnel"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'message_feedbacks'
        unique_together = ('user', 'message')
        verbose_name = 'Feedback message'
        verbose_name_plural = 'Feedbacks messages'

    def __str__(self):
        return f"{self.feedback} - msg {self.message.id} - {self.user.username}"
