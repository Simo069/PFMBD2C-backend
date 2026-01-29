import os
from celery import Celery


"""
Configuration Celery pour utiliser Redis
À placer dans votre dossier Django principal (même niveau que settings.py)
"""
import os
from celery import Celery

# Définir le module de paramètres Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')  # ← IMPORTANT: 'backend.settings'

app = Celery('backend')

# Charger la configuration depuis les paramètres Django
app.config_from_object('django.conf:settings', namespace='CELERY')

# Découvrir automatiquement les tâches dans les applications Django
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')