import os
from celery import Celery

# Définir le module de paramètres Django par défaut pour Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

app = Celery('backend')

# Utiliser une chaîne ici signifie que le worker ne doit pas sérialiser
# l'objet de configuration à l'enfant.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Charger automatiquement les tâches depuis tous les apps Django enregistrés.
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')