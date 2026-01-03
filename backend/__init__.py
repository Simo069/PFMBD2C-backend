# Cela garantira que l'app est toujours importée lorsque
# Django démarre afin que shared_task utilise cette app.
from .celery import app as celery_app

__all__ = ('celery_app',)