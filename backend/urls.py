from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # API routes
    path('api/auth/', include('account.urls')),
    path('api/documents/', include('document.urls')),
    path('api/chat/', include('chat.urls')),
<<<<<<< HEAD
    path('api/dashboard/', include('dashboard.urls')),

=======
>>>>>>> c1e83bb169a8ddcf9777e866126cb8e577764cb2
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    
    