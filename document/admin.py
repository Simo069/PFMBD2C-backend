from django.contrib import admin
from .models import PDFFile , Chunk
# Register your models here.


admin.site.register(PDFFile)
admin.site.register(Chunk)
