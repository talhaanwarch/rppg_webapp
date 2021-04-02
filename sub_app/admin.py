from django.contrib import admin

# Register your models here.
from .models import VideoModel,OutModel
 
admin.site.register(VideoModel)
admin.site.register(OutModel)