from django.db import models

# Create your models here.
class VideoModel(models.Model):
	name = models.CharField(max_length=30)
	bmp=models.FloatField()
	file=models.FileField(upload_to='video')


	def __str__(self):
		return self.name

class OutModel(models.Model):
	result=models.FloatField(blank=True, null=True,)
	upload_video=models.ForeignKey(VideoModel, blank=True, null=True, on_delete = models.CASCADE)