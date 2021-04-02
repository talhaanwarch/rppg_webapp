from django.shortcuts import render
from .form import VideoForm
from .py_templates import predict_vitals
import numpy as np
from .models import VideoModel,OutModel
import pandas as pd
# Create your views here.


def home(request):
	
	if request.method=='POST':
		form=VideoForm(request.POST,request.FILES)
		#without request.Files, it would not work
		if form.is_valid():
			#
			#print(len(pathces))

			initial_obj = form.save(commit=False)
			initial_obj.save()
			vid=initial_obj.file.url
			hr=predict_vitals.predict_vitals(vid)
			insert = OutModel.objects.create(result=hr,upload_video=VideoModel.objects.last())
		

		return render(request,'home.html',{"prints_hr":hr,'form':form})
	else:
		form=VideoForm()
		return render(request,'home.html',{'form':form})


def display(request):
	data1 = list(VideoModel.objects.all().values())
	data1=pd.DataFrame.from_dict(data1)

	data2 = list(OutModel.objects.all().values())
	data2  =pd.DataFrame.from_dict(data2)
	data= pd.merge(data1,data2,on='id')
	
	data=data[['name','bmp','result']]
	print(data.head())
	return render(request,'display.html',{'result':data.to_html(index=...)})
