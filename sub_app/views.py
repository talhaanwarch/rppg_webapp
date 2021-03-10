from django.shortcuts import render
from .form import VideoForm
from .py_templates import code
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
			patches=code.patch_extract(initial_obj.file.url)
			hr=code.heart_rate(patches)
			print('form saved')
		return render(request,'home.html',{"prints":hr,'form':form})
	else:
		form=VideoForm()
		return render(request,'home.html',{'form':form})