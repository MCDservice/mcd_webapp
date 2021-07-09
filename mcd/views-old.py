from mcd.models import MCD_Photo_Analysis
from django.shortcuts import render, get_object_or_404

# Create your views here:

# default /mcd/ view
def index(request):
    # connect to database, get all the user's photos:
    user_photos = MCD_Photo_Analysis.objects.all()
    # information needed by the template
    context = {
        'user_photos' : user_photos
    }
    return render(request, template_name='mcd/index.html', context=context)

def photo_analysis_detailed(request, photo_analysis_id):
    # query the database, if there is a photo-pair with that ID:
    photo_analysis = get_object_or_404(MCD_Photo_Analysis, pk=photo_analysis_id)

    return render(request, template_name='mcd/detailed_photo_analysis.html',
                  context={'photo_analysis_id' : photo_analysis_id,
                           'photo_analysis'    : photo_analysis })


def upload_to_analysis(request):
    return render(request, template_name='mcd/upload_to_analysis.html')


def submit_to_analysis(request):
    try:
        print("got name: ", request.POST['input_img'])
        image_to_analyse = request.POST['input_img']
    except (KeyError, MCD_Photo_Analysis.DoesNotExist):
        return render(request, template_name='mcd/upload_to_analysis.html',
                      context={
                          'error_message' : "error submitting"
                      })

    else:
        # get current logged in user:
        # ... TODO
        # insert to database of the current user:
        # ... TODO
        pass
        return render(request, template_name='mcd/upload_to_analysis.html',
                      context={
                          'success_message': "Image "+image_to_analyse+" successfully uploaded for analysis"
                      })