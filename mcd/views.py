import datetime

from django.contrib.auth.decorators import login_required
from django.views import generic
from mcd.models import MCD_Photo_Analysis, MCD_Object, MCD_Record
from django.shortcuts import render, \
    redirect, \
    get_object_or_404
from django.urls import reverse_lazy

# form to create , edit and delete a database object
from django.views.generic.edit import CreateView, UpdateView, DeleteView

# imports specifically for REST API:
# 1) if object does not exist, return 404 error:
from django.shortcuts import get_object_or_404
# 2) make normal views return API data:
from rest_framework.views import APIView
# 3) send back specific response status
from rest_framework.response import Response
# 4) check response status for 200 - response went fine
from rest_framework import status
# 5) import the serializer itself:
from .serializers import MCD_Photo_AnalysisSerializer

import threading
from .F_Use_Model import analyse_photo


# user management:
# (verify the user in the existent database + adds session ID ...
#  ... so users do not need to reauthenticate)
from django.contrib.auth import authenticate, login, logout
from django.views.generic import View
from .forms import UserForm, MCD_Photo_AnalysisFORM
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

# @login_required
class IndexView(generic.ListView):
    # this variable 'template_name' is an attribute of ...
    # ... django.views.generic
    # THE NAME OF 'template_name' MUST NOT BE CHANGED ...
    # ... as this is how Django finds the html template ...
    # ... for the index view here
    template_name = 'mcd/index.html'

    # change the name to refer to user photos
    # ... (in the template) as this:
    context_object_name = 'user_photos'

    def get_queryset(self):
        # if the user is not logged in, do not return ANY objects ...
        # ... (if the user is logged in, is also checked in the ...
        # ...  mcd/index.html template too, with:
        # ...  {% if user.is_authenticated %} )
        if not self.request.user.is_authenticated:
            return MCD_Photo_Analysis.objects.none()

        else:
            # get current user:
            user = self.request.user
            # filter out the objects that have been uploaded by the current user:
            objects_per_user = MCD_Photo_Analysis.objects.filter(uploaded_by_user_id=user)\
                                            .values_list('uploaded_by_user_id', flat=True).first()
            # return only the objects that match the current user logged in:
            return MCD_Photo_Analysis.objects.filter(uploaded_by_user_id=objects_per_user)


# @login_required
class ListAllImagesView(generic.ListView):
    # this variable 'template_name' is an attribute of ...
    # ... django.views.generic
    # THE NAME OF 'template_name' MUST NOT BE CHANGED ...
    # ... as this is how Django finds the html template ...
    # ... for the index view here
    template_name = 'mcd/list_photo_analysis.html'

    # change the name to refer to user photos
    # ... (in the template) as this:
    context_object_name = 'user_photos'

    def get_queryset(self):
        # if the user is not logged in, do not return ANY objects ...
        # ... (if the user is logged in, is also checked in the ...
        # ...  mcd/index.html template too, with:
        # ...  {% if user.is_authenticated %} )
        if not self.request.user.is_authenticated:
            return MCD_Photo_Analysis.objects.none()

        else:
            # get current user:
            user = self.request.user
            # filter out the objects that have been uploaded by the current user:
            objects_per_user = MCD_Photo_Analysis.objects.filter(uploaded_by_user_id=user)\
                                            .values_list('uploaded_by_user_id', flat=True).first()
            # return only the objects that match the current user logged in:
            return MCD_Photo_Analysis.objects.filter(uploaded_by_user_id=objects_per_user)


# @login_required
class ListAllObjectsView(generic.ListView):
    # this variable 'template_name' is an attribute of ...
    # ... django.views.generic
    # THE NAME OF 'template_name' MUST NOT BE CHANGED ...
    # ... as this is how Django finds the html template ...
    # ... for the index view here
    template_name = 'mcd/list_objects.html'

    # change the name to refer to user photos
    # ... (in the template) as this:
    context_object_name = 'user_objects'

    def get_queryset(self):
        # if the user is not logged in, do not return ANY objects ...
        # ... (if the user is logged in, is also checked in the ...
        # ...  mcd/index.html template too, with:
        # ...  {% if user.is_authenticated %} )
        if not self.request.user.is_authenticated:
            return MCD_Object.objects.none()

        else:
            # get current user:
            user = self.request.user
            # filter out the objects that have been uploaded by the current user:
            objects_per_user = MCD_Object.objects.filter(uploaded_by_user_id=user)\
                                            .values_list('uploaded_by_user_id', flat=True).first()
            # return only the objects that match the current user logged in:
            return MCD_Object.objects.filter(uploaded_by_user_id=objects_per_user)





class DetailsView(generic.DetailView):
    model = MCD_Photo_Analysis
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'photo_analysis'
    template_name = 'mcd/detailed_photo_analysis.html'


class ObjectDetailsView(generic.DetailView):
    model = MCD_Object
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'object'
    template_name = 'mcd/detailed_object.html'

    # current_object


    """
        def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']
        # filter out the objects that have been uploaded by the current user:
        images_in_object = MCD_Photo_Analysis.objects.filter(object_id=current_object) \
            .values_list('object_id', flat=True).first()
        # return only the objects that match the current user logged in:
        # return MCD_Photo_Analysis.objects.filter(object_id=images_in_object)

        data = super().get_context_data(**kwargs)
        data['images_of_object'] = MCD_Photo_Analysis.objects.filter(object_id=images_in_object)
        return data
    """

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']
        # filter out the objects that have been uploaded by the current user:
        records_in_object = MCD_Record.objects.filter(object_id=current_object) \
            .values_list('object_id', flat=True).first()
        # return only the objects that match the current user logged in:
        # return MCD_Photo_Analysis.objects.filter(object_id=images_in_object)

        data = super().get_context_data(**kwargs)
        data['records_in_object'] = MCD_Record.objects.filter(object_id=records_in_object)
        return data


        # data = super().get_context_data(**kwargs)
        # data['page_title'] = 'Aux Header info'
        # return data

    # def get_queryset(self):
    #     # get current object primary key (id):
    #     current_object = self.kwargs['pk']
    #     # filter out the objects that have been uploaded by the current user:
    #     images_in_object = MCD_Photo_Analysis.objects.filter(object_id=current_object) \
    #         .values_list('object_id', flat=True).first()
    #     # return only the objects that match the current user logged in:
    #     return MCD_Photo_Analysis.objects.filter(object_id=images_in_object)

    # def get_queryset(self):
    #     # get current user:
    #     current_object = super(MCD_Object, self)
    #     # filter out the images that are from current object:
    #     images_in_object = MCD_Object.objects.filter(object_id=current_object.pk) \
    #         .values_list('uploaded_by_user_id', flat=True).first()
    #     # return only the objects that match the current user logged in:
    #     return MCD_Object.objects.filter(object_id=images_in_object)


class RecordDetailsView(generic.DetailView):
    model = MCD_Record
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'record'
    template_name = 'mcd/detailed_record.html'

    # current_object

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_record = self.kwargs['pk']
        # filter out the objects that have been uploaded by the current user:
        images_in_record = MCD_Photo_Analysis.objects.filter(record_id=current_record) \
            .values_list('record_id', flat=True).first()
        # return only the objects that match the current user logged in:
        # return MCD_Photo_Analysis.objects.filter(object_id=images_in_object)

        data = super().get_context_data(**kwargs)
        data['images_of_record'] = MCD_Photo_Analysis.objects.filter(record_id=images_in_record)
        return data


class EnqueuePhotoAnalysis(threading.Thread):
    """
    after the user uploads their image, (or requests an update) ...
    ... enqueue the task and submit to the F_Use_Model.py to run it
    """
    def __init__(self, db_pk, title, user_id, input_url, output_url, completed):
        self.db_pk      = db_pk
        self.title      = title
        self.user_id    = user_id
        self.input_url  = input_url
        self.output_url = output_url
        self.completed  = completed

        threading.Thread.__init__(self)

    def run(self):
        output_photo_url = analyse_photo(self.input_url.url, self.title)
        print("photo analysed, posting to db index:", self.db_pk)

        # after the photo has been analysed ...
        t = MCD_Photo_Analysis.objects.get(id=self.db_pk)
        t.output_photo = output_photo_url  # change field
        t.analysis_complete = True # change field

        t.datetime_analysed = datetime.datetime.now()

        t.save() # this will update only

# =============================== Filler Views ================================== #
# --------------------------- MCD_Photo_Analysis -------------------------------- #
class PhotoAnalysisCreate(CreateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis
    template_name = 'mcd/mcd_photo_analysis_form.html'
    # what attributes do we allow the user to input?
    # fields = ['object_id', 'input_photo']


    form_class = MCD_Photo_AnalysisFORM


    # if the request it 'get', just display the blank form ...
    # ... with it being adjusted for the user to only be able to choose from objects ...
    # ... the y have themselves uploaded:
    def get(self, request):
        # by default - no data (context = None)
        form = self.form_class(None)

        objects_per_user = MCD_Object.objects.filter(uploaded_by_user_id=request.user) \
            .values_list('uploaded_by_user_id', flat=True).first()
        # return only the objects that match the current user logged in:
        # print(form.fields)
        form.fields['object_id'].queryset=MCD_Object.objects.filter(uploaded_by_user_id=objects_per_user)
        if not form.fields['object_id'].queryset:
            form.fields['object_id'].disabled   = True
            form.fields['record_id'].disabled   = True
            form.fields['title'].disabled       = True
            form.fields['input_photo'].disabled = True
        # print(" queryset >>> ", form.fields['object_id'].queryset.exists())

        return render(request, self.template_name, {'form' : form,
                                                    'object_exists' : form.fields['object_id'].queryset.exists() })

    def form_valid(self, form):
        # get the uploaded photo name:
        # uploaded_file = form.fields['input_photo'].instance
        uploaded_filename = str(form.instance.input_photo).rsplit('.', 1)[0]
        print(">>> uploaded file: ", uploaded_filename)

        current_datetime = datetime.datetime.now()

        print("in form_valid (self.request.user |", self.request.user, ")")

        # if no record was specified, create new record and assign to selected object:
        if form.instance.record_id == None:
            print("===> debugging worked - 'record_id' instance not set")
            mcd_record = MCD_Record(title=uploaded_filename+" Record",
                                    uploaded_by_user_id=self.request.user,
                                    object_id=form.instance.object_id)

            # compute how many records are uploaded in total in the object:
            # (NOTE - +1 added since the new record is not yet saved to database ...
            #  .. so the counter would not count the image just uploaded)
            mcd_record.num_images = mcd_record.mcd_photo_analysis_set.count()+1

            saved_mcd_record = mcd_record.save()
            print("------------> saved mcd record:", saved_mcd_record)
            print("------------> pre-saved mcd record:", mcd_record)

            form.instance.record_id = mcd_record
            # form.save()

        else:
            print("===> debugging worked - 'record_id' set to ", form.instance.record_id)
            # compute how many records are uploaded in total in the object:
            # (NOTE - +1 added since the new record is not yet saved to database ...
            #  .. so the counter would not count the image just uploaded)
            mcd_record = MCD_Record.objects.get(pk=form.instance.record_id.pk)
            mcd_record.num_images = mcd_record.mcd_photo_analysis_set.count()+1

            print("-------> image count, in record:", mcd_record.mcd_photo_analysis_set.count())

            mcd_record.save()

        # compute how many records are uploaded in total in the object:
        parent_object = MCD_Object.objects.get(pk=form.instance.object_id.pk)
        # print("> > > ", parent_object.mcd_photo_analysis_set.count())
        # for numbered_object in num_images:
        # print("> > >", parent_object.title, parent_object.number_of_images)
        parent_object.num_records = parent_object.mcd_record_set.count()
        parent_object.save()

        # assign  the user ID to be the uploader of the photo:
        form.instance.uploaded_by_user_id = self.request.user
        form.instance.datetime_uploaded = current_datetime

        # get the file name uploaded to upload as the title
        # (remove the extension - split by '.'):
        if form.instance.title and form.instance.title != "Untitled":
            pass
        else:
            form.instance.title = uploaded_filename

        # save the changes made to the database ...
        # ... and get the new assigned ID (primary key by task.id)
        task = form.save()

        # update the number of records of an object:
        from django.db.models import Count
        # num_images = MCD_Object.objects.annotate(number_of_images=Count('mcd_photo_analysis'))  # annotate the queryset
        num_images = MCD_Object.objects.annotate(number_of_images=Count('mcd_record'))  # annotate the queryset

        print(">>> num_images", num_images)

        parent_object = MCD_Object.objects.get(pk=form.instance.object_id.pk)
        print("> > > ", parent_object.mcd_photo_analysis_set.count())
        # for numbered_object in num_images:
        # print("> > >", parent_object.title, parent_object.number_of_images)
        parent_object.num_images = parent_object.mcd_photo_analysis_set.count()
        parent_object.save()

        EnqueuePhotoAnalysis(task.id,
                             form.instance.title,
                             form.instance.uploaded_by_user_id,
                             form.instance.input_photo,
                             form.instance.output_photo,
                             form.instance.analysis_complete).start()

        return super(PhotoAnalysisCreate, self).form_valid(form)

class PhotoAnalysisUpdate(UpdateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis

    # what attributes do we allow the user to input?
    fields = ['input_photo', 'output_photo']


class PhotoAnalysisDelete(DeleteView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis

    success_url = reverse_lazy('mcd:index')


# --------------------------- MCD_Object -------------------------------- #
class ObjectCreate(CreateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Object

    # what attributes do we allow the user to input?
    # fields = ['title', ]
    # exclude = ['uploaded_by_user_id', 'num_photos']
    fields = ['title',
              'latitude',
              'longitude',
              'placename',
              'county',
              'address',
              'postcode']

    def form_valid(self, form):
        # get the uploaded photo name:
        # uploaded_file = form.fields['input_photo'].instance
        # uploaded_filename = str(form.instance.input_photo).rsplit('.', 1)[0]
        # print(">>> uploaded file: ", uploaded_filename)

        print("in form_valid (self.request.user |", self.request.user, ")")
        form.instance.uploaded_by_user_id = self.request.user
        # get the file name uploaded to upload as the title
        # (remove the extension - split by '.'):
        # form.instance.title = uploaded_filename

        # save the changes made to the database ...
        # ... and get the new assigned ID (primary key by task.id)
        form.save()

        # EnqueuePhotoAnalysis(task.id,
        #                      form.instance.title,
        #                      form.instance.uploaded_by_user_id,
        #                      form.instance.input_photo,
        #                      form.instance.output_photo,
        #                      form.instance.analysis_complete).start()

        return super(ObjectCreate, self).form_valid(form)
# =============================== Filler Views ================================== #

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
        print("got to ELSE in views.py")
        return render(request, template_name='mcd/upload_to_analysis.html',
                      context={
                          'success_message': "Image "+image_to_analyse+" successfully uploaded for analysis"
                      })


class UserFormView(View):
    # which form do we want to use:
    form_class = UserForm
    # html file the form is included in:
    template_name = 'mcd/registration_form.html'

    # if the request it 'get', just display the blank form:
    def get(self, request):
        # by default - no data (context = None)
        form = self.form_class(None)
        return render(request, self.template_name, {'form' : form})

    # when user fills in the registration information, ...
    # ... need to add them to database:
    def post(self, request):
        form = self.form_class(request.POST)

        if form.is_valid():
            # creates an object from the form ...
            # ... it does not save to database yet (storing locally)
            user = form.save(commit=False)
            # format the data
            username = form.cleaned_data['username']
            email    = form.cleaned_data['email']
            password = form.cleaned_data['password']
            # passwords are hashed, set password dynamically
            user.set_password(password)
            user.save()

            # return User objects given correct credentials:
            user = authenticate(username=username, email=email, password=password)

            if user is not None:
                # if the account is not deactivated:
                if user.is_active:
                    login(request, user)
                    # now we can refer to the user we can refer to them as:
                    #    request.user ...

                    # when the user logs in, redirect to a meaningful URL
                    return redirect('mcd:index')
            else:
                print("redirecting user to form ...")
                return render(request, self.template_name, {'form': form})


def logout_view(request):
    print("logging out user: ", request.user.username)
    # use a POST request to log the user out:
    if request.method == 'POST':
        logout(request)
    return redirect('mcd:index')


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)

        if form.is_valid():
            # log in the user ...
            user = form.get_user()
            login(request, user)

            # if there is an active redirect (after the
            if 'next' in request.POST:
                return redirect(request.POST.get('next'))
            else:
                return redirect('mcd:index')
                # rendering alone (as shown in the line below:
                # return render(request, "mcd/index.html")
                # ... does not change the url, ...
                # ... instead it just renders the index page ...
                # ... with a different URL in place, ...
                # ... which is not desirable)

    else: # get
        form = AuthenticationForm

    # return redirect('mcd:index')
    return render(request, "mcd/login_form.html", {'form': form})

# API Functions / API Views:



#@ url:  mcd/<username>
class API_ListByUser_MCD_Photo_Analysis(APIView):
    """
        List all MCD_Photo_Analysis or creates new
    """
    # when user makes a GET request:
    def get(self, request):
        """
            :return: all the photos analysed by the given user:
        """
        mcd_photo_analysis = MCD_Photo_Analysis.objects.all()
        # serialize them (convert to JSON)
        serializer = MCD_Photo_AnalysisSerializer(mcd_photo_analysis, many=True)
        # return an HTTP response (with JSON in it)
        return Response(serializer.data)

    # when user makes a POST request:
    def post(self):
        pass