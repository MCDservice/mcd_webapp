import datetime
import pandas as pd
import numpy as np

from django.contrib.auth.decorators import login_required
from django.views import generic
from mcd.models import MCD_Photo_Analysis, MCD_Project, MCD_Record
from django.shortcuts import render, \
    redirect, \
    get_object_or_404
from django.urls import reverse_lazy
from google.cloud import storage

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
# from .F_Use_Model import analyse_photo
from .P7_Use_Model import analyse_photo


# user management:
# (verify the user in the existent database + adds session ID ...
#  ... so users do not need to reauthenticate)
from django.contrib.auth import authenticate, login, logout
from django.views.generic import View
from .forms import UserForm, MCD_Photo_AnalysisFORM
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.conf import settings as conf_settings

import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_secret_477185057888-brm030gcqnjoo7uijrijesp1ogi8hkah.apps.googleusercontent.com.json'

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
    template_name = 'mcd/list_projects.html'

    # change the name to refer to user photos
    # ... (in the template) as this:
    context_object_name = 'user_objects'

    def get_queryset(self):
        # if the user is not logged in, do not return ANY objects ...
        # ... (if the user is logged in, is also checked in the ...
        # ...  mcd/index.html template too, with:
        # ...  {% if user.is_authenticated %} )
        if not self.request.user.is_authenticated:
            return MCD_Project.objects.none()

        else:
            # get current user:
            user = self.request.user
            # filter out the objects that have been uploaded by the current user:
            objects_per_user = MCD_Project.objects.filter(uploaded_by_user_id=user)\
                                            .values_list('uploaded_by_user_id', flat=True).first()
            # return only the objects that match the current user logged in:
            return MCD_Project.objects.filter(uploaded_by_user_id=objects_per_user)





class DetailsView(generic.DetailView):
    model = MCD_Photo_Analysis
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'photo_analysis'
    template_name = 'mcd/detailed_photo_analysis.html'


class ObjectDetailsView(generic.DetailView):
    model = MCD_Project
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'project'
    template_name = 'mcd/detailed_project.html'

    # current_object


    """
        def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']
        # filter out the objects that have been uploaded by the current user:
        images_in_object = MCD_Photo_Analysis.objects.filter(project_id=current_object) \
            .values_list('project_id', flat=True).first()
        # return only the objects that match the current user logged in:
        # return MCD_Photo_Analysis.objects.filter(project_id=images_in_object)

        data = super().get_context_data(**kwargs)
        data['images_of_object'] = MCD_Photo_Analysis.objects.filter(project_id=images_in_object)
        return data
    """

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']
        # filter out the objects that have been uploaded by the current user:
        records_in_project = MCD_Record.objects.filter(project_id=current_object) \
            .values_list('project_id', flat=True).first()
        # return only the objects that match the current user logged in:
        # return MCD_Photo_Analysis.objects.filter(project_id=images_in_object)

        data = super().get_context_data(**kwargs)
        data['records_in_project'] = MCD_Record.objects.filter(project_id=records_in_project)
        return data


        # data = super().get_context_data(**kwargs)
        # data['page_title'] = 'Aux Header info'
        # return data

    # def get_queryset(self):
    #     # get current object primary key (id):
    #     current_object = self.kwargs['pk']
    #     # filter out the objects that have been uploaded by the current user:
    #     images_in_object = MCD_Photo_Analysis.objects.filter(project_id=current_object) \
    #         .values_list('project_id', flat=True).first()
    #     # return only the objects that match the current user logged in:
    #     return MCD_Photo_Analysis.objects.filter(project_id=images_in_object)

    # def get_queryset(self):
    #     # get current user:
    #     current_object = super(MCD_Project, self)
    #     # filter out the images that are from current object:
    #     images_in_object = MCD_Project.objects.filter(project_id=current_object.pk) \
    #         .values_list('uploaded_by_user_id', flat=True).first()
    #     # return only the objects that match the current user logged in:
    #     return MCD_Project.objects.filter(project_id=images_in_object)

class RecordComparison1View(generic.DetailView):
    model = MCD_Record
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'record'
    template_name = 'mcd/detailed_record_comparison1.html'

    # current_object

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_record = self.kwargs['pk']
        # filter out the objects that have been uploaded by the current user:
        image_ids_in_record = MCD_Photo_Analysis.objects.filter(record_id=current_record) \
            .values_list('record_id', flat=True).first()
        # return only the objects that match the current user logged in:
        # return MCD_Photo_Analysis.objects.filter(project_id=images_in_object)

        images_in_record = MCD_Photo_Analysis.objects.filter(record_id=image_ids_in_record)

        biggest_crack_length_list = []
        display_cm_list = []
        epsilon = 0.0001
        for i, image in enumerate(images_in_record):
            biggest_crack_length_list.append(round(image.scale * image.crack_length,2))
            if abs(float(image.scale) - 1.0) < epsilon:
                display_cm_list.append(0)
            else:
                display_cm_list.append(1)

        # if the image is chosen (it appears in the URL as such:
        #  mcd/record/<recordID>/image-<imageID>/
        #  Example: mcd/record/19/image-85/
        display_image_1 = MCD_Photo_Analysis.objects.get(pk=self.kwargs['image_pk1'])
        display_image_2 = MCD_Photo_Analysis.objects.get(pk=self.kwargs['image_pk2'])



        def list_literal_eval(x):
            import ast
            return ast.literal_eval(x)

        crack_labels = []
        crack_locations = []
        crack_lengths = []
        sorted_ids = []
        # get biggest crack length:
        try:
            crack_labels_csv = pd.read_csv(display_image_1.crack_labels_csv)

            crack_labels = crack_labels_csv["Label"].astype(int)
            crack_locations = crack_labels_csv["Loc (x,y)"].apply(list_literal_eval)
            crack_lengths = crack_labels_csv["Length (pxls)"].astype(int)

            # (negation to sort in descending order!)
            sorted_ids = np.argsort(-crack_lengths)
        except:
            pass

        comparison = False
        comparison_method = False
        try:
            comparison_method = self.kwargs['comparison']
            comparison = True
        except:
            pass

        # t.crack_length = float(sizes["Length (pxls)"].max())

        data = super().get_context_data(**kwargs)
        data['images_of_record']  = images_in_record.reverse()
        data['display_image']     = display_image_1
        data['display_image_2']   = display_image_2
        data['display_cm']        = display_cm_list
        data['longest_crack']     = biggest_crack_length_list
        data['crack_labels']      = crack_labels
        data['crack_locations']   = crack_locations
        data['crack_lengths']     = crack_lengths
        data['sorted_ids']        = sorted_ids
        # data for record image comparison:
        data['comparison']        = comparison
        data['comparison_method'] = comparison_method
        return data

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
        image_ids_in_record = MCD_Photo_Analysis.objects.filter(record_id=current_record) \
            .values_list('record_id', flat=True).first()
        # return only the objects that match the current user logged in:
        # return MCD_Photo_Analysis.objects.filter(project_id=images_in_object)

        images_in_record = MCD_Photo_Analysis.objects.filter(record_id=image_ids_in_record)

        biggest_crack_length_list = []
        display_cm_list = []
        epsilon = 0.0001
        for i, image in enumerate(images_in_record):
            biggest_crack_length_list.append(round(image.scale * image.crack_length,2))
            if abs(float(image.scale) - 1.0) < epsilon:
                display_cm_list.append(0)
            else:
                display_cm_list.append(1)

        # if the image is chosen (it appears in the URL as such:
        #  mcd/record/<recordID>/image-<imageID>/
        #  Example: mcd/record/19/image-85/
        try:
            display_image = MCD_Photo_Analysis.objects.get(pk=self.kwargs['image_pk'])
        except:
            display_image = images_in_record.latest('datetime_uploaded')


        def list_literal_eval(x):
            import ast
            return ast.literal_eval(x)

        crack_labels = []
        crack_locations = []
        crack_lengths = []
        sorted_ids = []
        # get biggest crack length:
        try:
            crack_labels_csv = pd.read_csv(display_image.crack_labels_csv)

            crack_labels = crack_labels_csv["Label"].astype(int)
            crack_locations = crack_labels_csv["Loc (x,y)"].apply(list_literal_eval)
            crack_lengths = crack_labels_csv["Length (pxls)"].astype(int)

            # (negation to sort in descending order!)
            sorted_ids = np.argsort(-crack_lengths)
        except:
            pass

        comparison = False
        comparison_method = False
        try:
            comparison_method = self.kwargs['comparison']
            comparison = True
        except:
            pass

        # t.crack_length = float(sizes["Length (pxls)"].max())

        data = super().get_context_data(**kwargs)
        data['images_of_record']  = images_in_record.reverse()
        data['display_image']     = display_image
        data['display_cm']        = display_cm_list
        data['longest_crack']     = biggest_crack_length_list
        data['crack_labels']      = crack_labels
        data['crack_locations']   = crack_locations
        data['crack_lengths']     = crack_lengths
        data['sorted_ids']        = sorted_ids
        # data for record image comparison:
        data['comparison']        = comparison
        data['comparison_method'] = comparison_method
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
        # change field to say that it is currently processing:
        t = MCD_Photo_Analysis.objects.get(id=self.db_pk)
        t.analysis_complete = False
        t.save()  # this will only that analysis is not complete

        overlay_photo_url, \
        output_photo_url,\
        crack_len_url,\
        crack_labels_url = analyse_photo(self.input_url.url, self.title)

        print("crack_len_url", crack_len_url)
        print("photo analysed, posting to db index:", self.db_pk)

        # get biggest crack length:
        # sizes = pd.read_csv(conf_settings.MEDIA_URL.split('/')[1]+"\\"+crack_len_url)

        # "https://storage.cloud.google.com/mcd_file_storage/media/142_epoch_45_f1_m_dil_0.796/a_6_32/Sizes.csv"

        # sizes = pd.read_csv(conf_settings.MEDIA_URL.split('/')[1]+"\\"+crack_len_url)
        # csv_cloud_url = conf_settings.MEDIA_URL + crack_len_url.replace('\\', '/')
        csv_cloud_url = 'gs://'+conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET+"/media/"+crack_len_url.replace('\\', '/')
        print(">>> reading csv from URL: ", csv_cloud_url )
        # sizes = pd.read_csv(csv_cloud_url, sep=",", lineterminator='\r')
        sizes = pd.read_csv(csv_cloud_url, sep=",", lineterminator='\n')
        print(">  >  > dataframe: ", list(sizes))
        t.crack_length = float(sizes["Length (pxls)"].max())

        # after the photo has been analysed ...
        t.output_photo       = output_photo_url  # change field
        t.overlay_photo      = overlay_photo_url  # change field
        t.crack_labels_csv = crack_len_url  # change field
        t.crack_labels_photo = crack_labels_url  # change field
        t.analysis_complete  = True # change field

        t.datetime_analysed = datetime.datetime.now()

        t.save() # this will update only

# =============================== Filler Views ================================== #
# --------------------------- MCD_Photo_Analysis -------------------------------- #
class PhotoAnalysisCreate(CreateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis
    template_name = 'mcd/mcd_photo_analysis_form.html'
    # what attributes do we allow the user to input?
    # fields = ['project_id', 'input_photo']


    form_class = MCD_Photo_AnalysisFORM


    # if the request it 'get', just display the blank form ...
    # ... with it being adjusted for the user to only be able to choose from objects ...
    # ... the y have themselves uploaded:
    def get(self, request):
        # by default - no data (context = None)
        form = self.form_class(None)

        objects_per_user = MCD_Project.objects.filter(uploaded_by_user_id=request.user) \
            .values_list('uploaded_by_user_id', flat=True).first()
        # return only the objects that match the current user logged in:
        # print(form.fields)
        form.fields['project_id'].queryset=MCD_Project.objects.filter(uploaded_by_user_id=objects_per_user)
        if not form.fields['project_id'].queryset:
            form.fields['project_id'].disabled   = True
            form.fields['record_id'].disabled   = True
            form.fields['title'].disabled       = True
            form.fields['input_photo'].disabled = True
        # print(" queryset >>> ", form.fields['project_id'].queryset.exists())

        return render(request, self.template_name, {'form' : form,
                                                    'object_exists' : form.fields['project_id'].queryset.exists()})

    def form_valid(self, form):
        # get the uploaded photo name:
        # uploaded_file = form.fields['input_photo'].instance
        uploaded_filename = str(form.instance.input_photo).rsplit('.', 1)[0]
        print(">>> uploaded file: ", uploaded_filename)

        current_datetime = datetime.datetime.now()

        print("in form_valid (self.request.user |", self.request.user, ")")



        ### --------- GOOGLE CLOUD STORAGE COMPATIBILITY ----------- ###
        # Create a Cloud Storage client.
        gcs = storage.Client()

        # Get the bucket that the file will be uploaded to.
        bucket = gcs.get_bucket(conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET)

        # Create a new blob and upload the file's content.
        blob = bucket.blob('media/'+uploaded_filename)

        uploaded_file = form.instance.input_photo
        blob.upload_from_string(
            uploaded_file.read(),
            # content_type=uploaded_file.content_type
        )

        # The public URL can be used to directly access the uploaded file via HTTP.
        print("blob path:", blob.path)
        print("blob purl:", blob.public_url)
        # print("blob help:", blob.path_helper(conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET, uploaded_file))
        form.instance.input_photo = uploaded_filename #blob.path
        ### --------- GOOGLE CLOUD STORAGE COMPATIBILITY ----------- ###



        # if no record was specified, create new record and assign to selected object:
        if form.instance.record_id == None:
            print("===> debugging worked - 'record_id' instance not set")
            mcd_record = MCD_Record(title=uploaded_filename+" Record",
                                    uploaded_by_user_id=self.request.user,
                                    project_id=form.instance.project_id)

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
        parent_object = MCD_Project.objects.get(pk=form.instance.project_id.pk)
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
        # num_images = MCD_Project.objects.annotate(number_of_images=Count('mcd_photo_analysis'))  # annotate the queryset
        num_images = MCD_Project.objects.annotate(number_of_images=Count('mcd_record'))  # annotate the queryset

        print(">>> num_images", num_images)

        parent_object = MCD_Project.objects.get(pk=form.instance.project_id.pk)
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
    template_name = 'mcd/mcd_photo_analysis_update_form.html'

    # what attributes do we allow the user to input?
    fields = ['input_photo', 'project_id', 'record_id', 'scale']

    # def form_valid(self, form):
    #     print("form filled in", form.instance.reanalyse)
    #     return super(PhotoAnalysisUpdate, self).form_valid(form)

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']

        data = super().get_context_data(**kwargs)
        data['current_id'] = int(current_object)
        return data

    def post(self, request, pk):
        print("recv POST", pk)
        # form = self.form_class(None)

        current_image = MCD_Photo_Analysis.objects.get(pk=pk)

        print("current form:", current_image.title)
        print(current_image.uploaded_by_user_id)
        print(current_image.input_photo)
        print(current_image.output_photo)
        print(current_image.analysis_complete)

        EnqueuePhotoAnalysis(pk,
                             current_image.title,
                             current_image.uploaded_by_user_id,
                             current_image.input_photo,
                             current_image.output_photo,
                             current_image.analysis_complete).start()

        # return redirect('mcd:index')
        return redirect('mcd:detailed_record_image_pk', current_image.record_id.pk, pk)


class PhotoAnalysisDelete(DeleteView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis

    success_url = reverse_lazy('mcd:index')


# --------------------------- MCD_Project -------------------------------- #
class ObjectCreate(CreateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Project

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


def add_scale(request, pk):

    display_image = MCD_Photo_Analysis.objects.get(pk=pk)

    try:
        (cx,cy)=list(request.GET.keys())[0].split(',')
        # cx=request.GET.get('param1')
        # cy=request.GET.get('param2')
    # print(">>>> received GET request:", list(request.GET.keys())[0].split(','))
    except:
        cx = 0
        cy = 0
    # map was clicked at cx,cy coordinates
    x=int(cx)
    y=int(cy)

    return render(request, "mcd/add_scale.html", {'display_image' : display_image,
                                                  'clicked_x' : x,
                                                  'clicked_y' : y})

def add_scale_2(request, pk, cx, cy):

    display_image = MCD_Photo_Analysis.objects.get(pk=pk)

    try:
        (cx2,cy2)=list(request.GET.keys())[0].split(',')
        # cx=request.GET.get('param1')
        # cy=request.GET.get('param2')
    # print(">>>> received GET request:", list(request.GET.keys())[0].split(','))
    except:
        cx2 = 0
        cy2 = 0

    # map was clicked at cx,cy coordinates
    x=int(cx)
    y=int(cy)

    x2 = int(cx2)
    y2 = int(cy2)



    return render(request, "mcd/add_scale_2.html", {'display_image' : display_image,
                                                    'clicked_x' : x,
                                                    'clicked_y' : y,
                                                    'clicked_x2': x2,
                                                    'clicked_y2': y2,
                                                    })

def add_scale_3(request, pk, cx, cy, point):

    display_image = MCD_Photo_Analysis.objects.get(pk=pk)

    try:
        (cx2,cy2)=list(request.GET.keys())[0].split(',')
        # cx=request.GET.get('param1')
        # cy=request.GET.get('param2')
    # print(">>>> received GET request:", list(request.GET.keys())[0].split(','))
    except:
        cx2 = 0
        cy2 = 0

    # map was clicked at cx,cy coordinates
    x=int(cx)
    y=int(cy)

    x2 = int(cx2)
    y2 = int(cy2)

    # when user fills in the scale information, ...
    # ... need to add that to database:
    if request.POST:
        print(">>> GOT POST form: ", request.POST.get('input_real_length'),
                                     request.POST.get('input_px_length'))

        real_length = float(request.POST.get('input_real_length'))
        px_length   = float(request.POST.get('input_px_length'))

        print("ratio: ", real_length/px_length)

        display_image.scale = real_length/px_length
        display_image.save()



        # compute the real scale




    return render(request, "mcd/add_scale_3.html", {'display_image' : display_image,
                                                    'clicked_x' : x,
                                                    'clicked_y' : y,
                                                    'clicked_x2': x2,
                                                    'clicked_y2': y2,
                                                    'point'     : point
                                                    })

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

