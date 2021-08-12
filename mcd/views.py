import datetime
import json

import math
import pandas as pd
import numpy as np

# import database models (defined in file models.py)
from mcd.models import MCD_Photo_Analysis, MCD_Project, MCD_Record
from google.cloud import storage

# form to create , edit and delete a database object
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.http import HttpResponseForbidden, JsonResponse

from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib import messages


# modules for user account registration:
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
# from .forms import SignupForm
from django.views import generic
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from django.contrib.auth.models import User
from django.core.mail import EmailMessage
from django.urls import reverse_lazy, reverse
from .tokens import account_activation_token
# end of modules for user account registration

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
# use the P7_Use_Model.py to analyse the photos:
# the core functionality is in the function 'analyse_photo()'
from .P7_Use_Model import analyse_photo


# user management:
# (verify the user in the existent database + adds session ID ...
#  ... so users do not need to reauthenticate)
from django.contrib.auth import authenticate, login, logout
from django.views.generic import View
from .forms import UserForm, UserUpdateForm, MCD_Photo_AnalysisFORM, MCD_RecordFORM, MCD_ProjectFORM
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.conf import settings as conf_settings

import os
from .secretizer import hide_username, hash256sha

# if you get an error:
# DefaultCredentialsError at /mcd/upload/25/118
# "Could not automatically determine credentials.
#  Please set GOOGLE_APPLICATION_CREDENTIALS ...
#  ... or explicitly create credentials and re-run the application."
# then, go to this link:
# https://console.cloud.google.com/iam-admin/serviceaccounts/details/105516471793314199998/keys?authuser=4&project=cool-keel-320414&supportedpurview=project
# (Or if the link does not work in future years:
# 1) Google Cloud Platform Home Page
# 2) Find "IAM & Admin" in the menu
# 3) Service Accounts
# 4) Add Key
# 5) Create New Key (OR DOWNLOAD EXISTING KEY!)
# 6) download the .json file and put it in same folder where manage.py is
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_secret_477185057888-brm030gcqnjoo7uijrijesp1ogi8hkah.apps.googleusercontent.com.json'

RUN_LOCALLY = False
if RUN_LOCALLY:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(),'cool-keel-320414-6ceb64e6ce52.json')

def clear_tmp_dir():
    import os, shutil
    folder = conf_settings.MEDIA_ROOT
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

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
    
    def get_context_data(self, **kwargs):
        data = super().get_context_data(**kwargs)
        data['num_projects'] = 0
        data['num_records'] = 0
        data['num_images'] = 0
        try:
            data['num_projects'] = self.request.user.mcd_project_set.count()
            data['num_records']  = self.request.user.mcd_record_set.count()
            data['num_images']   = self.request.user.mcd_photo_analysis_set.count()
        except:
            pass
        return data
    
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
    
    # if you want to turn on paging ...
    # ... (for example, if there are 100 images, ...
    # ...  paginate_by = 10 will split 100 images to 10 pages)
    # ... uncomment the following:
    # paginate_by = 10

    # GO BACK
    def paginate_queryset(self, queryset, request, view=None):
        if 'no_page' in request.query_params:
            return None

        return super().paginate_queryset(queryset, request, view)

    def get_context_data(self, *, object_list=None, **kwargs):
        data = super().get_context_data(**kwargs)
        url = reverse('mcd:get_filtered_images', args=[], kwargs={})

        current_url = reverse('mcd:image-list', args=[], kwargs={})
        newest_first_url = current_url + '?ordering=pk'
        oldest_first_url = current_url + '?ordering=-pk'

        data['search_view'] = url
        data['current_url'] = current_url
        data['newest_first_url'] = newest_first_url
        data['oldest_first_url'] = oldest_first_url

        data['num_projects'] = 0
        data['num_records'] = 0
        data['num_images'] = 0
        try:
            data['num_projects'] = self.request.user.mcd_project_set.count()
            data['num_records']  = self.request.user.mcd_record_set.count()
            data['num_images']   = self.request.user.mcd_photo_analysis_set.count()
        except:
            pass

        return data

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

            ordering = self.request.GET.get('ordering')
            if ordering is None:
                ordering = '-pk'

            # return only the objects that match the current user logged in:
            return MCD_Photo_Analysis.objects.filter(uploaded_by_user_id=objects_per_user).order_by(ordering)


# @login_required
class ListAllObjectsView(LoginRequiredMixin, generic.ListView):
    # this variable 'template_name' is an attribute of ...
    # ... django.views.generic
    # THE NAME OF 'template_name' MUST NOT BE CHANGED ...
    # ... as this is how Django finds the html template ...
    # ... for the index view here
    template_name = 'mcd/list_projects.html'

    # change the name to refer to user photos
    # ... (in the template) as this:
    context_object_name = 'user_objects'
    login_url = '/login/'
    redirect_field_name = 'redirect_to'
    ordering = ['?']


    def get_context_data(self, *, object_list=None, **kwargs):
        data = super().get_context_data(**kwargs)
        url = reverse('mcd:get_filtered_projects', args=[], kwargs={})
        current_url = reverse('mcd:object-list', args=[], kwargs={})

        newest_first_url = current_url+'?ordering=pk'
        oldest_first_url = current_url+'?ordering=-pk'

        data['search_view'] = url
        data['current_url'] = current_url
        data['newest_first_url'] = newest_first_url
        data['oldest_first_url'] = oldest_first_url
        return data

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

            ordering = self.request.GET.get('ordering')
            print("GOT ordering:", ordering)
            if ordering is None:
                ordering = '-pk'
            return MCD_Project.objects.filter(uploaded_by_user_id=objects_per_user).order_by(ordering)



class DetailsView(generic.DetailView):
    model = MCD_Photo_Analysis
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'photo_analysis'
    template_name = 'mcd/detailed_photo_analysis.html'


class ObjectDetailsView(UserPassesTestMixin, generic.DetailView):
    model = MCD_Project
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'project'
    template_name = 'mcd/detailed_project.html'

    def test_func(self):
        # not allow the user who has not uploaded it to access the data
        requested_project = MCD_Project.objects.get(pk=self.kwargs['pk'])

        if not requested_project.uploaded_by_user_id.pk == self.request.user.pk:
            # return HttpResponseForbidden("You can't view this Bar.")
            return False
        else:
            return True

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

        # GO BACK
        url = reverse('mcd:get_filtered_records', args=[], kwargs={})
        current_url = reverse('mcd:detailed_object', args=[current_object], kwargs={})
        newest_first_url = current_url+'?ordering=pk'
        oldest_first_url = current_url+'?ordering=-pk'

        data = super().get_context_data(**kwargs)
        data['search_view'] = url
        data['current_url'] = current_url
        data['newest_first_url'] = newest_first_url
        data['oldest_first_url'] = oldest_first_url

        ordering = self.request.GET.get('ordering')
        print("GOT ordering:", ordering)
        if ordering is None:
            ordering = '-pk'

        data['records_in_project'] = MCD_Record.objects.filter(project_id=records_in_project).order_by(ordering)
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


def read_csv_from_cloud(csv_filename, newline_char):
    csv_cloud_url = 'gs://' + conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET + \
                    '/' + conf_settings.MEDIA_DIR_NAME + csv_filename.replace(
        '\\', '/')

    try:
        processed_csv = pd.read_csv(csv_cloud_url, sep=",", lineterminator=newline_char)
        return processed_csv
    except:
        return pd.DataFrame()


class RecordComparison1View(generic.DetailView):
    model = MCD_Record
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'record'
    template_name = 'mcd/detailed_record_comparison1.html'

    # get template name depending on which comparison option is chosen:
    def get_template_names(self):
        display_image_1 = MCD_Photo_Analysis.objects.get(pk=self.kwargs['image_pk1'])
        display_image_2 = MCD_Photo_Analysis.objects.get(pk=self.kwargs['image_pk2'])

        # to avoid unexpected behaviours, ...
        # ... check if the analysis of both images have been completed ...
        # ... (since we will be comparing analysis outputs)

        if display_image_1.analysis_complete and display_image_2.analysis_complete:
            if self.kwargs["comparison"] == "comparison1":
                return ['mcd/detailed_record_comparison1.html']
            elif self.kwargs["comparison"] == "comparison2":
                return ['mcd/detailed_record_comparison2.html']
            else:
                return ['mcd/detailed_record_comparison1.html']
        else:
            messages.warning(self.request,
                             'Warning: Since the analysis of one of the selected images has not been completed,'
                             ' we cannot compare the images. Please try comparing images where both images'
                             ' have the analysis completed.',
                             extra_tags='danger')
            return ['mcd/detailed_record.html'] #, display_image_1.record_id.pk, display_image_1.pk]


    # current_object
    # def __init__(self, **kwargs):
    #     # super().__init__(**kwargs)
    #     if kwargs["comparison"] == "comparison1":
    #         template_name = 'mcd/detailed_record_comparison1.html'
    #     elif kwargs["comparison"] == "comparison2":
    #         template_name = 'mcd/detailed_record_comparison2.html'

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_record = self.kwargs['pk']
        # filter out the images that belong to the current reco:
        image_ids_in_record = MCD_Photo_Analysis.objects.filter(record_id=current_record) \
            .values_list('record_id', flat=True).first()
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
            print(">>> Reading detailed crack information from", display_image_1.crack_labels_csv)
            relative_csv_path = get_cloud_relative_path_from_folder("media", display_image_1.crack_labels_csv.url)
            crack_labels, crack_locations, crack_lengths = get_crack_info_from_cloud_csv(relative_csv_path)

            # crack_labels_csv = pd.read_csv(display_image_1.crack_labels_csv)
            # crack_labels_csv = read_csv_from_cloud(display_image_1.crack_labels_csv)
            # crack_labels = crack_labels_csv["Label"].astype(int)
            # crack_locations = crack_labels_csv["Loc (x,y)"].apply(list_literal_eval)
            # crack_lengths = crack_labels_csv["Length (pxls)"].astype(int)

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


def get_local_relative_path_from_folder(folder, url):
    return url.split(os.path.join(folder, ''), 1)[1]

def get_cloud_relative_path_from_folder(folder, url):
    return url.split(folder+'/', 1)[1]


def get_crack_info_from_cloud_csv(relative_csv_path):
    crack_labels = []
    crack_locations = []
    crack_lengths = []

    def list_literal_eval(x):
        import ast
        return ast.literal_eval(x)
    # [interoperability]
    # Google Cloud uses Unix-like operating system use \n for new lines, ...
    # ... whereas for testing locally, \r might be used:
    # (in such cases, the heading index becomes:
    # 'Length (pxls)\r' or 'Length (pxls)\n', instead of 'Length (pxls)' ...
    # ... which gives a KeyError

    crack_labels_csv = read_csv_from_cloud(relative_csv_path, '\n')
    if crack_labels_csv.empty:
        return None, None, None

    try:
        # processed_csv = pd.read_csv(csv_cloud_url, sep=",", lineterminator='\n')
        crack_labels_csv = read_csv_from_cloud(relative_csv_path, '\n')
        crack_labels = crack_labels_csv["Label"].astype(int)
        crack_locations = crack_labels_csv["Loc (x,y)"].apply(list_literal_eval)
        crack_lengths = crack_labels_csv["Length (pxls)"].astype(int)
    except KeyError:
        crack_labels_csv = read_csv_from_cloud(relative_csv_path, '\r')
        # for '\r\n', pandas does not support multi-character line-terminators ...
        # ... therefore, a workaround is needed - splitting on \r and then removing leftover \n
        # https://stackoverflow.com/questions/53844875/how-to-deal-with-multi-value-lineterminators-in-pandas
        crack_labels_csv.iloc[:, 0] = crack_labels_csv.iloc[:, 0].str.lstrip()
        crack_labels_csv.dropna(inplace=True)  # removes empty lines

        crack_labels = crack_labels_csv["Label"].astype(int)
        crack_locations = crack_labels_csv["Loc (x,y)"].apply(list_literal_eval)
        crack_lengths = crack_labels_csv["Length (pxls)"].astype(int)
        # processed_csv = pd.read_csv(csv_cloud_url, sep=",", lineterminator='\r')

    return crack_labels, crack_locations, crack_lengths


class RecordDetailsView(UserPassesTestMixin, generic.DetailView):
    model = MCD_Record
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'record'
    template_name = 'mcd/detailed_record.html'

    def test_func(self):
        # not allow the user who has not uploaded it to access the data
        mcd_record = MCD_Record.objects.get(pk=self.kwargs['pk'])

        if mcd_record.num_images == 0:
            return False

        if not mcd_record.uploaded_by_user_id.pk == self.request.user.pk:
            # return HttpResponseForbidden("You can't view this Bar.")
            return False
        else:
            return True

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_record = self.kwargs['pk']
        mcd_record = MCD_Record.objects.get(pk=current_record )

        # filter out the records that are associated with the current record:
        image_ids_in_record = MCD_Photo_Analysis.objects.filter(record_id=current_record) \
            .values_list('record_id', flat=True).first()
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
            display_image_2 = images_in_record.earliest('datetime_uploaded')
        except:
            display_image = images_in_record.latest('datetime_uploaded')
            display_image_2 = images_in_record.earliest('datetime_uploaded')

        sorted_ids = []
        crack_labels = []
        crack_locations = []
        crack_lengths = []
        # get biggest crack length:
        # try:
        # crack_labels_csv = pd.read_csv(display_image.crack_labels_csv)
        try:
            relative_csv_path = get_cloud_relative_path_from_folder("media", display_image.crack_labels_csv.url)
            crack_labels, crack_locations, crack_lengths = get_crack_info_from_cloud_csv(relative_csv_path)
            # (negation to sort in descending order!)
            if crack_lengths is not None:
                sorted_ids = np.argsort(-crack_lengths)
        except ValueError:
            # The 'crack_labels_csv' attribute has no file associated with it.
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


# [IMPORTANT]
class EnqueuePhotoAnalysis(threading.Thread):
    """
    after the user uploads their image, (or requests an update) ...
    ... enqueue the task and submit to the F_Use_Model.py to run it
    """
    def __init__(self, db_pk, title, user_id, input_url, output_url, completed,
                 user, project_id, record_id, analysis_id, status_json):
        self.db_pk       = db_pk
        self.title       = title
        self.user_id     = user_id
        self.input_url   = input_url
        self.output_url  = output_url
        self.completed   = completed


        self.user        = user
        self.project_id  = project_id
        self.record_id   = record_id
        self.analysis_id = analysis_id
        self.status_json = status_json

        threading.Thread.__init__(self)

    def run(self):
        # change field to say that it is currently processing:
        t = MCD_Photo_Analysis.objects.get(id=self.db_pk)
        t.analysis_complete = False
        t.save()  # this will only that analysis is not complete


        # CURRENTLY, analyse_photo() in file P7_Use_Model returns FOUR (4)
        # ... images as output!
        # IF you want to add more images as output, ...
        # ... please edit the 'analyse_photo()' function to RETURN more photo URLs!
        overlay_photo_url, \
        output_photo_url,\
        crack_len_url,\
        crack_labels_url = analyse_photo(self.input_url.url, self.title,
                                         self.user, self.project_id,
                                         self.record_id, self.analysis_id,
                                         self.status_json.url)
        # also if there are more models that you have, you can add them like this:
        # block_photo_url = analyse_blocks(self.input_url.url, self.title,
        #                                  self.user, self.project_id,
        #                                  self.record_id, self.analysis_id,
        #                                  self.status_json.url)

        print("[INFO] Photo analysed, posting to Database Index:", self.db_pk)

        # get biggest crack length:
        csv_cloud_url = 'gs://'+conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET + \
                        '/'+ conf_settings.MEDIA_DIR_NAME + crack_len_url.replace('\\', '/')
        print("[INFO] Reading Crack Sizes.csv from URL: ", csv_cloud_url )

        # [interoperability]
        # Google Cloud uses Unix-like operating system use \n for new lines, ...
        # ... whereas for testing locally, \r might be used:
        # (in such cases, the heading index becomes:
        # 'Length (pxls)\r' or 'Length (pxls)\n', instead of 'Length (pxls)' ...
        # ... which gives a KeyError 3
        try:
            sizes = pd.read_csv(csv_cloud_url, sep=",", lineterminator='\n')
            t.crack_length = float(sizes["Length (pxls)"].max())
        except KeyError:
            sizes = pd.read_csv(csv_cloud_url, sep=",", lineterminator='\r')
            t.crack_length = float(sizes["Length (pxls)"].max())
            # for '\r\n', pandas does not support multi-character line-terminators ...
            # ... therefore, a workaround is needed - splitting on \r and then removing leftover \n
            # https://stackoverflow.com/questions/53844875/how-to-deal-with-multi-value-lineterminators-in-pandas
            sizes.iloc[:, 0] = sizes.iloc[:, 0].str.lstrip()
            sizes.dropna(inplace=True)  # removes empty lines

        # if the algorithm did not find any cracks ...
        # ... then it will be nan and cause this exception:
        # django.db.utils.OperationalError
        # ... therefore, we set crack length to 0.0
        if math.isnan(t.crack_length):
            t.crack_length = 0.0

        print("[INFO] Successfully Read Dataframe, with Headings: ", list(sizes))
        print("[INFO] Data Preview: ", sizes)


        # after the photo has been analysed ...
        # ... put the links URLs of the output images on the cloud ...
        # ... in our User Database!
        t.output_photo       = output_photo_url   # change field
        t.overlay_photo      = overlay_photo_url  # change field
        t.crack_labels_csv = crack_len_url        # change field
        t.crack_labels_photo = crack_labels_url   # change field
        t.analysis_complete  = True               # change field

        # if you add MORE outputs, add something like:
        # t.another_analysis_output = another_analysis_output_URL # change field

        t.datetime_analysed = datetime.datetime.now()

        try:
            t.save() # this will save the new output images to database!
            
        except Exception as e:  # work on python 3.x
            print("[INFO] Posting to database failed, removing files")
            print("[DEBUG] Info for post failure:")
            print(e)
            # if for any reason post to database fails ...
            # ... delete the generated files from database
            delete_file_from_cloud_media(output_photo_url)
            delete_file_from_cloud_media(overlay_photo_url)
            delete_file_from_cloud_media(crack_len_url)
            delete_file_from_cloud_media(crack_labels_url)
            delete_file_from_cloud_media(t.analysis_status_json)

            # [EXPAND]
            # if you added more files as output ...
            # ... please add more cleanup, like this:
            # delete_file_from_cloud_media(some_another_file)

            print("[INFO] File deletion complete")

def bytes_to_img(bytes):
    import numpy as np
    import cv2

    readFlag = cv2.IMREAD_COLOR
    # print(bytes)
    image = np.asarray(bytearray(bytes), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image


def resize_img(img_in_bytes_to_resize):
    import cv2

    # img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED)
    img = bytes_to_img(img_in_bytes_to_resize)

    print('[INFO] Resizing Image, Original Dimensions : ', img.shape)

    # scale_percent = 60  # percent of original size
    scale = (math.trunc((img.shape[1] * img.shape[0]) / 1000000)) / 1.65
    if scale < 1:
        scale = 1

    width = int(img.shape[1] / scale)
    height = int(img.shape[0] / scale)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('[INFO] Resized Dimensions : ', resized.shape)

    img_str = cv2.imencode('.jpg', resized)[1].tostring()
    return img_str

# =============================== Filler Views ================================== #
# --------------------------- MCD_Photo_Analysis -------------------------------- #
def upload_file_to_cloud(file_to_upload, filename, resize=False, set_content_type=False):

    if resize:
        file_to_upload = resize_img(file_to_upload.read())
        # im = Image.fromarray(resizelist[val])

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(conf_settings.MEDIA_DIR_NAME + filename)

    if set_content_type != 'application/json':
        # file_to_upload = form.instance.input_photo
        if resize:
            blob.upload_from_string(
                file_to_upload,
                # content_type=uploaded_file.content_type
            )
        else:
            blob.upload_from_string(
                file_to_upload.read(),
                # content_type=uploaded_file.content_type
            )
    else:
        blob.upload_from_string(
            file_to_upload,
            content_type=set_content_type
        )


    # The public URL can be used to directly access the uploaded file via HTTP.
    print("blob path:", blob.path)
    print("blob purl:", blob.public_url)


def delete_file_from_cloud_media(filename):
    import google.api_core

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET)

    # Create a new blob and upload the file's content.
    try:
        blob = bucket.blob(conf_settings.MEDIA_DIR_NAME + filename)
        blob.delete()
    except google.api_core.exceptions.NotFound:
        # file does not exist - already deleted, or by other means
        print("File", conf_settings.MEDIA_DIR_NAME + filename, "already not present")
        pass


def make_cloud_filename(username, project_id, record_id, analysis_id):
    return hash256sha(username+'/'+str(project_id)+'/'+str(record_id)+'/'+str(analysis_id))

def make_cloud_filename_json(username, project_id, record_id, analysis_id, csv_title):
    return hash256sha(username+'/'+str(project_id)+'/'+str(record_id)+'/'+str(analysis_id)+'/'+csv_title)


class PhotoAnalysisCreate(UserPassesTestMixin, CreateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis
    template_name = 'mcd/mcd_photo_analysis_form.html'
    # what attributes do we allow the user to input?
    # fields = ['project_id', 'input_photo']

    form_class = MCD_Photo_AnalysisFORM

    def test_func(self):


        if 'project_id' not in self.kwargs and 'record_id' not in self.kwargs:
            return True

        print("running tests on: ", self.kwargs['project_id'], self.kwargs['record_id'])

        import mcd
        try:
            # not allow the user who has not uploaded it to access the data
            requested_project = MCD_Project.objects.get(pk=self.kwargs['project_id'])
        except mcd.models.MCD_Project.DoesNotExist:
            return False

        try:
            requested_record  = MCD_Record.objects.get(pk=self.kwargs['record_id'])
        except mcd.models.MCD_Record.DoesNotExist:
            return False

        if not requested_project.uploaded_by_user_id.pk == self.request.user.pk:
            # return HttpResponseForbidden("Unauthorised Access")
            return False

        if not requested_record.uploaded_by_user_id.pk == self.request.user.pk:
            # return HttpResponseForbidden("Unauthorised Access")
            return False

        if requested_record.project_id.pk != requested_project.pk:
            return False

        else:
            return True

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        # pk = self.kwargs.get("pk")
        kwargs.update({'current_user': self.request.user})
        return kwargs

    # def get_context_data(self, **kwargs):
    #
    #     print("getting context data:")
    #     print("kwargs:", self.kwargs)
    #
    #     # get current object primary key (id):
    #     chosen_project_id = self.kwargs['project_id']
    #     chosen_record_id  = self.kwargs['record_id']
    #
    #     data = super().get_context_data(**kwargs)
    #
    #     # data['current_id'] = int(current_object)
    #     data['project_id'] = int(chosen_project_id)
    #     data['record_id']  = int(chosen_record_id)
    #     return data

    # if the request it 'get', just display the blank form ...
    # ... with it being adjusted for the user to only be able to choose from objects ...
    # ... the y have themselves uploaded:
    def get(self, request, project_id = None, record_id=None):
        # proj_id = self.kwargs.get("project_id")
        # by default - no data (context = None)
        # form = self.form_class(None)

        js_flag = True
        if project_id is not None and record_id is not None:
            js_flag = False


        print("GOT: project_id", project_id, "record_id", record_id)

        form = self.form_class(current_user=self.request.user,
                               project_id=project_id,
                               record_id=record_id)
                               # initial={'project_id': u'12'})

        # in the upload form, let the user choose only from projects/records they themselves have created:
        # (filter out records and projects by user id)
        objects_per_user = MCD_Project.objects.filter(uploaded_by_user_id=request.user) \
            .values_list('uploaded_by_user_id', flat=True).first()

        records_per_user = MCD_Record.objects.filter(uploaded_by_user_id=request.user) \
            .values_list('uploaded_by_user_id', flat=True).first()
        # limit the form fields:
        # form.fields['project_id'].queryset=MCD_Project.objects.filter(uploaded_by_user_id=objects_per_user)
        # form.fields['record_id'].queryset=MCD_Record.objects.filter(uploaded_by_user_id=records_per_user)

        if not form.fields['project_id'].queryset:
            form.fields['project_id'].disabled   = True
            form.fields['record_id'].disabled   = True
            form.fields['title'].disabled       = True
            form.fields['input_photo'].disabled = True
        # print(" queryset >>> ", form.fields['project_id'].queryset.exists())

        return render(request, self.template_name, {'form' : form,
                                                    'object_exists' : form.fields['project_id'].queryset.exists(),
                                                    'project_id': project_id,
                                                    'record_id' : record_id,
                                                    'run_js' : js_flag})

    def form_valid(self, form):
        # get the uploaded photo name:
        # uploaded_file = form.fields['input_photo'].instance

        # get the file name uploaded to upload as the title
        # (remove the extension - split by '.'):
        if form.instance.title and form.instance.title != "Untitled":
            uploaded_filename = form.instance.title
        else:
            uploaded_filename = str(form.instance.input_photo).rsplit('.', 1)[0]

        current_datetime = datetime.datetime.now()

        ### --------- GOOGLE CLOUD STORAGE COMPATIBILITY ----------- ###

        # if no record was specified, create new record and assign to selected object:
        if form.instance.record_id == None or form.instance.record_id == "new":
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

        # save the current changes to get the ID (needed for filename generation)
        form.save()
        print("form pk1:", form.instance.pk)

        filename_on_cloud = make_cloud_filename(hide_username(self.request.user),
                                                form.instance.project_id.pk,
                                                mcd_record.pk, form.instance.pk)
        print("input photo:")
        print("{")
        print(form.instance.input_photo)
        # print(form.instance.input_photo.read())
        print("}")
        upload_file_to_cloud(form.instance.input_photo, filename_on_cloud, resize=True)
        form.instance.input_photo = filename_on_cloud

        # generate the status csv:
        status_json_filename_on_cloud = make_cloud_filename_json(hide_username(self.request.user),
                                                                 form.instance.project_id.pk,
                                                                 mcd_record.pk, form.instance.pk,
                                                                 "status_csv.json")

        status_json = {'percentage_complete': 0.0,
                       'status': 'initialized',
                       'error' : 'none'}

        json_object = json.dumps(status_json, indent=4)
        upload_file_to_cloud(json_object, status_json_filename_on_cloud, set_content_type='application/json')

        form.instance.analysis_status_json = status_json_filename_on_cloud
        print("[INFO] Uploaded status JSON to Cloud with filename: ", form.instance.analysis_status_json)



        # clear temporarily saved files in App Engine /tmp/
        clear_tmp_dir()

        # save the changes made to the database ...
        # ... and get the new assigned ID (primary key by task.id)
        task = form.save()

        # update the number of records of an object:
        from django.db.models import Count
         # annotate the queryset
        num_images = MCD_Project.objects.annotate(number_of_images=Count('mcd_record'))

        parent_object = MCD_Project.objects.get(pk=form.instance.project_id.pk)

        # for numbered_object in num_images:
        parent_object.num_images = parent_object.mcd_photo_analysis_set.count()
        parent_object.save()

        print("enqueue IN  photo url: ", form.instance.input_photo)
        print("enqueue OUT photo url: ", form.instance.output_photo)

        # [IMPORTANT] - Sending Photo To ANALYSE!
        # THIS EnqueuePhotoAnalysis calls P7_Use_Model analyse_photo() function ...
        # ... if you add more output images, please edit EnqueuePhotoAnalysis() ...
        # ... and analyse_photo(), to deal with additional photos and save them ...
        # ... to the database
        EnqueuePhotoAnalysis(task.id,
                             form.instance.title,
                             form.instance.uploaded_by_user_id,
                             form.instance.input_photo,
                             form.instance.output_photo,
                             form.instance.analysis_complete,
                             hide_username(self.request.user), form.instance.project_id.pk,
                             mcd_record.pk, form.instance.pk, form.instance.analysis_status_json
                             ).start()

        return super(PhotoAnalysisCreate, self).form_valid(form)


class ProjectUpdate(generic.UpdateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Project

    template_name_suffix = '_update_form'
    previous_name = ""

    # what attributes do we allow the user to edit?
    # ... the fields are defined in 'forms.py'
    form_class = MCD_ProjectFORM

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']

        data = super().get_context_data(**kwargs)
        data['current_id'] = int(current_object)
        return data

    def form_invalid(self, form):
        print("[INFO] GOT form invalid")
        return HttpResponse("form is invalid.. this is just an HttpResponse object")

    def form_valid(self, form):

        pk = self.kwargs.get("pk")
        new_title = form.cleaned_data.get("title")

        current_project = MCD_Project.objects.get(pk=pk)
        self.previous_name = current_project.title

        msg_change = 'Successfully changed details of project "'+self.previous_name+'"'
        if self.previous_name != new_title :
            msg_change += ' (new title "'+new_title+'")'

        form.save()

        messages.success(self.request,
                         msg_change,
                         extra_tags='success')
        return redirect('mcd:object-list')


class ProjectDelete(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Project

    def test_func(self):
        # not allow the user who has not uploaded it to access the data
        requested_project = MCD_Project.objects.get(pk=self.kwargs['pk'])
        if not requested_project.uploaded_by_user_id.pk == self.request.user.pk:
            # return HttpResponseForbidden("Unauthorised Access")
            return False
        else:
            return True

    template_name = 'mcd/mcd_project_confirm_delete.html'
    def get(self, request, pk):
        title = MCD_Project.objects.get(pk=pk).title
        return render(request, self.template_name, {'pk': pk,
                                                    'title': title})

    def post(self, request, *args, **kwargs):
        current_project = self.get_object()  # Add this to load the object

        print("deleting project ", current_project.title)
        print("records connected with this record:")

        records_in_project = MCD_Record.objects.filter(project_id=current_project) \
            .values_list('project_id', flat=True).first()
        associated_records = MCD_Record.objects.filter(project_id=records_in_project)

        print("Listing:", associated_records )

        for mcd_record in associated_records:
            record_name = mcd_record.title
            delete_record(mcd_record)
            print("[INFO] Finished deleting record:", record_name)

        # delete the project itself:
        self.delete(request, *args, **kwargs)

        # success message and redirect back to project view
        messages.success(self.request,
                         'Successfully deleted project '+current_project.title,
                         extra_tags='success')
        return redirect('mcd:object-list')

    # messages.success(self.request,
    #                  'Successfully renamed record from <b>' + self.previous_name + ' to <b>' + current_record.title + '</b>.',
    #                  extra_tags='success')
    # return redirect('mcd:detailed_object', current_project.pk)
    success_url = reverse_lazy('mcd:index')

# [NOT IMPLEMENTED] - Did not get time to implement DeleteUser securely ...
# ... so therefore there is not a possibility at this stage to do it

# def delete_user(request, username):
#     try:
#         u = User.objects.get(username = username)
#         u.delete()
#         messages.success(request, "The user is deleted")
#
#     except User.DoesNotExist:
#         messages.error(request, "User doesnot exist")
#         return redirect('mcd:index')
#
#     except Exception as e:
#         return redirect('mcd:index')
#
#     return redirect('mcd:index')

class UserProfile(View):
    model = User

    # html file the form is included in:
    # template_name = 'mcd/user_profile.html'
    template_name = 'mcd/user_profile.html'

    # if the request it 'get', just display the blank form:
    def get(self, request):
        return render(request, self.template_name, {'username' : request.user,
                                                    'email' : request.user.email})

class UserUpdate(View):

    model = User

    # which form do we want to use:
    form_class = UserUpdateForm
    # html file the form is included in:
    # template_name = 'mcd/user_profile.html'
    template_name = 'mcd/user_update_form.html'

    # if the request it 'get', just display the blank form:
    def get(self, request):
        # by default - no data (context = None)
        form = self.form_class(None)

        # form = UserUpdateForm(data=request.POST, instance=request.user)

        return render(request, self.template_name, {'form' : form,
                                                    'username' : request.user,
                                                    'email' : request.user.email})

    # when user fills in the registration information, ...
    # ... need to add them to database:
    def post(self, request):
        # form = self.form_class(request.POST)
        form = UserUpdateForm(data=request.POST, instance=request.user)

        if form.is_valid():
            # creates an object from the form ...
            # ... it does not save to database yet (storing locally)
            # user = form.save(commit=False)
            # format the data
            email    = form.cleaned_data['email']

            # set the user as inactive until they confirm their email:
            # request.user.is_active = False
            # user.save()
            user = User.objects.filter(pk=request.user.pk).update(email=email)
            user = User.objects.filter(pk=request.user.pk).update(is_active=False)

            user = User.objects.get(pk=request.user.pk)

            current_site = get_current_site(request)
            mail_subject = 'Activate your MCD account.'
            message = render_to_string('mcd/activate_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uidb64': force_text(urlsafe_base64_encode(force_bytes(user.pk))),
                'token': account_activation_token.make_token(user),
            })
            to_email = form.cleaned_data.get('email')

            print("[INFO] Sending email (to_email)", str(to_email)[:6], "...")

            email = EmailMessage(
                mail_subject, message, to=[to_email]
            )
            email.send()

            print("[INFO] Email Successfully sent to ", str(to_email)[:6], "...")

            logout(request)

            messages.warning(request,
                             'Warning: Since you changed the email, the account is no longer active. '
                             'Please check your email inbox and confirm your email before logging in',
                             extra_tags='warning')
            return redirect('mcd:login')

        else:
            messages.error(request,
                           form.errors.as_text(),
                           extra_tags='danger')
            return redirect('mcd:profile-update')



class RecordUpdate(generic.UpdateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Record
    # form = 'mcd/mcd_record_update_form.html'

    # what attributes do we allow the user to edit?
    # fields = ['title', 'project_id']
    template_name_suffix = '_update_form'
    previous_name = ""

    form_class = MCD_RecordFORM

    def get(self, request, *args, **kwargs):
        # self.user = self.request.user
        # by default - no data (context = None)

        # get current project-id and set it as default:
        current_mcd_record = MCD_Record.objects.get(pk=self.kwargs["pk"])

        form = self.form_class(initial={'project_id': current_mcd_record.project_id.pk,
                                        'title': current_mcd_record.title})

        # in the upload form, let the user choose only from projects/records they themselves have created:
        # (filter out records and projects by user id)
        objects_per_user = MCD_Project.objects.filter(uploaded_by_user_id=request.user) \
            .values_list('uploaded_by_user_id', flat=True).first()

        projects_per_user = MCD_Record.objects.filter(uploaded_by_user_id=request.user) \
            .values_list('uploaded_by_user_id', flat=True).first()
        # limit the form fields:
        form.fields['project_id'].queryset=MCD_Project.objects.filter(uploaded_by_user_id=projects_per_user)

        return render(request, 'mcd/mcd_record_update_form.html', {'form': form})


    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']

        data = super().get_context_data(**kwargs)
        data['current_id'] = int(current_object)
        return data


    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        # if form_valid(self, form):
        if form.is_valid():

            pk = self.kwargs.get("pk")

            current_record = MCD_Record.objects.get(pk=pk)
            self.previous_name = current_record.title
            self.previous_proj = current_record.project_id.pk

            current_record.title = form.cleaned_data.get("title")#form.fields["title"]
            current_record.project_id = form.cleaned_data.get("project_id") #form.fields["project_id"]

            current_record.save()

            current_record = MCD_Record.objects.get(pk=pk)
            current_project = MCD_Project.objects.get(pk=current_record.project_id.pk)

            # update the counters if project/record moved:
            move_record(current_record,
                        MCD_Project.objects.get(pk=self.previous_proj),
                        form.cleaned_data.get("project_id"))

            messages.success(self.request,
                             'Successfully changed project "'+current_project.title+'" details',
                             extra_tags='success')
            return redirect('mcd:detailed_object', current_project.pk)
        else:
            return HttpResponse("Form is invalid")


def move_record(record_to_move, from_project, to_project):

    image_ids_in_record = MCD_Photo_Analysis.objects.filter(record_id=record_to_move.pk) \
        .values_list('record_id', flat=True).first()
    associated_img_analysis = MCD_Photo_Analysis.objects.filter(record_id=image_ids_in_record)

    for img_analysis in associated_img_analysis:
        print("[INFO] Moving record from Project #:", from_project)
        print("[...] to Project #: ", to_project)
        img_analysis.project_id = to_project
        img_analysis.save()
        print("[INFO] Successfully moved.")

    # get current project:

    # compute how many records are uploaded in total in the project:
    from_project.num_records = from_project.mcd_record_set.count()
    from_project.num_images = from_project.mcd_photo_analysis_set.count()
    from_project.save()

    to_project.num_records = to_project.mcd_record_set.count()
    to_project.num_images = to_project.mcd_photo_analysis_set.count()
    to_project.save()

# [IMPORTANT] - if you add more photos - please delete them afterwards!
# (else the files will no longer be associated with the user ...
# ... but they will still take up the space since they did not get deleted
def delete_record(record_to_delete):

    print("[INFO] Deleting record ", record_to_delete.title)
    print("[INFO] Listing MCD_Photo_Analysis connected with this record:")

    image_ids_in_record = MCD_Photo_Analysis.objects.filter(record_id=record_to_delete.pk) \
        .values_list('record_id', flat=True).first()
    associated_img_analysis = MCD_Photo_Analysis.objects.filter(record_id=image_ids_in_record)

    for img_analysis in associated_img_analysis:
        print("[INFO] deleting 'input_photo':", img_analysis.input_photo.name)
        delete_file_from_cloud_media(img_analysis.input_photo.name)

        print("[INFO] deleting 'overlay_photo':", img_analysis.overlay_photo.name)
        delete_file_from_cloud_media(img_analysis.overlay_photo.name)

        print("[INFO] deleting 'output_photo':", img_analysis.output_photo.name)
        delete_file_from_cloud_media(img_analysis.output_photo.name)

        print("[INFO] deleting 'crack_labels_photo':", img_analysis.crack_labels_photo.name)
        delete_file_from_cloud_media(img_analysis.crack_labels_photo.name)

        print("[INFO] deleting 'crack_labels_csv':", img_analysis.crack_labels_csv.name)
        delete_file_from_cloud_media(img_analysis.crack_labels_csv.name)

        print("[INFO] deleting 'analysis_status_json':", img_analysis.analysis_status_json.name)
        delete_file_from_cloud_media(img_analysis.analysis_status_json.name)

        print("[...] Deleted All")

        # [FUTURE] [ADD]
        # print("[INFO] deleting 'another_file_field':", image_to_delete.another_file_field.name)
        # delete_file_from_cloud_media(image_to_delete.another_file_field.name)

    # get current project:
    current_project = MCD_Project.objects.get(pk=record_to_delete.project_id.pk)

    # delete the record itself:
    record_to_delete.delete()

    # compute how many records are uploaded in total in the object:
    # (NOTE - +1 added since the new record is not yet saved to database ...
    #  .. so the counter would not count the image just uploaded)
    current_project.num_records = current_project.mcd_record_set.count()
    current_project.num_images = current_project.mcd_photo_analysis_set.count()
    current_project.save()

# class RecordDelete(DeleteView):
class RecordDelete(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Record

    def test_func(self):
        # not allow the user who has not uploaded it to access the data
        requested_record = MCD_Record.objects.get(pk=self.kwargs['pk'])
        if not requested_record.uploaded_by_user_id.pk == self.request.user.pk:
            # return HttpResponseForbidden("Unauthorised Access")
            return False
        else:
            return True

    template_name = 'mcd/mcd_record_confirm_delete.html'

    def get(self, request, pk):
        title = MCD_Record.objects.get(pk=pk).title
        return render(request, self.template_name, {'pk': pk,
                                                    'title': title})

    def post(self, request, *args, **kwargs):
        record_to_delete = self.get_object()
        record_name = record_to_delete.title
        record_parent_project = record_to_delete.project_id

        delete_record(record_to_delete)

        # success message and redirect back to project view
        messages.success(self.request,
                         'Successfully deleted record "'+record_name+'"',
                         extra_tags='success')
        return redirect('mcd:detailed_object', record_parent_project.pk)

    success_url = reverse_lazy('mcd:index')


class PhotoAnalysisUpdate(LoginRequiredMixin, UpdateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis
    template_name = 'mcd/mcd_photo_analysis_update_form.html'

    context_object_name = 'photo_analysis'

    # what attributes do we allow the user to input?
    # fields = ['input_photo', 'project_id', 'record_id', 'scale']
    form_class = MCD_Photo_AnalysisFORM

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        # pk = self.kwargs.get("pk")
        kwargs.update({'current_user': self.request.user})
        return kwargs


    def form_valid(self, form):

        pk = self.kwargs.get("pk")
        new_title = form.cleaned_data.get("title")

        current_img = MCD_Photo_Analysis.objects.get(pk=pk)
        self.previous_name = current_img.title

        previous_record  = current_img.record_id
        previous_project = current_img.project_id

        new_project = MCD_Project.objects.get(pk=form.instance.project_id.pk)
        try:
            new_record  = MCD_Record.objects.get(pk=form.instance.record_id.pk)
        except:
            # the new record might not exist yet, ...
            # ...we deal with this case in:
            # ... if form.instance.record_id == None or form.instance.record_id == "new":
            pass

        # in case the user chooses to create a new record automatically:
        # get the file name uploaded to upload as the title
        # (remove the extension - split by '.'):
        if form.instance.title and form.instance.title != "Untitled":
            uploaded_filename = form.instance.title
        else:
            uploaded_filename = str(form.instance.input_photo).rsplit('.', 1)[0]

        # if no record was specified, create new record and assign to selected object:
        if form.instance.record_id == None or form.instance.record_id == "new":
            print("===> debugging worked - 'record_id' instance not set")
            new_record = MCD_Record(title=uploaded_filename+" Record",
                                    uploaded_by_user_id=self.request.user,
                                    project_id=form.instance.project_id)

            # compute how many records are uploaded in total in the object:
            # (NOTE - +1 added since the new record is not yet saved to database ...
            #  .. so the counter would not count the image just uploaded)
            new_record.num_images = new_record.mcd_photo_analysis_set.count()+1

            new_record.save()

            form.instance.record_id = new_record

        print("[INFO] current img pre-saving:")
        print("[INFO] current_img.title:", current_img.title)
        print("[INFO] current_img.project_id:", current_img.project_id)
        print("[INFO] current_img.record_id:", current_img.record_id)
        current_img = form.save()
        print("[INFO] current img post-saving:")
        print("[INFO] current_img.title:", current_img.title)
        print("[INFO] current_img.project_id:", current_img.project_id)
        print("[INFO] current_img.record_id:", current_img.record_id)


        # recompute the statistics of ...
        # ... projects (previous project)
        # ... and records (previous record):

        # remove -1 image count from previous project/record
        previous_record.num_images  = previous_record.mcd_photo_analysis_set.count() #-1
        previous_record.save()


        # if the old record now has 0 photos, delete the record:
        if previous_record.num_images == 0:
            messages.warning(self.request,
                             'NOTE: Since Image "' + new_title + '" was the last image in Record "' +
                             previous_record.title + '", it was deleted.',
                             extra_tags='warning')
            # delete the previous record, since it is now empty:
            # previous_record.delete()
            delete_record(previous_record)
            # update counts of precious project, that previous record was a part of:
            previous_project.num_records = new_project.mcd_record_set.count()
            previous_project.num_images = new_project.mcd_photo_analysis_set.count()
            previous_project.save()


        print("\n[INFO] updating old project:", previous_project.title, previous_project.num_images, previous_project.num_records)
        previous_project.num_images = previous_project.mcd_photo_analysis_set.count() #-1
        previous_project.num_records = previous_project.mcd_record_set.count() #-1
        previous_project.save()
        print("[INFO] updated old project:", previous_project.title, previous_project.num_images, previous_project.num_records)

        # add +1 image count from previous project/record
        print("\n[INFO] updating new record:", new_record.title, new_record.num_images)
        new_record.num_images = new_record.mcd_photo_analysis_set.count() #+1
        new_record.save()
        print("[INFO] updated new record:", new_record.title, new_record.num_images)

        print("\n[INFO] updating new project:", new_project.title, new_project.num_images, new_project.num_records)
        new_project.num_images = new_project.mcd_photo_analysis_set.count() #+1
        new_project.num_records = new_project.mcd_record_set.count() #+1
        new_project.save()
        print("[INFO] updated new project:", new_project.title, new_project.num_images, previous_project.num_records)


        msg_change = 'Successfully changed details of Image "' + self.previous_name + '"'
        #  if the title was changed, then add this to the change message:
        if self.previous_name != new_title:
            msg_change += ' (new title "' + new_title + '")'

        messages.success(self.request,
                         msg_change,
                         extra_tags='success')
        return redirect('mcd:detailed_record_image_pk', new_record.pk, pk)

    def form_invalid(self, form):
        pk = self.kwargs.get("pk")
        current_img = MCD_Photo_Analysis.objects.get(pk=pk)

        messages.error(self.request,
                       form.errors.as_text(),
                       extra_tags='danger')
        return redirect('mcd:detailed_record_image_pk', current_img.record_id.pk, pk)

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']

        data = super().get_context_data(**kwargs)
        data['current_id'] = int(current_object)
        data['photo_analysis'] = MCD_Photo_Analysis.objects.get(pk=int(current_object))
        return data


class PhotoAnalysisReanalyse(View):

    model = MCD_Photo_Analysis
    template_name = 'mcd/mcd_photo_analysis_confirm_reanalysis.html'

    def get(self, request, pk):
        title = MCD_Photo_Analysis.objects.get(pk=pk).title
        return render(request, self.template_name, {'pk' : pk,
                                                    'title' : title})

    def post(self, request, pk):
        current_image = MCD_Photo_Analysis.objects.get(pk=pk)

        # generate the status csv:
        try:
            status_json_filename_on_cloud = current_image.analysis_status_json.url.split("media/")[1]
        # if the json file was not created before, create it now:
        except ValueError:
            # generate the status csv:
            status_json_filename_on_cloud = make_cloud_filename_json(hide_username(request.user),
                                                                     current_image.project_id.pk,
                                                                     current_image.record_id.pk, current_image.pk,
                                                                     "status_csv.json")
            # add the status json URL to MCD_Photo_Analysis object:
            current_image.analysis_status_json = status_json_filename_on_cloud
            # and finally, save the added URL to database:
            current_image.save()

        status_json = {'percentage_complete': 0.0,
                       'status': 'initialized',
                       'error' : 'none'}

        json_object = json.dumps(status_json, indent=4)
        upload_file_to_cloud(json_object, status_json_filename_on_cloud, set_content_type='application/json')

        # current_image.analysis_status_json = status_json_filename_on_cloud
        print("[INFO] Re-initialised status JSON to Cloud with filename: ", current_image.analysis_status_json)

        # [IMPORTANT] - Sending Photo To ANALYSE!
        # THIS EnqueuePhotoAnalysis calls P7_Use_Model analyse_photo() function ...
        # ... if you add more output images, please edit EnqueuePhotoAnalysis() ...
        # ... and analyse_photo(), to deal with additional photos and save them ...
        # ... to the database
        EnqueuePhotoAnalysis(pk,
                             current_image.title,
                             current_image.uploaded_by_user_id,
                             current_image.input_photo,
                             current_image.output_photo,
                             current_image.analysis_complete,
                             hide_username(self.request.user), current_image.project_id.pk,
                             current_image.record_id.pk, current_image.pk, current_image.analysis_status_json
                             ).start()

        # return redirect('mcd:index')
        return redirect('mcd:detailed_record_image_pk', current_image.record_id.pk, pk)

class PhotoAnalysisDelete(DeleteView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis
    template_name = 'mcd/mcd_photo_analysis_confirm_delete.html'

    def test_func(self):
        # not allow the user who has not uploaded it to access the data
        requested_image = MCD_Photo_Analysis.objects.get(pk=self.kwargs['pk'])
        if not requested_image.uploaded_by_user_id.pk == self.request.user.pk:
            return False
        else:
            return True

    def get(self, request, pk):
        title = MCD_Photo_Analysis.objects.get(pk=pk).title
        return render(request, self.template_name, {'pk': pk,
                                                    'title': title})

    def post(self, request, *args, **kwargs):
        image_to_delete = self.get_object()
        image_name = image_to_delete.title
        image_parent_record = image_to_delete.record_id
        image_parent_project = image_to_delete.project_id

        print("[INFO] deleting 'input_photo':", image_to_delete.input_photo.name)
        delete_file_from_cloud_media(image_to_delete.input_photo.name)

        print("[INFO] deleting 'overlay_photo':", image_to_delete.overlay_photo.name)
        delete_file_from_cloud_media(image_to_delete.overlay_photo.name)

        print("[INFO] deleting 'output_photo':", image_to_delete.output_photo.name)
        delete_file_from_cloud_media(image_to_delete.output_photo.name)

        print("[INFO] deleting 'crack_labels_photo':", image_to_delete.crack_labels_photo.name)
        delete_file_from_cloud_media(image_to_delete.crack_labels_photo.name)

        print("[INFO] deleting 'crack_labels_csv':", image_to_delete.crack_labels_csv.name)
        delete_file_from_cloud_media(image_to_delete.crack_labels_csv.name)

        print("[INFO] deleting 'analysis_status_json':", image_to_delete.analysis_status_json.name)
        delete_file_from_cloud_media(image_to_delete.analysis_status_json.name)

        # [FUTURE] [ADD]
        # print("[INFO] deleting 'another_file_field':", image_to_delete.another_file_field.name)
        # delete_file_from_cloud_media(image_to_delete.another_file_field.name)

        # delete the record itself:
        image_to_delete.delete()

        # compute how many records are uploaded in total in the object:
        # (NOTE - +1 added since the new record is not yet saved to database ...
        #  .. so the counter would not count the image just uploaded)
        image_parent_project.num_records = image_parent_project.mcd_record_set.count()
        image_parent_project.num_images = image_parent_project.mcd_photo_analysis_set.count()
        image_parent_project.save()

        image_parent_record.num_images = image_parent_record.mcd_photo_analysis_set.count()
        image_parent_record.save()

        if image_parent_record.num_images == 0:
            # image_parent_record = MCD_Record.objects.get(pk=image_parent_record.pk)
            # image_parent_record.delete()
            delete_record(image_parent_record)
            image_parent_project.num_records = image_parent_project.mcd_record_set.count()
            image_parent_project.num_images = image_parent_project.mcd_photo_analysis_set.count()
            image_parent_project.save()

            messages.success(self.request,
                             'NOTE: Since Image "' + image_name + '" was the last image in Record "'+
                             image_parent_record.title + '", it was deleted.',
                             extra_tags='warning')
            image_parent_project.num_records = image_parent_project.mcd_record_set.count()
            return redirect('mcd:detailed_object', image_parent_project.pk)

            # success message and redirect back to project view
        messages.success(self.request,
                         'Successfully deleted Image '+image_name,
                         extra_tags='success')
        return redirect('mcd:detailed_record', image_parent_record.pk)

    success_url = reverse_lazy('mcd:index')




# --------------------------- MCD_Project -------------------------------- #
class ObjectCreate(CreateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Project

    # what attributes do we allow the user to input?
    # fields = ['title', ]
    # exclude = ['uploaded_by_user_id', 'num_photos']
    required_css_class = 'required'
    fields = ['title',
              'latitude',
              'longitude',
              'placename',
              'county',
              'address',
              'postcode']

    def form_valid(self, form):
        print("in form_valid (self.request.user |", self.request.user, ")")
        form.instance.uploaded_by_user_id = self.request.user

        # save the changes made to the database ...
        # ... and get the new assigned ID (primary key by task.id)
        form.save()

        # return super(ObjectCreate, self).form_valid(form)
        messages.success(self.request,
                         'Project "'+form.instance.title+'" successfully created!',
                         extra_tags='success')
        return redirect('mcd:object-list')
# =============================== Filler Views ================================== #

def upload_to_analysis(request):
    return render(request, template_name='mcd/upload_to_analysis.html')


def submit_to_analysis(request):
    try:
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
        return render(request, template_name='mcd/upload_to_analysis.html',
                      context={
                          'success_message': "Image "+image_to_analyse+" successfully uploaded for analysis"
                      })

def activate(request, uidb64, token):
    print("reached 'activate', with uidb64", uidb64, ", token", token)
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except(TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()
        login(request, user)
        # return redirect('home')
        # return HttpResponse('Thank you for your email confirmation. Now you can login your account.')
        messages.warning(request,
                         'Thank you for your email confirmation. Now you can login your account.',
                         extra_tags='success')
        return redirect('mcd:login')
    else:
        return HttpResponse('Activation link is invalid!')

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
            # set the user as inactive until they confirm their email:
            user.is_active = False
            user.save()

            print("[INFO] New User Saved")

            current_site = get_current_site(request)
            mail_subject = 'Activate your MCD account.'
            message = render_to_string('mcd/activate_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uidb64': force_text(urlsafe_base64_encode(force_bytes(user.pk))),
                'token': account_activation_token.make_token(user),
            })
            to_email = form.cleaned_data.get('email')

            print("[INFO] Sending email (to_email)", str(to_email)[:6], "...")

            email = EmailMessage(
                mail_subject, message, to=[to_email]
            )
            email.send()

            print("[INFO] Email was successfuly sent to ", str(to_email)[:6], "...")
            # return HttpResponse('Please confirm your email address to complete the registration')

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
                # return render(request, self.template_name, {'form': form})
                messages.warning(request,
                                 'Warning: The newly created account is not active, '
                                 'please check your email inbox and confirm your email before logging in',
                                 extra_tags='warning')
                return redirect('mcd:login')

        else:
            print("form invalid:")
            print(form.errors)

            messages.error(request,
                           form.errors.as_text(),
                           extra_tags='danger')
            return redirect('mcd:register')



# [DEPRECATED] - use add_scale_3
def add_scale(request, pk):

    display_image = MCD_Photo_Analysis.objects.get(pk=pk)

    try:
        (cx,cy)=list(request.GET.keys())[0].split(',')

    except:
        cx = 0
        cy = 0
    # map was clicked at cx,cy coordinates
    x=int(cx)
    y=int(cy)

    return render(request, "mcd/add_scale.html", {'display_image' : display_image,
                                                  'clicked_x' : x,
                                                  'clicked_y' : y})
# [DEPRECATED] - use add_scale_3
def add_scale_2(request, pk, cx, cy):

    display_image = MCD_Photo_Analysis.objects.get(pk=pk)

    try:
        (cx2,cy2)=list(request.GET.keys())[0].split(',')

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
        print("[INFO] GOT POST form: ", request.POST.get('input_real_length'),
                                     request.POST.get('input_px_length'))

        real_length = float(request.POST.get('input_real_length'))
        px_length   = float(request.POST.get('input_px_length'))

        print("[INFO] Computed ratio: ", real_length/px_length)

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
        else:
            # user = form.get_user()
            try: # give a custom message when account has not been activated:
                user = User.objects.get(username=form.cleaned_data['username'])

                if user.is_active:
                    messages.error(request,
                                   'Invalid Log In Credentials',
                                   extra_tags='danger')
                else:
                    messages.warning(request,
                                     'Warning: The account is not active, please confirm your email before logging in',
                                     extra_tags='warning')
                return redirect('mcd:login')
            except: # when the account is active:
                messages.error(request,
                               'Invalid Log In Credentials',
                               extra_tags='danger')
                return redirect('mcd:login')

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


# AJAX
def get_user_records(request):
    # user_id = request.GET.get('user_id')
    project_id = request.GET.get('project_id')

    print("[INFO] Getting records of project: ", project_id )
    
    # current_user = User.objects.get(pk=user_id)
    current_project = MCD_Project.objects.get(pk=project_id)
    
    # first filter records that user has uploaded:
    # mcd_records = MCD_Record.objects.filter(uploaded_by_user_id=current_user).all()

    # then add filter that only match the project as well:
    mcd_records = MCD_Record.objects.filter(project_id=current_project.pk).all()

    # return render(request, 'persons/city_dropdown_list_options.html', {'cities': cities})
    return JsonResponse(list(mcd_records.values('id', 'title')), safe=False)


def get_filtered_projects(request):
    # user_id = request.GET.get('user_id')
    search_query = request.GET.get('search_query')

    # first filter records that user has uploaded:
    mcd_projects = MCD_Project.objects.filter(uploaded_by_user_id=request.user).all()

    # then add filter that only match the project as well:
    mcd_projects = mcd_projects.filter(title__icontains=search_query).all()

    # return render(request, 'persons/city_dropdown_list_options.html', {'cities': cities})
    return JsonResponse(list(mcd_projects .values('id', 'num_records', 'title')), safe=False)


def get_filtered_images(request):
    # user_id = request.GET.get('user_id')
    search_query = request.GET.get('search_query')

    # first filter records that user has uploaded:
    mcd_photos = MCD_Photo_Analysis.objects.filter(uploaded_by_user_id=request.user).all()

    # then add filter that only match the project as well:
    mcd_photos = mcd_photos.filter(title__icontains=search_query).all()

    # return render(request, 'persons/city_dropdown_list_options.html', {'cities': cities})
    return JsonResponse(list(mcd_photos.values('id', 'project_id__title', 'analysis_complete', 'title')), safe=False)


def get_filtered_records(request):
    # user_id = request.GET.get('user_id')
    search_query = request.GET.get('search_query')

    # first filter records that user has uploaded:
    mcd_record = MCD_Record.objects.filter(uploaded_by_user_id=request.user).all()
    mcd_record = mcd_record.filter(project_id=request.GET.get('project_id')).all()

    # then add filter that only match the project as well:
    mcd_record = mcd_record.filter(title__icontains=search_query).all()

    # return render(request, 'persons/city_dropdown_list_options.html', {'cities': cities})
    return JsonResponse(list(mcd_record.values('id', 'num_images', 'title')), safe=False)