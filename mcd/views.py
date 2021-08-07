import datetime

import math
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
from django.http import HttpResponseForbidden

from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib import messages


# modules for user account registration:
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
# from .forms import SignupForm
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from .tokens import account_activation_token
from django.contrib.auth.models import User
from django.core.mail import EmailMessage
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
# from .F_Use_Model import analyse_photo
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
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_secret_477185057888-brm030gcqnjoo7uijrijesp1ogi8hkah.apps.googleusercontent.com.json'

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


class ObjectDetailsView(UserPassesTestMixin, generic.DetailView):
    model = MCD_Project
    # change the name to refer to photo analysis
    # ... (in the template) as this:
    context_object_name = 'project'
    template_name = 'mcd/detailed_project.html'

    def test_func(self):
        # not allow the user who has not uploaded it to access the data
        requested_project = MCD_Project.objects.get(pk=self.kwargs['pk'])

        print("TESTING AUTH for pk", self.kwargs['pk'])
        print("for user id ", self.request.user.pk)
        print("analysis uploaded by: ", requested_project.uploaded_by_user_id.pk)

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


def read_csv_from_cloud(csv_filename, newline_char):
    csv_cloud_url = 'gs://' + conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET + \
                    '/' + conf_settings.MEDIA_DIR_NAME + csv_filename.replace(
        '\\', '/')
    print(">>> reading csv from URL: ", csv_cloud_url)

    try:
        processed_csv = pd.read_csv(csv_cloud_url, sep=",", lineterminator=newline_char)
        print(">  >  > dataframe: ", list(processed_csv))
        print(">  >  > last heading: ", list(processed_csv)[-1])

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
        if self.kwargs["comparison"] == "comparison1":
            return ['mcd/detailed_record_comparison1.html']
        elif self.kwargs["comparison"] == "comparison2":
            return ['mcd/detailed_record_comparison2.html']
        else:
            return ['mcd/detailed_record_comparison1.html']


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
            print(">>> Reading detailed crack information from", display_image_1.crack_labels_csv)
            relative_csv_path = get_cloud_relative_path_from_folder("media", display_image.crack_labels_csv.url)
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
    print("got url", url)
    return url.split(os.path.join(folder, ''), 1)[1]

def get_cloud_relative_path_from_folder(folder, url):
    print("got url", url)
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

        print("TESTING AUTH for pk", self.kwargs['pk'])
        print("for user id ", self.request.user.pk)
        print("analysis uploaded by: ", mcd_record.uploaded_by_user_id.pk)

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


class EnqueuePhotoAnalysis(threading.Thread):
    """
    after the user uploads their image, (or requests an update) ...
    ... enqueue the task and submit to the F_Use_Model.py to run it
    """
    def __init__(self, db_pk, title, user_id, input_url, output_url, completed,
                 user, project_id, record_id, analysis_id):
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

        threading.Thread.__init__(self)

    def run(self):
        # change field to say that it is currently processing:
        t = MCD_Photo_Analysis.objects.get(id=self.db_pk)
        t.analysis_complete = False
        t.save()  # this will only that analysis is not complete

        overlay_photo_url, \
        output_photo_url,\
        crack_len_url,\
        crack_labels_url = analyse_photo(self.input_url.url, self.title,
                                         self.user, self.project_id,
                                         self.record_id, self.analysis_id)

        print("crack_len_url", crack_len_url)
        print("photo analysed, posting to db index:", self.db_pk)

        # get biggest crack length:
        # sizes = pd.read_csv(conf_settings.MEDIA_URL.split('/')[1]+"\\"+crack_len_url)

        # "https://storage.cloud.google.com/mcd_file_storage/media/142_epoch_45_f1_m_dil_0.796/a_6_32/Sizes.csv"

        # sizes = pd.read_csv(conf_settings.MEDIA_URL.split('/')[1]+"\\"+crack_len_url)
        # csv_cloud_url = conf_settings.MEDIA_URL + crack_len_url.replace('\\', '/')
        csv_cloud_url = 'gs://'+conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET + \
                        '/'+ conf_settings.MEDIA_DIR_NAME + crack_len_url.replace('\\', '/')
        print(">>> reading csv from URL: ", csv_cloud_url )

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

        print(">  >  > 541 dataframe: ", list(sizes))
        print(">  >  > 542 dataframe: ", sizes)

        # after the photo has been analysed ...
        t.output_photo       = output_photo_url  # change field
        t.overlay_photo      = overlay_photo_url  # change field
        t.crack_labels_csv = crack_len_url  # change field
        t.crack_labels_photo = crack_labels_url  # change field
        t.analysis_complete  = True # change field

        t.datetime_analysed = datetime.datetime.now()

        print("overlay_photo url:", t.overlay_photo)

        t.save() # this will update only

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

    print('Original Dimensions : ', img.shape)

    # scale_percent = 60  # percent of original size
    scale = (math.trunc((img.shape[1] * img.shape[0]) / 1000000)) / 1.5
    if scale < 1:
        scale = 1

    print("pixels:", img.shape[1] * img.shape[0], "scale:", scale)

    width = int(img.shape[1] / scale)
    height = int(img.shape[0] / scale)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    img_str = cv2.imencode('.jpg', resized)[1].tostring()

    return img_str

# =============================== Filler Views ================================== #
# --------------------------- MCD_Photo_Analysis -------------------------------- #
def upload_file_to_cloud(file_to_upload, filename, resize=False):

    if resize:
        file_to_upload = resize_img(file_to_upload.read())
        # im = Image.fromarray(resizelist[val])

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(conf_settings.GOOGLE_CLOUD_STORAGE_BUCKET)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(conf_settings.MEDIA_DIR_NAME + filename)

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

        # in the upload form, let the user choose only from projects/records they themselves have created:
        # (filter out records and projects by user id)
        objects_per_user = MCD_Project.objects.filter(uploaded_by_user_id=request.user) \
            .values_list('uploaded_by_user_id', flat=True).first()

        records_per_user = MCD_Record.objects.filter(uploaded_by_user_id=request.user) \
            .values_list('uploaded_by_user_id', flat=True).first()
        # limit the form fields:
        form.fields['project_id'].queryset=MCD_Project.objects.filter(uploaded_by_user_id=objects_per_user)
        form.fields['record_id'].queryset=MCD_Record.objects.filter(uploaded_by_user_id=records_per_user)

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

        # get the file name uploaded to upload as the title
        # (remove the extension - split by '.'):
        if form.instance.title and form.instance.title != "Untitled":
            uploaded_filename = form.instance.title
        else:
            uploaded_filename = str(form.instance.input_photo).rsplit('.', 1)[0]

        current_datetime = datetime.datetime.now()

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

        # clear temporarily saved files in App Engine /tmp/
        clear_tmp_dir()

        # save the changes made to the database ...
        # ... and get the new assigned ID (primary key by task.id)
        task = form.save()
        print("form pk2:", form.instance.pk)

        # update the number of records of an object:
        from django.db.models import Count
         # annotate the queryset
        num_images = MCD_Project.objects.annotate(number_of_images=Count('mcd_record'))

        print(">>> num_images", num_images)

        parent_object = MCD_Project.objects.get(pk=form.instance.project_id.pk)
        print("> > > ", parent_object.mcd_photo_analysis_set.count())
        # for numbered_object in num_images:
        # print("> > >", parent_object.title, parent_object.number_of_images)
        parent_object.num_images = parent_object.mcd_photo_analysis_set.count()
        parent_object.save()

        print("enqueue IN  photo url: ", form.instance.input_photo)
        print("enqueue OUT photo url: ", form.instance.output_photo)

        EnqueuePhotoAnalysis(task.id,
                             form.instance.title,
                             form.instance.uploaded_by_user_id,
                             form.instance.input_photo,
                             form.instance.output_photo,
                             form.instance.analysis_complete,
                             hide_username(self.request.user), form.instance.project_id.pk,
                             mcd_record.pk, form.instance.pk
                             ).start()

        return super(PhotoAnalysisCreate, self).form_valid(form)


class ProjectUpdate(generic.UpdateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Project
    # form = 'mcd/mcd_record_update_form.html'

    # what attributes do we allow the user to edit?
    # fields = ['title', 'project_id']
    template_name_suffix = '_update_form'
    previous_name = ""

    form_class = MCD_ProjectFORM

    # def form_valid(self, form):
    #     print("form filled in", form.instance.reanalyse)
    #     return super(PhotoAnalysisUpdate, self).form_valid(form)

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']

        data = super().get_context_data(**kwargs)
        data['current_id'] = int(current_object)
        return data

    def form_invalid(self, form):
        print("GOT form invalid")
        print("form is invalid")
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


# class RecordDelete(DeleteView):
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

            print("ID853 user saved")

            current_site = get_current_site(request)
            mail_subject = 'Activate your MCD account.'
            message = render_to_string('mcd/activate_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uidb64': force_text(urlsafe_base64_encode(force_bytes(user.pk))),
                'token': account_activation_token.make_token(user),
            })
            to_email = form.cleaned_data.get('email')

            print("ID879 email (to_email)", to_email)

            email = EmailMessage(
                mail_subject, message, to=[to_email]
            )
            email.send()

            print("email is sent to ", to_email)

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


    # def get(self):
    #
    # def post(self, request):
    #     kwargs = {'data': request.POST}
    #     try:
    #         kwargs['instance'] = User.objects.get(username=request.POST['username'])
    #     except:
    #         pass
    #     form = UserForm(kwargs **)
    #     if form.is_valid():
    #         user = form.save(commit=False)


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

    # def form_valid(self, form):
    #     print("form filled in", form.instance.reanalyse)
    #     return super(PhotoAnalysisUpdate, self).form_valid(form)

    def get_context_data(self, **kwargs):
        # get current object primary key (id):
        current_object = self.kwargs['pk']

        data = super().get_context_data(**kwargs)
        data['current_id'] = int(current_object)
        return data

    # def form_invalid(self, form):
    #     return HttpResponse("Form is invalid")

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        # if form_valid(self, form):
        if form.is_valid():
            print("editing form valid")
            print(form.fields["title"])
            print(form.fields["project_id"])
            print(form.cleaned_data.get("title"))
            print(form.cleaned_data.get("project_id"))
            print("end")

            pk = self.kwargs.get("pk")
            print("pk = ", pk)

            current_record = MCD_Record.objects.get(pk=pk)
            self.previous_name = current_record.title
            self.previous_proj = current_record.project_id.pk

            current_record.title = form.cleaned_data.get("title")#form.fields["title"]
            current_record.project_id = form.cleaned_data.get("project_id") #form.fields["project_id"]

            current_record.save()
            # return super(RecordUpdate, self).form_valid(form)

            print("debug recv POST", pk)
            # form = self.form_class(None)

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


    # def post(self, request, pk):
    #     print("debug recv POST", pk)
    #     # form = self.form_class(None)
    #
    #     current_record = MCD_Record.objects.get(pk=pk)
    #     current_project = current_record.project_id
    #
    #     messages.success(request,
    #                      'Successfully renamed record from <b>'+self.previous_name+' to <b>'+current_record.title+'</b>.',
    #                      extra_tags='success')
    #     return redirect('mcd:detailed_object', current_project.pk)

def move_record(record_to_move, from_project, to_project):

    print("moving record ", record_to_move.title)
    print("photo analysis connected with this record:")

    image_ids_in_record = MCD_Photo_Analysis.objects.filter(record_id=record_to_move.pk) \
        .values_list('record_id', flat=True).first()
    associated_img_analysis = MCD_Photo_Analysis.objects.filter(record_id=image_ids_in_record)

    print("Listing:", associated_img_analysis)

    for img_analysis in associated_img_analysis:
        print(">>> moving record from: ", from_project)
        print("    moving record to: ", to_project)
        img_analysis.project_id = to_project
        img_analysis.save()
        print("------------------")

    # get current project:

    # compute how many records are uploaded in total in the project:
    from_project.num_records = from_project.mcd_record_set.count()
    from_project.num_images = from_project.mcd_photo_analysis_set.count()
    from_project.save()

    to_project.num_records = to_project.mcd_record_set.count()
    to_project.num_images = to_project.mcd_photo_analysis_set.count()
    to_project.save()


def delete_record(record_to_delete):

    print("deleting record ", record_to_delete.title)
    print("photo analysis connected with this record:")

    image_ids_in_record = MCD_Photo_Analysis.objects.filter(record_id=record_to_delete.pk) \
        .values_list('record_id', flat=True).first()
    associated_img_analysis = MCD_Photo_Analysis.objects.filter(record_id=image_ids_in_record)

    print("Listing:", associated_img_analysis)

    for img_analysis in associated_img_analysis:
        print("deleting 'input_photo':", img_analysis.input_photo.name)
        delete_file_from_cloud_media(img_analysis.input_photo.name)

        print("deleting 'overlay_photo':", img_analysis.overlay_photo.name)
        delete_file_from_cloud_media(img_analysis.overlay_photo.name)

        print("deleting 'output_photo':", img_analysis.output_photo.name)
        delete_file_from_cloud_media(img_analysis.output_photo.name)

        print("deleting 'crack_labels_photo':", img_analysis.crack_labels_photo.name)
        delete_file_from_cloud_media(img_analysis.crack_labels_photo.name)

        print("deleting 'crack_labels_csv':", img_analysis.crack_labels_csv.name)
        delete_file_from_cloud_media(img_analysis.crack_labels_csv.name)

        print("------------------")

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

    # messages.success(self.request,
    #                  'Successfully renamed record from <b>' + self.previous_name + ' to <b>' + current_record.title + '</b>.',
    #                  extra_tags='success')
    # return redirect('mcd:detailed_object', current_project.pk)
    success_url = reverse_lazy('mcd:index')


class PhotoAnalysisUpdate(LoginRequiredMixin, UpdateView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis
    template_name = 'mcd/mcd_photo_analysis_update_form.html'

    context_object_name = 'photo_analysis'

    # what attributes do we allow the user to input?
    # fields = ['input_photo', 'project_id', 'record_id', 'scale']
    form_class = MCD_Photo_AnalysisFORM

    # def form_valid(self, form):
    #     print("form filled in", form.instance.reanalyse)
    #     return super(PhotoAnalysisUpdate, self).form_valid(form)

    def form_valid(self, form):

        pk = self.kwargs.get("pk")
        new_title = form.cleaned_data.get("title")

        current_img = MCD_Photo_Analysis.objects.get(pk=pk)
        self.previous_name = current_img.title

        msg_change = 'Successfully changed details of Image "'+self.previous_name+'"'
        if self.previous_name != new_title :
            msg_change += ' (new title "'+new_title+'")'

        form.save()

        messages.success(self.request,
                         msg_change,
                         extra_tags='success')
        return redirect('mcd:detailed_record_image_pk', current_img.record_id.pk, pk)

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

    # def post(self, request, pk):
    #     print("recv POST", pk)
    #     # form = self.form_class(None)
    #
    #     current_image = MCD_Photo_Analysis.objects.get(pk=pk)
    #
    #     print("current form:", current_image.title)
    #     print(current_image.uploaded_by_user_id)
    #     print(current_image.input_photo)
    #     print(current_image.output_photo)
    #     print(current_image.analysis_complete)
    #
    #     EnqueuePhotoAnalysis(pk,
    #                          current_image.title,
    #                          current_image.uploaded_by_user_id,
    #                          current_image.input_photo,
    #                          current_image.output_photo,
    #                          current_image.analysis_complete,
    #                          hide_username(self.request.user), current_image.project_id.pk,
    #                          current_image.record_id.pk, current_image.pk
    #                          ).start()
    #
    #     return redirect('mcd:detailed_record_image_pk', current_image.record_id.pk, pk)

class PhotoAnalysisReanalyse(View):

    model = MCD_Photo_Analysis
    template_name = 'mcd/mcd_photo_analysis_confirm_reanalysis.html'

    def get(self, request, pk):
        return render(request, self.template_name, {'pk' : pk})

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
                             current_image.analysis_complete,
                             hide_username(self.request.user), current_image.project_id.pk,
                             current_image.record_id.pk, current_image.pk
                             ).start()

        # return redirect('mcd:index')
        return redirect('mcd:detailed_record_image_pk', current_image.record_id.pk, pk)

class PhotoAnalysisDelete(DeleteView):
    # database model we will allow the user to edit/fill-in
    model = MCD_Photo_Analysis

    def test_func(self):
        # not allow the user who has not uploaded it to access the data
        requested_image = MCD_Photo_Analysis.objects.get(pk=self.kwargs['pk'])
        if not requested_image.uploaded_by_user_id.pk == self.request.user.pk:
            return False
        else:
            return True

    def post(self, request, *args, **kwargs):
        image_to_delete = self.get_object()
        image_name = image_to_delete.title
        image_parent_record = image_to_delete.record_id
        image_parent_project = image_to_delete.project_id

        print("deleting 'input_photo':", image_to_delete.input_photo.name)
        delete_file_from_cloud_media(image_to_delete.input_photo.name)

        print("deleting 'overlay_photo':", image_to_delete.overlay_photo.name)
        delete_file_from_cloud_media(image_to_delete.overlay_photo.name)

        print("deleting 'output_photo':", image_to_delete.output_photo.name)
        delete_file_from_cloud_media(image_to_delete.output_photo.name)

        print("deleting 'crack_labels_photo':", image_to_delete.crack_labels_photo.name)
        delete_file_from_cloud_media(image_to_delete.crack_labels_photo.name)

        print("deleting 'crack_labels_csv':", image_to_delete.crack_labels_csv.name)
        delete_file_from_cloud_media(image_to_delete.crack_labels_csv.name)

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
            image_parent_record.delete()
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

    # messages.success(self.request,
    #                  'Successfully renamed record from <b>' + self.previous_name + ' to <b>' + current_record.title + '</b>.',
    #                  extra_tags='success')
    # return redirect('mcd:detailed_object', current_project.pk)
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

            print("ID853 user saved")

            current_site = get_current_site(request)
            mail_subject = 'Activate your MCD account.'
            message = render_to_string('mcd/activate_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uidb64': force_text(urlsafe_base64_encode(force_bytes(user.pk))),
                'token': account_activation_token.make_token(user),
            })
            to_email = form.cleaned_data.get('email')

            print("ID879 email (to_email)", to_email)

            email = EmailMessage(
                mail_subject, message, to=[to_email]
            )
            email.send()

            print("email is sent to ", to_email)
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
        print("login form posted ... ")
        print("form:", form, "valid? ", form.is_valid())
        if form.is_valid():
            # log in the user ...
            user = form.get_user()
            print("user - active? ", user.is_active)
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
                print("form invalid, is user active?", user.is_active)
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

