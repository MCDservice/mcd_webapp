# urls specifically for 'masonry crack detection' (mcd) application
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views

from .forms import UserForm, MCD_Photo_AnalysisFORM, MCD_RecordFORM

# import mcd views:
from . import views

# imports specifically for REST API:
from rest_framework.urlpatterns import format_suffix_patterns

# namespace the urls of the individual application:
app_name = 'mcd'

urlpatterns = [
    # path('admin/', admin.site.urls),
    # /mcd/ (default/home page):
    url(r'^$', views.IndexView.as_view(), name='index'),

    # /mcd/image-list
    url(r'^image-list/$', login_required(login_url="/mcd/login")
                          (views.ListAllImagesView.as_view()), name="image-list"),

    # /mcd/list-images
    url(r'^object-list/$', login_required(login_url="/mcd/login")
                           (views.ListAllObjectsView.as_view()), name="object-list"),

    # /mcd/ID/ (detailed view of the photo analysis)
    url(r'^(?P<pk>[0-9]+)/$', views.DetailsView.as_view(),
        name='photo_analysis_detailed'),

    # /mcd/object/ID/ (detailed view of the photo analysis)
    url(r'^object/(?P<pk>[0-9]+)/$', views.ObjectDetailsView.as_view(),
        name='detailed_object'),

    # /mcd/object/ID/ (detailed view of the records)
    url(r'^record/(?P<pk>[0-9]+)/$', # login_required(login_url="/mcd/login"),
        views.RecordDetailsView.as_view(),
        name='detailed_record'),

    # /mcd/object/ID/ (detailed view of the records)
    url(r'^record/(?P<pk>[0-9]+)/image-(?P<image_pk>[0-9]+)/$', views.RecordDetailsView.as_view(),
        name='detailed_record_image_pk'),

    # /mcd/object/ID/ (comparing two images in the detailed view of the records)
    url(r'^record/(?P<pk>[0-9]+)/(?P<comparison>[\w\-]+)/(?P<image_pk1>[0-9]+)/(?P<image_pk2>[0-9]+)/$',
        views.RecordComparison1View.as_view(),
        name='detailed_record_compare1'),
    # url(r'^record/(?P<pk>[0-9]+)/(?P<comparison>[\w\-]+)/(?P<image_pk1>[0-9]+)/(?P<image_pk2>[0-9]+)/$', views.RecordComparison1View.as_view(),
    #     name='detailed_record_compare1'),

    url(r'^register/$', views.UserFormView.as_view(), name='register'),
    url(r'^login/$',  views.login_view, name='login'),
    url(r'^logout/$', views.logout_view, name='logout'),
    url(r'^activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,32})/$',
        views.activate, name='activate'),

    # url(r'^password-reset/$', auth_views.PasswordResetView.as_view(),
    #     {'template_name': "mcd/templates/registration/password_reset_form.html"},
    #     name='password_reset'),
    # url(r'^password-reset/done/$', auth_views.PasswordChangeDoneView.as_view(), name='password_reset_done'),
    # url(r'^reset/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$',
    #     auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    # url(r'^reset/done/$', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),

    # mcd/upload
    # [invoked by clicking ' + Upload Image for Analysis ']
    url(r'^upload/$', login_required(login_url="/mcd/login")
                      (views.PhotoAnalysisCreate.as_view()), name='upload'),

    url(r'^upload/(?P<project_id>[0-9]+)/(?P<record_id>[0-9]+)$', login_required(login_url="/mcd/login")
                      (views.PhotoAnalysisCreate.as_view()), name='upload_to_project_record'),

    url(r'^get-user-records', login_required(login_url="mcd/login")
                      (views.get_user_records), name='get_user_records'),

    # views for the SEARCH BOX:
    url(r'^get-filtered-projects', login_required(login_url="mcd/login")
                      (views.get_filtered_projects), name='get_filtered_projects'),
    url(r'^get-filtered-records', login_required(login_url="mcd/login")
                      (views.get_filtered_records), name='get_filtered_records'),
    url(r'^get-filtered-images', login_required(login_url="mcd/login")
                      (views.get_filtered_images), name='get_filtered_images'),

    # mcd/project-add
    url(r'^project-add/$', login_required(login_url="/mcd/login")
                      (views.ObjectCreate.as_view()), name='project-add'),

    url(r'^profile/$', login_required(login_url="/mcd/login")
    (views.UserProfile.as_view()), name='profile'),

    url(r'^profile-update/$', login_required(login_url="/mcd/login")
    (views.UserUpdate.as_view()), name='profile-update'),


    # mcd/update/pk
    url(r'^update/(?P<pk>[0-9]+)$', login_required(login_url="/mcd/login")
        (views.PhotoAnalysisUpdate.as_view()), name='update'),
    url(r'^delete/(?P<pk>[0-9]+)$', login_required(login_url="/mcd/login")
        (views.PhotoAnalysisDelete.as_view()), name='delete-photo'),
    url(r'^reanalyse/(?P<pk>[0-9]+)$', login_required(login_url="/mcd/login")
        (views.PhotoAnalysisReanalyse.as_view()), name='reanalyse-photo'),

    # mcd/edit-record/pk
    url(r'^edit-record/(?P<pk>[0-9]+)$', login_required(login_url="/mcd/login")(views.RecordUpdate.as_view()),
        name='edit-record'),
    # mcd/delete-record/pk
    url(r'^delete-record/(?P<pk>[0-9]+)$', login_required(login_url="/mcd/login")(views.RecordDelete.as_view()),
        name='delete-record'),

    # mcd/edit-project/pk
    url(r'^edit-project/(?P<pk>[0-9]+)$', login_required(login_url="/mcd/login")(views.ProjectUpdate.as_view()),
        name='edit-project'),
    # mcd/delete-project/pk
    url(r'^delete-project/(?P<pk>[0-9]+)$', login_required(login_url="/mcd/login")(views.ProjectDelete.as_view()),
        name='delete-project'),

    # add scale
    url(r'^add-scale/(?P<pk>[0-9]+)/$', views.add_scale, name='add-scale'),
    url(r'^add-scale/(?P<pk>[0-9]+)/(?P<x_co>[0-9]+),(?P<y_co>[0-9]+)$', views.add_scale, name='add-scale'),
    url(r'^add-scale-2/(?P<pk>[0-9]+)/(?P<cx>[0-9]+)/(?P<cy>[0-9]+)/$', views.add_scale_2, name='add-scale-2'),
    url(r'^add-scale-3/(?P<pk>[0-9]+)/(?P<cx>[0-9]+)/(?P<cy>[0-9]+)/(?P<point>[\w]{1})/$', views.add_scale_3, name='add-scale-3'),
    # url(r'^add-scale-2/(?P<pk>[0-9]+)/(?P<cx>[0-9]+)/(?P<cy>[0-9]+)/(?P<x_co>[0-9]+),(?P<y_co>[0-9]+)$',
    #     views.add_scale_2, name='add-scale-2'),

    url(r'^upload/submit$', views.submit_to_analysis, name='submit_to_analysis'),
    # url(r'^upload/submit$', views.submit_to_analysis, name='submit_to_analysis'),


    # urls specifically for the REST API:
    url(r'^api/list$', views.API_ListByUser_MCD_Photo_Analysis.as_view(), name='api_list'),

]

# urls specifically for the REST API:
urlpatterns = format_suffix_patterns(urlpatterns)