# urls specifically for 'masonry crack detection' (mcd) application
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from django.contrib.auth.decorators import login_required

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
    url(r'^record/(?P<pk>[0-9]+)/$', views.RecordDetailsView.as_view(),
        name='detailed_record'),

    # /mcd/object/ID/ (detailed view of the records)
    url(r'^record/(?P<pk>[0-9]+)/image-(?P<image_pk>[0-9]+)/$', views.RecordDetailsView.as_view(),
        name='detailed_record_image_pk'),

    # url(r'^upload/$', views.upload_to_analysis, name='upload'),
    url(r'^register/$', views.UserFormView.as_view(), name='register'),
    url(r'^login/$',  views.login_view, name='login'),
    url(r'^logout/$', views.logout_view, name='logout'),

    # mcd/upload
    # [invoked by clicking ' + Upload Image for Analysis ']
    url(r'^upload/$', login_required(login_url="/mcd/login")
                      (views.PhotoAnalysisCreate.as_view()), name='upload'),
    # mcd/project-add
    url(r'^project-add/$', login_required(login_url="/mcd/login")
                      (views.ObjectCreate.as_view()), name='project-add'),


    # mcd/update/pk
    url(r'^update/(?P<pk>[0-9]+)$', views.PhotoAnalysisUpdate.as_view(), name='update'),
    url(r'^delete/(?P<pk>[0-9]+)$', views.PhotoAnalysisDelete.as_view(), name='upload'),

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