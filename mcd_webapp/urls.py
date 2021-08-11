"""mcd_webapp URL Configuration
(Table of contents for the website)

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.mcd, name='mcd')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='mcd')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include

# import settings (to make sure media files are handled)
from django.conf import settings
from django.conf.urls.static import static

from django.contrib.auth import views as auth_views
from django.shortcuts import render, redirect

def redirect_to_index(request):
    return redirect('mcd/')

def redirect_to_login(request):
    return redirect('/mcd/login/')


urlpatterns = [
    # path('admin/', admin.site.urls),
    url(r'^$', redirect_to_index, name='index_redirect'),
    url(r'^admin/', admin.site.urls),
    url(r'^mcd/', include('mcd.urls')),
    url(r'^accounts/login', redirect_to_login, name='login_redirect'),

    url(r'^password-reset/$', auth_views.PasswordResetView.as_view(),
        {'template_name': "mcd/templates/registration/password_reset_form.html"},
        name='password_reset'),
    url(r'^password-reset/done/$', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    url(r'^reset/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,32})/$',
        auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    url(r'^reset/done/$', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),

    # url(r'^$', view_homepage),
]

# in development mode - use the URL given
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)