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


urlpatterns = [
    # path('admin/', admin.site.urls),
    url(r'^admin/', admin.site.urls),
    url(r'^mcd/', include('mcd.urls')),
    # url(r'^$', view_homepage),
]

# in development mode - use the URL given
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)