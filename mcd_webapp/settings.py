"""
Django settings for mcd_webapp project.

Generated by 'django-admin startproject' using Django 3.2.4.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.2/ref/settings/
"""
import os
from pathlib import Path
from .getKey import getKey1, getKey2

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-2r%bph2^-^0zz(%*szr7#9ag%f!2r8p7c&wilq*5!l25)-^7=t'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['cool-keel-320414.nw.r.appspot.com', '127.0.0.1']


# Application definition

INSTALLED_APPS = [
    'mcd.apps.HomeConfig',
    'anymail',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework', # for REST API
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'mcd_webapp.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')]
        ,
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
            'libraries': {
                'index_templatetag': 'mcd.templatetags.index_templatetag',

            }
        },
    },
]

WSGI_APPLICATION = 'mcd_webapp.wsgi.application'


# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases

# Old database:
#
# DATABASES = {
#    'default': {
#        'ENGINE': 'django.db.backends.sqlite3',
#        'NAME': BASE_DIR / 'db.sqlite3',
#    }
#}

# [START db_setup]
print("debug:",  os.getenv('GAE_APPLICATION', None)==True)
if os.getenv('GAE_APPLICATION', None):
    # Running on production App Engine, so connect to Google Cloud SQL using
    # the unix socket at /cloudsql/<your-cloudsql-connection string>
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'HOST': '/cloudsql/cool-keel-320414:europe-west2:polls-instance',
            'USER': 'polls',
            'PASSWORD': getKey1(),
            'NAME': 'mcd_webapp_test',
        }
    }
else:
    # Running locally so connect to either a local MySQL instance or connect to
    # Cloud SQL via the proxy. To start the proxy via command line:
    #
    #     $ cloud_sql_proxy -instances=[INSTANCE_CONNECTION_NAME]=tcp:3306
    #
    # See https://cloud.google.com/sql/docs/mysql-connect-proxy
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'HOST': '127.0.0.1',
            'PORT': '3306',
            'NAME': 'mcd_webapp_test',
            'USER': 'polls',
            'PASSWORD': getKey1(),
        }
    }
# [END db_setup]


# Password validation
# https://docs.djangoproject.com/en/3.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.2/howto/static-files/

STATIC_URL = 'staticfiles/'
STATIC_ROOT = BASE_DIR #os.path.join(BASE_DIR, 'staticfiles')

# Default primary key field type
# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# where the user uploaded data is stored!
# MEDIA_ROOT = os.path.join(BASE_DIR, 'media') # on the computer
MEDIA_ROOT = "/tmp" # on the computer
# MEDIA_URL = '/media/' # relative URL - for browsers to access files over HTTP

GOOGLE_CLOUD_STORAGE_BUCKET = 'mcd_file_storage'
# MEDIA_URL = 'https://storage.cloud.google.com/mcd_file_storage/media/'
MEDIA_DIR_NAME = "media/"
MEDIA_URL = 'https://storage.googleapis.com/mcd_file_storage/'+MEDIA_DIR_NAME
# MEDIA_ROOT = 'https://storage.googleapis.com/mcd_file_storage/media/'
MESSAGE_STORAGE = 'django.contrib.messages.storage.session.SessionStorage'


EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'

EMAIL_USE_TLS = True
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_HOST_USER = 'mcd.recover@gmail.com'
EMAIL_HOST_PASSWORD = getKey2()


AUTHENTICATION_BACKENDS = (
    # Needed to login by username in Django admin, regardless of `allauth`
    "django.contrib.auth.backends.ModelBackend",

    # `allauth` specific authentication methods, such as login by e-mail
    # "allauth.account.auth_backends.AuthenticationBackend",
)