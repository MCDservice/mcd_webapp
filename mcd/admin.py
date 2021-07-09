from django.contrib import admin

# Register your models here.
from django.contrib import admin
from mcd.models import MCD_Users
from mcd.models import MCD_Photo_Analysis
from mcd.models import MCD_Object
from mcd.models import MCD_Record

admin.site.register(MCD_Users)
admin.site.register(MCD_Photo_Analysis)
admin.site.register(MCD_Object)
admin.site.register(MCD_Record)
