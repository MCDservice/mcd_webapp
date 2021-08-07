# Template on how we will be storing data
import os

from django.db import models
from django.urls import reverse
# import the Users,  managed by Django framework:
from django.contrib.auth.models import User
from django_countries.fields import CountryField

# Create your models here.

# The Users database model
class MCD_Users(models.Model):
    # id          = models.IntegerField(primary_key=True)
    email         = models.CharField(max_length=100, unique=True, null=False, blank=False)
    password      = models.CharField(max_length=60,  null=False, blank=False)

    def __str__(self):
        return self.email

class MCD_Project(models.Model):
    uploaded_by_user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    title               = models.CharField(max_length=100, null=False, default="Untitled Object")
    latitude            = models.FloatField(null=True, blank=True)
    longitude           = models.FloatField(null=True, blank=True)
    placename           = models.CharField(max_length=100, null=True, blank=True)
    county              = CountryField(default=None, blank=True, blank_label='(select country)', multiple=True)
    address             = models.CharField(max_length=100, null=True, blank=True)
    postcode            = models.CharField(max_length=15, null=True, blank=True)
    num_records         = models.IntegerField(null=False, default=0)
    num_images          = models.IntegerField(null=False, default=0)

    # whenever we create a new photo analysis record, ...
    # ... add to database and redirect to the record detailed page::
    def get_absolute_url(self):
        return reverse('mcd:index')        # make the url /detail/pk)
        #        return reverse('mcd:photo_analysis_detailed', # view to redirect to
        #                kwargs={'pk' : self.pk})       # make the url /detail/pk)

    def __str__(self):
        return "#"+str(self.pk) + ' "' + str(self.title)+'"' + '" | by ' + str(self.uploaded_by_user_id)

class MCD_Record(models.Model):
    """
    by default, when user selects 'upload new photo', create a record
    """
    title = models.CharField(max_length=100, null=False, default="Untitled Record")
    uploaded_by_user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    project_id = models.ForeignKey(MCD_Project, default=1, on_delete=models.CASCADE)

    num_images = models.IntegerField(null=False, default=0)

    def __str__(self):
        # return "#" + str(self.pk) + ' "' + str(self.title)+'"'
        return "(user: " + str(self.uploaded_by_user_id) + \
               ") | " + str(self.title)



class MCD_Photo_Analysis (models.Model):
    uploaded_by_user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    project_id           = models.ForeignKey(MCD_Project, default=1, on_delete=models.CASCADE)
    record_id           = models.ForeignKey(MCD_Record, null=True, blank=True, on_delete=models.CASCADE)

    title               = models.CharField(max_length=100, null=False, default="Untitled")
    input_photo         = models.FileField(null=False, blank=False)
    overlay_photo       = models.FileField(null=True, blank=True)
    output_photo        = models.FileField(null=True, blank=True)
    crack_labels_photo  = models.FileField(null=True, blank=True)

    crack_labels_csv    = models.FileField(null=True, blank=True)
    analysis_complete   = models.BooleanField(null=False, default=False)

    crack_length        = models.DecimalField(null=True, default=-1,
                                              max_digits=8, decimal_places=2)

    scale               = models.DecimalField(null=True, default=1,
                                              max_digits=8, decimal_places=2)

    datetime_taken          = models.DateTimeField(null=True)
    datetime_uploaded       = models.DateTimeField(null=True)
    datetime_analysed       = models.DateTimeField(null=True)

    def filename(self):
        return os.path.basename(self.input_photo.name)

    # whenever we create a new photo analysis record, ...
    # ... add to database and redirect to the record detailed page::
    def get_absolute_url(self):
        return reverse('mcd:detailed_record_image_pk', # view to redirect to
                       kwargs={ 'pk' : self.record_id.pk,
                                'image_pk' : self.pk})       # make the url /detail/pk)

    def __str__(self):
        return "(user: " + str(self.uploaded_by_user_id) + \
               ") IN: " + str(self.input_photo) + \
               " | OUT: " + str(self.output_photo)

