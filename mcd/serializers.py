"""
Take the models and converting it to data that ...
... can be saved on hard drive or sent over the network (transfer)

"""

from rest_framework import serializers
from .models import MCD_Photo_Analysis


class MCD_Photo_AnalysisSerializer(serializers.ModelSerializer):

    class Meta:
        # specify exact class intended to be serialized
        model = MCD_Photo_Analysis
        fields = ('input_photo', 'output_photo')
        # fields = '__all__'


