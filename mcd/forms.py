from django.contrib.auth.models import User
from .models import MCD_Photo_Analysis, MCD_Object
from django import forms

# make new user form class

class UserForm(forms.ModelForm):
    # make password appear as '********'
    password = forms.CharField(widget=forms.PasswordInput)
    # make a blueprint
    class Meta:
        model = User
        fields = ['username', 'email', 'password']


# class MCD_Photo_AnalysisFORM(forms.ModelForm):
class MCD_Photo_AnalysisFORM(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        # self.current_user_id = kwargs.pop('uploaded_by_user_id', None)
        super(MCD_Photo_AnalysisFORM, self).__init__(*args, **kwargs)

    # object_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(user_id))
    class Meta:
        model = MCD_Photo_Analysis
        fields = ['object_id',  'record_id', 'title', 'input_photo']

        # fields['object_id'].queryset = MCD_Object.objects.filter(uploaded_by_user_id=self.current_user_id) \
        #     .values_list('uploaded_by_user_id', flat=True).first()
        # # return only the objects that match the current user logged in:
        # object_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(uploaded_by_user_id=objects_per_user))

