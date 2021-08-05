import django
from django.contrib.auth.models import User
from .models import MCD_Photo_Analysis, MCD_Project, MCD_Record
from django import forms

# make new user form class

class UserForm(forms.ModelForm):
    # make password appear as '********'
    password = forms.CharField(widget=forms.PasswordInput)
    # make a blueprint
    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def clean_email(self):
        # Get the email
        email = self.cleaned_data.get('email')

        # Check to see if any users already exist with this email as a username.
        try:
            match = User.objects.get(email=email)
        except User.DoesNotExist:
            # Unable to find a user, this is fine
            return email

        # A user was found with this as a username, raise an error.
        raise forms.ValidationError('This email address is already in use.', code='invalid')


class MCD_RecordFORM(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        # self.current_user_id = kwargs.pop('uploaded_by_user_id', None)
        super(MCD_RecordFORM, self).__init__(*args, **kwargs)

    # project_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(user_id))
    class Meta:
        model = MCD_Record
        fields = ['title', 'project_id']

class MCD_ProjectFORM(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        # self.current_user_id = kwargs.pop('uploaded_by_user_id', None)
        super(MCD_ProjectFORM, self).__init__(*args, **kwargs)

    # project_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(user_id))
    class Meta:
        model = MCD_Project
        fields = ['title']

class UserUpdateForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        # self.current_user_id = kwargs.pop('uploaded_by_user_id', None)
        super(UserUpdateForm, self).__init__(*args, **kwargs)

    # project_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(user_id))
    class Meta:
        model = User
        fields = ['email']

    def clean_email(self):
        # Get the email
        email = self.cleaned_data.get('email')

        # Check to see if any users already exist with this email as a username.
        try:
            match = User.objects.get(email=email)
        except User.DoesNotExist:
            # Unable to find a user, this is fine
            return email
        except django.contrib.auth.models.User.MultipleObjectsReturned:
            # A user was found with this as a username, raise an error.
            raise forms.ValidationError('This email address is already in use.', code='invalid')

        # A user was found with this as a username, raise an error.
        raise forms.ValidationError('This email address is already in use.', code='invalid')


# class MCD_Photo_AnalysisFORM(forms.ModelForm):
class MCD_Photo_AnalysisFORM(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        # self.current_user_id = kwargs.pop('uploaded_by_user_id', None)
        super(MCD_Photo_AnalysisFORM, self).__init__(*args, **kwargs)

    # project_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(user_id))
    class Meta:
        model = MCD_Photo_Analysis
        fields = ['project_id', 'record_id', 'title', 'input_photo']

        # fields['project_id'].queryset = MCD_Project.objects.filter(uploaded_by_user_id=self.current_user_id) \
        #     .values_list('uploaded_by_user_id', flat=True).first()
        # # return only the objects that match the current user logged in:
        # project_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(uploaded_by_user_id=objects_per_user))

