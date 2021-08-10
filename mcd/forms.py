import django
from crispy_forms.bootstrap import FieldWithButtons, PrependedAppendedText, AppendedText, StrictButton
from crispy_forms.layout import Div, Layout, Button, Submit, Field, HTML
from django.contrib.auth.models import User
from .models import MCD_Photo_Analysis, MCD_Project, MCD_Record
from django import forms
from crispy_forms.helper import FormHelper

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
    required_css_class = 'required'
    def __init__(self, *args, **kwargs):
        # self.current_user_id = kwargs.pop('uploaded_by_user_id', None)
        super(MCD_ProjectFORM, self).__init__(*args, **kwargs)

    # project_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(user_id))
    class Meta:
        model = MCD_Project
        fields = ['title',
                  'latitude',
                  'longitude',
                  'placename',
                  'county',
                  'address',
                  'postcode']

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
# class MCD_Photo_AnalysisFORM(FormHelper):
    def __init__(self, current_user, project_id = None, record_id=None, *args, **kwargs):
        self.helper = FormHelper()
        super(MCD_Photo_AnalysisFORM, self).__init__(*args, **kwargs)

        print("MCD_Photo_AnalysisFORM - these fields in kwargs:")
        print(kwargs.keys())


        # self.fields['project_id'].initial
        self.fields['title'].label = "Image Title"

        self.helper.layout = Layout(
            HTML("<div class='tutorial-box'>"
                    "<span class='glyphicon glyphicon-info-sign'></span>&nbsp"
                    "<b>Note</b> Recommended use-case:"
                    "<br>"
                    "The <b>Project</b> can represent the entire House, whereas the <b>Record</b> can represent "
                    "individual Walls, where each Record consists of images of the same Wall throughout time."
                    "<br>"
                 "</div>"),
            'title',
            HTML("{% if current_id %} "
                    "<div class='tutorial-box explanatory-box margin-bottom-15px'>"
                         "<span class='glyphicon glyphicon-info-sign'></span>&nbsp"
                         "You can move the Image to a different Project or different Record here. "
                         "If 'Create Record Automatically' is selected, a Record will be created with the "
                         "associated Project (using the Image Title for the Record Name)"
                    "</div>"
                 "{% else %}"
                    "<div class='tutorial-box explanatory-box margin-bottom-15px'>"
                         "<span class='glyphicon glyphicon-info-sign'></span>&nbsp"
                         "Each Record in the Project is dedicated to <b>tracking the history</b> of the wall crack."
                         "<br>The intended use-case for Records is uploading images of the same wall, days/weeks/months "
                         "later. <br><br>This is to help see if the wall crack is getting larger."
                    "</div>"
                 "{% endif %}"),
            # Field('contact_email', readonly=True) if email_readonly else Field('contact_email'),
            # Field('project_id', readonly=True),
            # Field('record_id', readonly=True),
            # 'project_id',
            HTML("<div class='tutorial-text'>"
                    "<span class='glyphicon glyphicon-info-sign'></span>&nbsp"
                    "Each Image must be associated with a Project (House)"
                 "</div>"),
            FieldWithButtons('project_id', StrictButton('<span class="glyphicon glyphicon-plus"></span>',
                                                   id='add-project-btn',
                                                   onclick='window.location.href="{}"'.format(
                                                       "{% url 'mcd:project-add' %}"),
                                                   type='button',
                                                   css_class='btn btn-success')),
            HTML("{% if not project_id %} "
                     "<div class='tutorial-text'>"
                     "<span class='glyphicon glyphicon-info-sign'></span>&nbsp"
                        "If an existing Record ID is not chosen, a new one will be created automatically, "
                        "(it is possible to rename the Record auto-generated title by clicking "
                        "<span class='glyphicon glyphicon-pencil' aria-hidden='true'>)"
                     "</div>"
                 "{% endif %}"),
            'record_id',
            HTML('{% if not project_id %}'
                     '<div class="tutorial-text margin-bottom-15px">'
                        '<span class="glyphicon glyphicon-question-sign"></span>&nbsp'
                        '(example - "Project #1 House at 11 ABC Street" can have 2 records titled '
                        '"Record #1 - South Wall" and "Record #2 - North Wall", where each Record has images of '
                        'those walls months apart)'
                     '</div>'
                 '{% endif %}'),
            HTML("<div class='tutorial-box'>"
                     "<span class='glyphicon glyphicon-info-sign'></span>&nbsp"
                     "<b>Note</b> Understanding Scale:<br>"
                     "The 'scale' is a measurement of how many pixels in the image "
                     "make up one centimeter. <br> If scale is set, it is used to display crack length in centimetres rather than pixels <br>"
                     "{% if not current_id %}"
                         "It is difficult to know the scale, therefore, it is fine to leave it "
                         "set to '1.0', and then the crack length will be displayed in pixels. "
                     "{% endif %}"
                     "<br><br>"
                     "To set the scale interactively (draw a line on the image, and enter "
                     "the real length in centimetres), you can click the 'Update Scale Interactively' "
                     "button "
                     "{% if not current_id %}"
                         "(disabled now, since the image has not been uploaded yet) in the "
                         "Image Settings <span class='glyphicon glyphicon-cog'></span>&nbsp"
                         "or the Detailed Record View."
                     "{% endif %}"
                 "</div>"),
            FieldWithButtons('scale', StrictButton('Update Scale Interactively',
                                                   id='interactive-scale-btn',
                                                   onclick='window.location.href="{}"'.format('../add-scale-3/{{ current_id }}/1/1/g/?11,11'),
                                                   type='button',
                                                   css_class='btn btn-success')),
            HTML("{% if not current_id %}"
                     "<div class='tutorial-text margin-top-45px margin-bottom-15px'>"
                     "<span class='glyphicon glyphicon-info-sign'></span>&nbsp"
                     "Image will be sent for analysis after upload. It might take 3-10 minutes, depending on the size."
                     "<br>"
                     "If the image is not analysed after a longer time, you can click 'Reanalyse Photo' in the "
                     "Image Settings <span class='glyphicon glyphicon-cog'></span>&nbsp"
                     "</div>"
                 "{% endif %}"),
            'input_photo',
            Submit('submit', u'Submit', css_class='btn btn-success width100 margin-top-45px'),
        )
        self.helper.form_method = 'POST'
        self.helper.render_hidden_fields = True

        print("FORM GOT: project_id", project_id, "record_id", record_id)
        if project_id is not None and record_id is not None:
            self.fields['record_id'].queryset = self.fields['record_id'].queryset.filter(pk=record_id)
            self.fields['project_id'].queryset = self.fields['project_id'].queryset.filter(pk=project_id)

        else:
            self.fields['record_id'].queryset = MCD_Record.objects.none()
            self.fields['project_id'].queryset = self.fields['project_id'].queryset.filter(
                uploaded_by_user_id=current_user)

            print("GOT self.data:", self.data)
            if 'record_id' in self.data:
                try:
                    input_record_id = int(self.data.get('record_id'))
                    print("record_id:", input_record_id )
                    print("self.fields: ", self.fields)

                    valid_records = MCD_Record.objects.filter(pk=input_record_id ).order_by('pk')
                    print(valid_records)

                    self.fields['record_id'].queryset = valid_records
                except (ValueError, TypeError):
                    pass
                    # if the 'Create new record automatically' option was chosen ...
                    # ... then allow it (validate)
                    # if self.data.get('record_id') == None
                    #     self.fields['record_id'].queryset = MCD_Record.objects.none()

                    # else:
                    # invalid input from the client; ignore and fallback to empty queryset
                    # ... which will in on of itself generate an error for user to see and act upon

            # else, if we are EDITING/UPDATING the image (not CREATING as before ^)
            elif self.instance.pk:
                # then, we set the record query set to be all records that belong to the current project:
                # (mcd_record_set - gets all reecords associated with project_id)
                self.fields['record_id'].queryset = self.instance.project_id.mcd_record_set.order_by('pk')


    # project_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(user_id))
    class Meta:
        model = MCD_Photo_Analysis
        fields = ['title', 'project_id', 'record_id', 'scale', 'input_photo']

        # fields['project_id'].queryset = MCD_Project.objects.filter(uploaded_by_user_id=self.current_user_id) \
        #     .values_list('uploaded_by_user_id', flat=True).first()
        # # return only the objects that match the current user logged in:
        # project_id = forms.ModelChoiceField(queryset=MCD_Photo_Analysis.objects.filter(uploaded_by_user_id=objects_per_user))

