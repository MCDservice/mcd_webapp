# Generated by Django 3.2.4 on 2021-07-04 22:11

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mcd', '0010_auto_20210704_2304'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='mcd_photo_analysis',
            name='object_id',
        ),
        migrations.DeleteModel(
            name='MCD_Object',
        ),
    ]