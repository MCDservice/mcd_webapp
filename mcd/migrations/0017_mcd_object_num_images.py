# Generated by Django 3.2.4 on 2021-07-08 12:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mcd', '0016_auto_20210708_1255'),
    ]

    operations = [
        migrations.AddField(
            model_name='mcd_object',
            name='num_images',
            field=models.IntegerField(default=0),
        ),
    ]
