# Generated by Django 3.2.4 on 2021-06-26 18:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mcd', '0005_auto_20210623_1638'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mcd_photo_analysis',
            name='input_photo',
            field=models.FileField(upload_to=''),
        ),
        migrations.AlterField(
            model_name='mcd_photo_analysis',
            name='output_photo',
            field=models.FileField(blank=True, null=True, upload_to=''),
        ),
    ]