# Generated by Django 3.2.4 on 2021-07-15 14:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mcd', '0021_mcd_photo_analysis_overlay_photo'),
    ]

    operations = [
        migrations.AddField(
            model_name='mcd_photo_analysis',
            name='crack_labels_photo',
            field=models.FileField(blank=True, null=True, upload_to=''),
        ),
    ]
