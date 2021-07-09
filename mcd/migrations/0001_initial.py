# Generated by Django 3.2.4 on 2021-06-22 20:53

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Users',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('email', models.CharField(max_length=100, unique=True)),
                ('password', models.CharField(max_length=60, null=True)),
                ('input_photos', models.CharField(max_length=150, null=True)),
                ('output_photos', models.CharField(max_length=150, null=True)),
            ],
        ),
    ]