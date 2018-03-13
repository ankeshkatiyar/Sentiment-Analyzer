# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Sentiments(models.Model):
    sentiment = models.CharField(max_length=10)


    def __str__(self):
        return self.sentiment
