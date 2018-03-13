from rest_framework import serializers
from .models import Sentiments

class SentimentsSerializer(serializers.ModelSerializer):

    class Meta:
        model = Sentiments
        fields = ('sentiment')

