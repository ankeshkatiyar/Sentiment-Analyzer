# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import  rest_framework
from  .models import  Sentiments
from .serializers import SentimentsSerializer
from rest_framework.views import APIView
from  rest_framework.response import Response
from django.http import  HttpResponse
from django.http import  HttpRequest
from django.shortcuts import render
from SentimentAnalyzerMLLogic import eval
import  json
from django.http import  JsonResponse



class sentiMentView(APIView):
    def get(self , request):
        # sentiments1 = Sentiments.objects.all()
        # request.
        # serializer = SentimentsSerializer(sentiments1 , many=True)
        test = request.GET['comment']
        return JsonResponse(eval.doEval(str(test)) , safe=False)

    def post(self,request):
       test = request.POST.getlist('array1[]',False)
       #return Response(test)
       #print(test)
       output =  eval.doEval(test)
       return JsonResponse(output, safe=False)



