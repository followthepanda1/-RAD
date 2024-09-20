from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .chat_bot import chat
# Create your views here.
def main(request):

    return render(request, "../public_html/index.html", {'AAA':'aaa'})

class Chat(APIView):

    def post(self, request):
        data = request.data['zapros']
        otvet = chat.get_response_model(data)
        print('me', otvet)
        return Response({'title': otvet})

