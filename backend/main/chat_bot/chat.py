# Импорт необходимых библиотек
import random
import json
import numpy as np
import torch
import asyncio
import torch.nn as nn
from g4f.client import Client
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import word_tokenize

import os

currently_path = os.path.dirname(os.path.abspath(__file__))
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out



# Инициализация морфологического анализатора
morph = MorphAnalyzer()

# Определение устройства для работы с нейронной сетью (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Функция для создания мешка слов
def bag_of_words(tokenized_sentence, words):
    # Приведение каждого слова к нормальной форме
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Инициализация мешка с нулями для каждого слова
    bag = np.zeros(len(words), dtype=np.float32)
    # Установка единицы в мешке для каждого слова, присутствующего в предложении
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

# Функция для токенизации текста
def tokenize(text):
    # Токенизация текста на русском языке
    tokens = word_tokenize(text, language='russian')
    return tokens

# Функция для приведения слова к нормальной форме
def stem(word):
    # Приведение слова к нормальной форме с помощью морфологического анализатора
    return morph.parse(word)[0].normal_form

# Загрузка данных из файла intentss.json
with open(currently_path+'\intentss.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Загрузка данных из файла data.pth
FILE = currently_path+"\data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

# Определение размеров входных, скрытых и выходных слоев нейронной сети
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

# Определение списков всех слов и тегов
all_words = data['all_words']
tags = data['tags']

# Загрузка состояния модели из файла data.pth
model_state = data["model_state"]

# Инициализация нейронной сети
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Загрузка состояния модели в нейронную сеть
model.load_state_dict(model_state)

# Перевод модели в режим оценки
model.eval()

# Определение имени бота
bot_name = "Sam"

# Функция для получения ответа от бота
def get_response_model(msg):
    # Токенизация входного сообщения
    sentence = tokenize(msg)

    # Создание мешка слов для входного сообщения
    X = bag_of_words(sentence, all_words)

    # Преобразование мешка слов в тензор
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Получение выхода нейронной сети
    output = model(X)

    # Определение индекса максимального значения в выходе
    _, predicted = torch.max(output, dim=1)

    # Определение тега, соответствующего индексу
    tag = tags[predicted.item()]

    # Получение вероятности предсказанного тега
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    otvet = 'Нет связи с моделью'
    # Если вероятность предсказанного тега больше 0.75, возвращаем ответ из списка ответов
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    # Иначе вызываем функцию main для генерации ответа с помощью модели GPT-3.5 Turbo
    else:
        client = Client()
        try:
            response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": msg + " Ответьте на русском языке"}]
            )
            if response.choices and response.choices[0].message.content:
                otvet = (response.choices[0].message.content)
                print('---------------------')
                print(otvet)
            else:
                print("Нет ответа от модели")
                return("Нет ответа от модели")
        except Exception as e:
            print(f"Ошибка: {e}")
    return(otvet)

#функция для передачи сообщений 
def innpute_view(inpute):
    resp = get_response_model(inpute)
    print (resp)
    return resp


# if __name__ == "__main__":
#         # sentence = "do you use credit cards?"
#     sentence = input("Вы: ")

#     resp = get_response_model(sentence)
#     print('main',resp)

