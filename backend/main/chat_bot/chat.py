# Импорт необходимых библиотек
import random
import json
import numpy as np
import torch
import asyncio
from g4f.client import Client
from model import NeuralNet
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import word_tokenize

# Инициализация морфологического анализатора
morph = MorphAnalyzer()

# Определение устройства для работы с нейронной сетью (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
with open('backend\main\chat_bot\intentss.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Загрузка данных из файла data.pth
FILE = "backend\main\chat_bot\data.pth"
data = torch.load(FILE)

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
def get_response(msg):
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

    # Если вероятность предсказанного тега больше 0.75, возвращаем ответ из списка ответов
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    # Иначе вызываем функцию main для генерации ответа с помощью модели GPT-3.5 Turbo
    else:
        async def main():
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
                    print('---------------------')
                else:
                    print("Нет ответа от модели")
            except Exception as e:
                print(f"Ошибка: {e}")
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main(), debug=True)


if __name__ == "__main__":
    print("Ожидаю ваш вопрос! (введите 'quit' чтобы выйти)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("Вы: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

