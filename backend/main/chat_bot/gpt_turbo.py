import asyncio
from g4f.client import Client

async def main():
    client = Client()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "привет, когда родился Трамп ? Ответьте на русском языке"}]
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