import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

response = client.chat.completions.create(
    model=os.getenv("CHAT_MODEL"),
    messages=[{"role": "user", "content": "你好，回复我一句话即可"}],
)

print(response.choices[0].message.content)
