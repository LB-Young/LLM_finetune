"""
run in google colab
envs:pip install openai
"""

# method 1
from openai import OpenAI, AzureOpenAI
client = AzureOpenAI(
    api_key="xxxxxxxxxx",  
    api_version="xxxxxxxxxx",
    azure_endpoint="xxxxxxxxxxxxx"
)

def request_api():
  response = client.chat.completions.create(
    model="gpt-35-turbo-16k",
    messages=[{"role": "user", "content": "what is python?"}],
  )
  return response.choices[0].message.content

res = request_api()
print(res)



# method 2
client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxx",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "what is python",
        }
    ],
    model="gpt-4",
)

print(chat_completion.choices[0].message.content)
