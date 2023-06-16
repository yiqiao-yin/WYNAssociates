import requests

url = "https://api.synthesia.io/v2/videos"

headers = {
    "accept": "application/json",
    "authorization": "1617249c979543caaef58e3c726b7ff3",
    "test": "true",
    "scriptText": "Hello, World! This is my first video by Yin!"
}

response = requests.post(url, headers=headers)

print(response.text)