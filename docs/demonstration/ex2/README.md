

## Create video

You can use this endpoint to bespoke video content. In case there is a feature that STUDIO provides that this API doesn't support, we recommend you use the video from template endpoint.

Use shell:

```cmd
curl https://api.synthesia.io/v2/videos  \
  -H "Authorization: ${XXX}" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{ "test": true, "input": [{ "scriptText": "Hello, World! This is my first synthetic video, made with the Synthesia API!", "avatar": "anna_costume1_cameraA", "background": "green_screen"}] }'
```

Or, alternatively, use python:

```py
import requests

url = "https://api.synthesia.io/v2/videos"

headers = {
    "accept": "application/json",
    "Authorization": 1617249c979543caaef58e3c726b7ff3,
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers)

print(response.text)
```