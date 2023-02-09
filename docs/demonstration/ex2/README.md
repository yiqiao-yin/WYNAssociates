

```cmd
curl https://api.synthesia.io/v2/videos  \
  -H "Authorization: ${1617249c979543caaef58e3c726b7ff3}" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{ "test": true, "input": [{ "scriptText": "Hello, World! This is my first synthetic video, made with the Synthesia API!", "avatar": "anna_costume1_cameraA", "background": "green_screen"}] }'
```