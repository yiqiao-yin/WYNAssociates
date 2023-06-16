#!/bin/bash

API_KEY="xxx"

URL="https://api.synthesia.io/v2/videos"

curl -H "$URL"  \
  -H "Authorization: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{ "test": true, "input": [{ "scriptText": "Hello, World! This is my first synthetic video, made with the Synthesia API!", "avatar": "anna_costume1_cameraA", "background": "green_screen"}] }'