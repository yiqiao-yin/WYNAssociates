# Docker Command

Build a docker image

```
docker build . -t postcard:latest
docker run --rm -it --network=host postcard:latest
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'
```

One can also build it and enter it using bash

```
docker run --rm -it --network=host --entrypoint=bash postcard:latest
ls
cat.py
```
