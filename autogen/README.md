# Hacker Rank Code Sample
Recommended tools:
- Docker

## Building and Running Docker Image


Add a file named `.env` to root with at a minimum the following content:
```bash
GPT4_OAI_API=<oai_api_key> 
GPT4_ORGANIZATION=<oai_org>
GPT_MODEL=<oai_model>  # *optional* (default gpt-4o) 
VISION_MODEL=<oai_model> #*optional* (default gpt-4o) 
VISION_ORGANIZATION=<oai_api_key> # *optional* (default GPT4_ORGANIZATION)
VISION_OAI_API=<oai_org> # *optional* (default GPT4_OAI_API) 
```

This will throw an error without required envs. Check [Config](./app/config/config.py) for more configuration options.

In the root folder run the following to build
```bash
docker build -f ./Dockerfile -t autogen_dev_img .
```

After image is built, run the docker image with one of the following (replace `<file>.py` with target file):

```bash 
docker run --env-file ./.env  -it -v $(pwd)/app:/home/autogen/autogen/app autogen_dev_img:latest python /home/autogen/autogen/app/<file>.py
```


- The simple example using multiple agents in a coding scenario:
`main-simple.py`

- For complex example using multiple agents to solve [DimSum HackerRank practice problem](https://www.facebook.com/codingcompetitions/hacker-cup/2023/practice-round/problems/B):
`main-hackercup.py`

More info on Docker installation with autogen:  https://microsoft.github.io/autogen/docs/installation/Docker/
