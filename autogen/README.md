# Hacker Rank Code Sample
Recommended tools:
- Docker

## Building and Running Docker Image
Optionally configure env vars in dockerfile or in a .env file
env vars include:
- GPT4_OAI_API 
- GPT4_ORGANIZATION
- GPT_MODEL *optional* (default gpt-4o) 
- VISION_MODEL *optional* (default gpt-4o) 
- VISION_ORGANIZATION *optional* (default GPT4_ORGANIZATION)
- VISION_OAI_API *optional* (default GPT4_OAI_API)

This will throw an error without required envs. Check [Config](./app/config/config.py) for more configuration options.

In the root folder run the following to build
`docker build -f ./Dockerfile -t autogen_dev_img .`

After image is built, run the docker image with one of the following (replace `main-<file>.py` with target file):
- With .env file:`docker run --env-file ./.env  -it -v $(pwd)/app:/home/autogen/autogen/app autogen_dev_img:latest python /home/autogen/autogen/app/main-<file>.py`

- With dockerfile ENV: `docker run -it -v $(pwd)/app:/home/autogen/autogen/app autogen_dev_img:latest python /home/autogen/autogen/app/main-<file>.py`

The simple example case using agents in a coding scenario:
`main-simple.py`

For complex example using agents to solve HackerRank problem:
`main-hackercup.py`

More info on Docker installation with autogen:  https://microsoft.github.io/autogen/docs/installation/Docker/
