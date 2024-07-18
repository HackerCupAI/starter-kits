# HackerCup Code Sample
Recommended tools:
- Docker

## Building and Running Docker Image


Add a file named `.env` to root with at a minimum the following content:
```bash
OAI_API_KEY=<oai_api_key> 
OAI_ORGANIZATION=<oai_org>
OAI_MODEL=<oai_model>  # *optional* (default gpt-4o) 
OAI_BASE_URL=<oia_url> # *optional* 
OAI_API_VERSION=<oia_version> # *optional*
OAI_API_TYPE=<oai_type> # *optional* (default "openai")

VISION_OAI_MODEL=<oai_model> #*optional* (default gpt-4o) 
VISION_OAI_ORGANIZATION=<oai_org> # *optional* 
VISION_OAI_API_KEY=<oai_api_key> # *optional* 
VISION_OAI_BASE_URL=<oia_url> #* optional
VISION_OAI_API_VERSION=<oia_version> # *optional*
VISION_OAI_API_TYPE=<oai_type> # *optional* (default "openai")
```
see [LLM Config](https://microsoft.github.io/autogen/docs/topics/llm_configuration/) for more info.

Running a py script  will throw an error without required envs. Check [Config](./app/config/config.py) for more configuration options.

In the root folder run the following to build
```bash
docker build -f ./Dockerfile -t autogen_dev_img .
```

After image is built, run the docker image with one of the following, replace `<file`>.py and `<path to data dir`>  :

```bash 
docker run --env-file ./.env  -it -v $(pwd)/app:/home/autogen/autogen/app autogen_dev_img:latest python /home/autogen/autogen/app/<file>.py <path to data dir> 
```
For example,  to solve [2023 DimSum Delivery HackerCank practice problem](https://www.facebook.com/codingcompetitions/hacker-cup/2023/practice-round/problems/B):

```bash 
docker run --env-file ./.env  -it -v $(pwd)/app:/home/autogen/autogen/app autogen_dev_img:latest python /home/autogen/autogen/app/hackercup.py /home/autogen/autogen/app/assets/nim_sum_dim_sum
```

- The simple example using multiple agents in a coding scenario, which is a good place to start with AutoGen and coding agents:
`simple_agent.py`

- A more complex agent is defined in `groupchat_agents.py`

More info on Docker installation with autogen:  https://microsoft.github.io/autogen/docs/installation/Docker/
