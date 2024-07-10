import os

required_env_vars = [ "GPT4_OAI_API",  "GPT4_ORGANIZATION"]
for var in required_env_vars:
    if var not in os.environ:
        raise ValueError(f"{var} environment variable is not set.")


GPT_MODEL = os.environ["GPT_MODEL"] if "GPT_MODEL" in os.environ  else "gpt-4o"
GPT4_OAI_API = os.environ["GPT4_OAI_API"]
GPT4_ORGANIZATION = os.environ["GPT4_ORGANIZATION"] 
GPT4_MAX_TOKENS = 1000
GPT4_TIMEOUT = 120

VISION_OAI_API = os.environ["VISION_OAI_API"]  if "VISION_OAI_API" in os.environ else GPT4_OAI_API
VISION_MODEL = os.environ["VISION_MODEL"] if "VISION_MODEL" in os.environ else "gpt-4o"
VISION_ORGANIZATION = os.environ["VISION_ORGANIZATION"] if  "VISION_ORGANIZATION" in os.environ else GPT4_ORGANIZATION
VISION_MAX_TOKENS = 1000

VISION_CONFIG = {"config_list": [{"model": VISION_MODEL, "organization": VISION_ORGANIZATION,  "api_key": VISION_OAI_API, }],"temperature": 0.5,  "max_tokens": VISION_MAX_TOKENS, "timeout": GPT4_TIMEOUT}

GPT4_CONFIG ={"config_list": [{"model": GPT_MODEL,"api_key": GPT4_OAI_API, "organization": GPT4_ORGANIZATION}], "timeout": GPT4_TIMEOUT}

