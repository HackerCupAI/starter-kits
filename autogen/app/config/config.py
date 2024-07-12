import os


DEFAULT_API_TYPE = "openai"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TIMEOUT = 120
OIA_API_TYPE = os.environ["OAI_API_TYPE"] if "OAI_API_TYPE" in os.environ else DEFAULT_API_TYPE

VISION_OAI_API_TYPE = os.environ["VISION_OAI_API_TYPE"] if "VISION_OAI_API_TYPE" in os.environ else DEFAULT_API_TYPE

WORKING_DIR = "/home/autogen/autogen/app"


def check_required_vars(API_TYPE):
    if API_TYPE == "openai":
        required_env_vars = [ "OAI_API_KEY",  "OAI_ORGANIZATION"]
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"{var} environment variable is not set.")
    else:
        required_env_vars = [ "OAI_API_KEY",  "OAI_BASE_URL"]
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"{var} environment variable is not set.")
            
def setup_llm(API_TYPE):
    model = os.environ["OAI_MODEL"] if "OAI_MODEL" in os.environ  else "gpt-4o"
    api_key = os.environ["OAI_API_KEY"] if "OAI_API_KEY" in os.environ else ""
    org = os.environ["OAI_ORGANIZATION"]  if "OAI_ORGANIZATION" in os.environ else None
    base_url = os.environ["OAI_BASE_URL"] if "OAI_BASE_URL" in os.environ else None
    api_version = os.environ["OAI_API_VERSION"] if "OAI_API_VERSION" in os.environ else None

    model_config = {"model": model,"api_key": api_key, "organization": org, "base_url": base_url}
    if API_TYPE == "azure":
        model_config[ "api_type"] = API_TYPE
        model_config["api_version"] = api_version
    return model_config

def setup_vision_llm(API_TYPE):
   
    vision_base_url = os.environ["VISION_OAI_BASE_URL"] if "VISION_OAI_BASE_URL" in os.environ else None
    vision_api_version = os.environ["VISION_OAI_API_VERSION"] if "VISION_OAI_API_VERSION" in os.environ else None

    vision_api_key = os.environ["VISION_OAI_API_KEY"]  if "VISION_OAI_API_KEY" in os.environ else None
    vision_model = os.environ["VISION_OAI_MODEL"] if "VISION_OAI_MODEL" in os.environ else "gpt-4o"
    vision_org = os.environ["VISION_OAI_ORGANIZATION"] if  "VISION_OAI_ORGANIZATION" in os.environ else None

    model_config = {"model": vision_model,"api_key": vision_api_key, "organization": vision_org, "base_url": vision_base_url}
    if API_TYPE == "azure":
        model_config[ "api_type"] = API_TYPE
        model_config["api_version"] = vision_api_version
    return model_config




check_required_vars(OIA_API_TYPE)

model_config = setup_llm(OIA_API_TYPE)

vision_model_config = setup_vision_llm(VISION_OAI_API_TYPE)




VISION_CONFIG = {"config_list": [vision_model_config],"temperature": 0.5,  "max_tokens": DEFAULT_MAX_TOKENS, "timeout": DEFAULT_TIMEOUT}

GPT4_CONFIG ={"config_list": [model_config], "timeout": DEFAULT_TIMEOUT, "temperature": 0.8}

print(f"GPT4_CONFIG: {GPT4_CONFIG}")

