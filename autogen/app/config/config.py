import os


DEFAULT_API_TYPE = "openai"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TIMEOUT = 200
OIA_API_TYPE = os.environ["OAI_API_TYPE"] if "OAI_API_TYPE" in os.environ else DEFAULT_API_TYPE
VISION_OAI_API_TYPE = os.environ["VISION_OAI_API_TYPE"] if "VISION_OAI_API_TYPE" in os.environ else DEFAULT_API_TYPE
VISION_API_ENABLED = "VISION_OAI_API_KEY" in os.environ

WORKING_DIR = "/home/autogen/autogen/app"
BASE_LOGS_DIR = f"{WORKING_DIR}/logs"


def check_required_vars(API_TYPE):
    required_env_vars = ["OAI_API_KEY"]

    # For Azure API, we also need to set the following environment variables
    if API_TYPE == "azure":
        required_env_vars += ["OAI_BASE_URL"]

    missing_vars = [var for var in required_env_vars if var not in os.environ]
    if len(missing_vars) > 0:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


def setup_llm(API_TYPE):
    api_key = os.environ["OAI_API_KEY"]
    model = os.environ.get("OAI_MODEL", "gpt-4o")
    org = os.environ.get("OAI_ORGANIZATION", None)
    base_url = os.environ.get("OAI_BASE_URL", None)
    api_version = os.environ.get("OAI_API_VERSION", None)
    model_config = {"model": model, "api_key": api_key, "organization": org, "base_url": base_url}
    if API_TYPE == "azure":
        model_config["api_type"] = API_TYPE
        model_config["api_version"] = api_version
    return model_config


def setup_vision_llm(API_TYPE):
    model_config = {}
    if VISION_API_ENABLED:
        vision_api_key = os.environ["VISION_OAI_API_KEY"]
        vision_model = os.environ.get("VISION_OAI_MODEL", "gpt-4o")
        vision_org = os.environ.get("VISION_OAI_ORGANIZATION", None)
        vision_base_url = os.environ.get("VISION_OAI_BASE_URL", None)
        vision_api_version = os.environ.get("VISION_OAI_API_VERSION", None)
        model_config = {"model": vision_model, "api_key": vision_api_key, "organization": vision_org, "base_url": vision_base_url}
        if API_TYPE == "azure":
            model_config["api_type"] = API_TYPE
            model_config["api_version"] = vision_api_version
    return model_config


check_required_vars(OIA_API_TYPE)

model_config = setup_llm(OIA_API_TYPE)
vision_model_config = setup_vision_llm(VISION_OAI_API_TYPE)

VISION_CONFIG = {"config_list": [vision_model_config],"temperature": 0.5, "max_tokens": DEFAULT_MAX_TOKENS, "timeout": DEFAULT_TIMEOUT, "cache_seed": 42}
GPT4_CONFIG = {"config_list": [model_config], "timeout": DEFAULT_TIMEOUT, "temperature": 0.5, "cache_seed": 42}
