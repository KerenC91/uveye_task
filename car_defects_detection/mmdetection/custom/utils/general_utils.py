# ------------------------------------------------------------
# Utility to fetch model config
# ------------------------------------------------------------
def get_model_config(model_key: str, model_zoo):
    if model_key not in model_key:
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {list(model_zoo.keys())}"
        )
    cfg = model_zoo[model_key]
    return cfg["model_name"], cfg["checkpoint"]