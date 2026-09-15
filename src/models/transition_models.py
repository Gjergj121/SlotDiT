from models.attention import TransformerBlock


def get_transition_module(model_name, **kwargs):
    if model_name != "TransformerBlock":
        raise ValueError(f"Unsupported transition: {model_name}")
    slot_dim = kwargs.pop("slot_dim")
    kwargs.pop("hidden_dim", None)
    kwargs.pop("residual", None)
    return TransformerBlock(embed_dim=slot_dim, **kwargs)
