from model.Activations import SwiGLU, GeGLU, ReLU, ReGLU, LeakyReLU, ELU, GELU

def getActivation(config):
    activation_fn = config["activation_fn"].lower()  
    if activation_fn == "swiglu":
        return SwiGLU()
    elif activation_fn == "geglu":
        return GeGLU()
    elif activation_fn == "relu":
        return ReLU()
    elif activation_fn == "reglu":
        return ReGLU()
    elif activation_fn == "leakyrelu":
        return LeakyReLU()
    elif activation_fn == "elu":
        return ELU()
    elif activation_fn == "gelu":
        return GELU()
    else:
        raise ValueError(f"Unknown activation function: {config['activation_fn']}")

