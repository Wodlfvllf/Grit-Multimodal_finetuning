
# ============================================================================
# GRIT Model Wrapper
# ============================================================================

def replace_linear_with_grit(root: nn.Module, qualified_name: str, cfg) -> LinearWithGRIT:
    """Replace a linear module with GRIT wrapper"""
    parts = qualified_name.split(".")
    parent = root
    
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise AttributeError(f"Parent missing attribute '{p}' while replacing {qualified_name}")
        parent = getattr(parent, p)
    
    last = parts[-1]
    if not hasattr(parent, last):
        raise AttributeError(f"Parent missing attribute '{last}' while replacing {qualified_name}")
    
    orig = getattr(parent, last)
    if not isinstance(orig, nn.Linear):
        raise TypeError(f"Expected nn.Linear at {qualified_name}, got {type(orig)}")
    
    wrapper = LinearWithGRIT(orig, cfg)
    setattr(parent, last, wrapper)
    return wrapper