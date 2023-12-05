def in_notebook():
    try:
        from IPython.core.getipython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # type: ignore
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
