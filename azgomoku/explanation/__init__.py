"""State-specific AlphaZero decision explanation artifacts."""

__all__ = ["explain_decision"]


def explain_decision(*args, **kwargs):
    """Lazily import the public API so ``python -m`` remains warning-free."""
    from .explanation_export import explain_decision as implementation
    return implementation(*args, **kwargs)
