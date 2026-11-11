# Local stub: `layer_generator` is imported by the training scripts but is not
# included in the repository and `generate` is never actually called.
def generate(*args, **kwargs):
    raise NotImplementedError("layer_generator is not part of this repository")
