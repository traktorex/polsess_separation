"""Tiny attribute-access dict used to feed JSON hyperparameters into the model.

Lifted verbatim from upstream `env.py`. The upstream file also has a
`build_env` helper that copies the config alongside the checkpoint at
training time; we don't need it here (inference only).
"""


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
