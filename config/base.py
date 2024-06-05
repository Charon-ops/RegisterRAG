from abc import ABC


class Config(ABC):
    def __init__(self) -> None:
        raise NotImplementedError(
            "Can not instantiate Config class, you should use a subclass instead"
        )
