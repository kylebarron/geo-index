from enum import Enum, auto


class StrEnum(str, Enum):
    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, (str, auto)):
            raise TypeError(
                f"Values of StrEnums must be strings: {value!r} is a {type(value)}"
            )
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self):
        return str(self.value)

    def _generate_next_value_(name, *_):
        return name.lower()


class RTreeMethod(StrEnum):
    Hilbert = auto()
    """Use hilbert curves for sorting the RTree
    """

    STR = auto()
    """Use the Sort-Tile-Recursive algorithm for sorting the RTree
    """
