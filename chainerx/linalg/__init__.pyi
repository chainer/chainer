import typing as tp


def inv(a: ndarray) -> ndarray: ...


def solve(a: ndarray, b: ndarray) -> ndarray: ...


def qr(a: ndarray) -> tp.Union[tp.Tuple[ndarray, ndarray], ndarray]: ...
