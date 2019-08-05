import typing as tp


def inv(a: ndarray) -> ndarray: ...


def pinv(a: ndarray, rcond: float=...) -> ndarray: ...


def solve(a: ndarray, b: ndarray) -> ndarray: ...


def svd(a: ndarray,
        full_matrices: bool=...,
        compute_uv: bool=...) -> tp.Union[tp.Tuple[ndarray, ndarray, ndarray], ndarray]: ...
