import typing as tp


def eigh(a: ndarray, UPLO: str=...) -> tp.Tuple[ndarray, ndarray]: ...


def eigvalsh(a: ndarray, UPLO: str=...) -> ndarray: ...


def inv(a: ndarray) -> ndarray: ...


def pinv(a: ndarray, rcond: float=...) -> ndarray: ...


def solve(a: ndarray, b: ndarray) -> ndarray: ...


def svd(a: ndarray,
        full_matrices: bool=...,
        compute_uv: bool=...) -> tp.Union[tp.Tuple[ndarray, ndarray, ndarray], ndarray]: ...


def qr(a: ndarray) -> tp.Union[tp.Tuple[ndarray, ndarray], ndarray]: ...


def cholesky(a: ndarray) -> ndarray: ...
