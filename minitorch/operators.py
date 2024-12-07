"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply two numbers and return the product"""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers and return their sum"""
    return x + y


def neg(x: float) -> float:
    """Negates a number, f(x) = -x"""
    return -x


def lt(x: float, y: float) -> float:
    """Returns 1.0 if x is strictly less than y, else return 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Returns 1.0 if x equals y, else return 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Takes two numbers and return the larger number"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Returns 1.0 if two numbers are close in value, else return 0.0"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Computes the sigmoid function of x

    Args:
    ----
        x (float): input

    Returns:
    -------
        float: function f(x) where
        f(x) = 1.0 / (1.0 + e^(-x)) if x >= 0
        f(x) = e^x / (1.0 + e^x) if x < 0

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Computes the rectified linear unit (ReLU) activation function of x

    Args:
    ----
        x (float): input

    Returns:
    -------
        float: f(x) = 0 if x <= 0, f(x) = x if x > 0

    """
    return x if x > 0 else 0.0


EPS = 1e-12


def log(x: float) -> float:
    """Returns the natural logarithm of x"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Returns the exponential function of x: f(x) = e^x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the reciprocal of x"""
    return 1.0 / x


def log_back(x: float, k: float) -> float:
    """Computes the derivative of f(x) where f(x) = k * log(x)"""
    return k / (x + EPS)


def inv_back(x: float, k: float) -> float:
    """Computes the derivative of f(x) where f(x) = k / x"""
    # -(1.0 / x**2) * k
    return -k / (abs(x) + EPS) ** 2


def relu_back(x: float, k: float) -> float:
    """Computes the derivative of f(x) where f(x) = k * relu(x)"""
    return k if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher order function that applies a given function(fn) to each element of an iterable(ls)"""

    def new_fn(ls: Iterable[float]) -> Iterable[float]:
        new_ls = [fn(ele) for ele in ls]
        return new_ls

    return new_fn


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function"""

    def new_fn(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        new_list = [fn(a, b) for (a, b) in zip(ls1, ls2)]
        return new_list

    return new_fn


def reduce(
    fn: Callable[[float, float], float], default_value: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function"""

    def new_fn(ls: Iterable[float]) -> float:
        result = default_value
        for x in ls:
            result = fn(result, x)
        return result

    return new_fn


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Return a new iterable with all elements negated."""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Return a new iterable with corresponding elements from two lists added together."""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Return the sum of all values in a list."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Return the product of all values in a list."""
    return reduce(mul, 1.0)(ls)
