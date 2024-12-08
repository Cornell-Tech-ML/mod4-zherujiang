from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for a generic scalar function.

        This method should be implemented by subclasses to define the specific
        backward pass computation for each scalar function.
        """
        raise NotImplementedError("Not Implemented")

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values.

        This method handles the application of the scalar function to both Scalar
        objects and raw float values. It performs the following steps:
        1. Converts all inputs to Scalar objects if they aren't already.
        2. Creates a Context object for storing intermediate values.
        3. Calls the forward method with raw float values.
        4. Creates a new Scalar object with the result and its computation history.

        Args:
        ----
            *vals (ScalarLike): One or more values to apply the function to.
                These can be Scalar objects or raw float values.

        Returns:
        -------
            Scalar: A new Scalar object representing the result of the function application.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for natural log function."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for natural log function."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# Task 1.2.


class Mul(ScalarFunction):
    """Multiply function f(x, y) = x * y"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Forward pass for multiplication"""
        ctx.save_for_backward(x, y)
        return operators.mul(float(x), float(y))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication."""
        x, y = ctx.saved_values
        return (d_output * y, d_output * x)


class Inv(ScalarFunction):
    """Returns the reciprocal of x"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for reciprocal function.

        Computes 1/x and saves x for the backward pass.
        """
        ctx.save_for_backward(x)
        return operators.inv(float(x))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for reciprocal function.

        Computes the derivative of the output with respect to the input x.
        We use the inv_back method from the operators library to compute the derivative.
        """
        (x,) = ctx.saved_values
        return operators.inv_back(x, d_output)


class Neg(ScalarFunction):
    """Negates a number, f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for negation.Negates the input value x."""
        # No need to save any values for backward pass as negation is a simple operation
        return operators.neg(float(x))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation. The derivative of negation is always -1, so we simply negate the incoming gradient."""
        return -d_output


class Sigmoid(ScalarFunction):
    """f(x) = 1.0 / (1.0 + e^(-x)) if x >= 0, f(x) = e^x / (1.0 + e^x) if x < 0"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for sigmoid function.

        Computes the sigmoid of x: f(x) = 1 / (1 + e^(-x))
        This implementation uses the sigmoid method from the operators module.
        """
        out = operators.sigmoid(float(x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid function.

        Computes the derivative of the sigmoid function with respect to its input.
        The derivative of the sigmoid function is: f'(x) = f(x) * (1 - f(x))
        where f(x) is the sigmoid function.
        """
        (sigma,) = ctx.saved_values
        return d_output * sigma * (1 - sigma)


class ReLU(ScalarFunction):
    """Computes the rectified linear unit (ReLU) activation function of x"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for ReLU function."""
        ctx.save_for_backward(x)
        return operators.relu(float(x))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU function."""
        (x,) = ctx.saved_values
        return operators.relu_back(x, d_output)


class Exp(ScalarFunction):
    """Returns the exponential function of x: f(x) = e^x"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for exponential function.

        Computes e^x using the exp method from the operators module.
        Saves the input x for use in the backward pass.
        """
        out = operators.exp(float(x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential function.

        Computes the derivative of e^x with respect to x.
        The derivative of e^x is e^x itself.
        """
        (out,) = ctx.saved_values
        return d_output * out


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1 if x < y else 0"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Forward pass for less than function."""
        return operators.lt(float(x), float(y))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than function.

        The derivative of the less than function is always 0 with respect to both inputs,
        as it's a step function and not differentiable at the point of equality.
        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = 1 if x == y else 0"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Forward pass for equal function. Returns 1.0 if equal otherwise 0.0"""
        return operators.eq(float(x), float(y))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equal function.

        The derivative of the equal function is always 0 with respect to both inputs,
        as it's a step function and not differentiable at the point of equality.
        """
        return 0.0, 0.0
