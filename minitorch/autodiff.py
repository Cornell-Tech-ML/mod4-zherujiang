from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Task 1.1.
    up_vals = [val for val in vals]
    up_vals[arg] += epsilon
    low_vals = [val for val in vals]
    low_vals[arg] -= epsilon

    f_plus = f(*up_vals)
    f_minus = f(*low_vals)
    slope = (f_plus - f_minus) / (2 * epsilon)

    return slope


variable_count = 1


class Variable(Protocol):
    """A variable in the computational graph.
    This protocol defines the interface for a variable in the autodiff system.
    Variables can accumulate gradients, have unique identifiers, and participate
    in the computation graph structure.
    """

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the final output with respect to this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if this variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Returns True if this variable is a constant (i.e., its value doesn't change)."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns an iterable of this variable's parent nodes in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients with respect to this variable's parents."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """

    # Task 1.4.
    def dfs(node: Variable) -> None:
        if node.unique_id in visited or node.is_constant():
            return
        visited.add(node.unique_id)
        for parent in node.parents:
            dfs(parent)
        order.append(node)

    visited = set()
    order = []
    dfs(variable)
    return order[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # Task 1.4.
    order = topological_sort(variable)
    derivatives = dict()
    derivatives[variable.unique_id] = deriv

    for node in order:
        local_deriv = derivatives[node.unique_id]

        if node.is_leaf():
            node.accumulate_derivative(local_deriv)
        else:
            parent_derivs = node.chain_rule(local_deriv)
            for parent, p_deriv in parent_derivs:
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += p_deriv
                else:
                    derivatives[parent.unique_id] = p_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the values stored for backward computation.

        This property provides access to the values that were saved during the forward pass
        using the `save_for_backward` method. These saved values are typically used in the
        backward pass to compute gradients.
        """
        return self.saved_values
