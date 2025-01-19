import numpy as np


class Himmelblau:
    """
    Parameters:
        position : A 2D vector
    Minimum:
        The Himmelblau function has four global minimum, all with a function value of 0.
        They are located at approximately:
            (3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)

    """
    @staticmethod
    def calculate(position):
        x, y = position[0], position[1]
        return np.power(np.power(x, 2) + y - 11, 2) + np.power(x + np.power(y, 2) - 7, 2)


class Rosenbrock:
    """
    Parameters:
        position: A 2D vector
    Minimum:
        The Rosenbrock function has a global minimum at (1, 1) with a function value of 0.
    """
    @staticmethod
    def calculate(position):
        x, y = position[0], position[1]
        return 100 * np.power((y - np.power(x, 2)), 2) + np.power((1 - x), 2)


class Easom:
    """
    Parameters:
        position: A 2D vector
    Minimum:
        The Easom function has a global minimum at (π, π) with a function value of -1.
    """
    @staticmethod
    def calculate(position):
        x, y = position[0], position[1]
        return -np.cos(x) * np.cos(y) * np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2)


class CrossInTray:
    """
    Parameters:
        position : A 2D vector
    Minimum:
       The Cross-in-Tray function has four global minima with a function value of approximately -2.06261.
       They are located at approximately: (+-1.34941, +-1.34941).
    """
    @staticmethod
    def calculate(position):
        x, y = position[0], position[1]
        return -0.0001 * np.power(abs(np.sin(x) * np.sin(y) * np.exp(abs(100 - np.sqrt(x ** 2 + y ** 2) / np.pi))) + 1,
                                  0.1)


class Booth:
    """
    Parameters:
        position : A 3D vector
    Minimum:
        The Booth function has a global minimum at (1, 3, 0) with a function value of 0.
    """
    @staticmethod
    def calculate(position):
        x, y, z = position[0], position[1], position[2]
        return np.power(x + 2 * y - 7, 2) + np.power(2 * x + y - 5, 2) + np.power(z, 2)


class GriffithStokes:
    """
    Parameters:
        position: A 3D vector
    Minimum:
        The GriffithStokes function is known to have global minimum close to (0, 0, 0).
        This function also has symmetrical minimums in pi positions.
    """
    @staticmethod
    def calculate(position):
        x, y, z = position[0], position[1], position[2]
        return np.sin(x + y) * np.exp((1 - np.cos(z))**2) + np.cos(y + z) * np.exp((1 - np.sin(x))**2) + (x - z)**2


class Sphere:
    """
     Parameters:
        position : A vector representing the coordinates of any dimension.
    Minimum:
        The Sphere function has a global minimum at the all coordinates are 0 with a function value of 0.
    """
    @staticmethod
    def calculate(position):
        return sum([x**2 for x in position])


class TestFunc:
    """
    Provides a static method for calculating function values.

    Input Parameters:
        func_class : The class of the function to be calculated (e.g., Himmelblau, Rosenbrock).
        position : A vector representing the coordinates of any dimension.

    Returns:
        float: The calculated function value.
    """
    @staticmethod
    def calculate(func_class, position):
        return func_class.calculate(position)
