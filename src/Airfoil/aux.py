from scipy.special import comb
import numpy as np


# def bpoly(n, t, i):
#     """ Polinômios de Bernstein  """
#     return comb(n, i) * (t**i) * (1 - t)**(n-i)


def eng_string(
    x: float,
    unit: str = "",
    format="%.3g",
    si=True,
    add_space_after_number: bool = None,
) -> str:
    """
    Taken from: https://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6/40691220

    Returns float/int value <x> formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.

    Args:

        x: The value to be formatted. Float or int.

        unit: A unit of the quantity to be expressed, given as a string. Example: Newtons -> "N"

        format: A printf-style string used to format the value before the exponent.

        si: if true, use SI suffix for exponent. (k instead of e3, n instead of
            e-9, etc.)

    Examples:

    With format='%.2f':
        1.23e-08 -> 12.30e-9
             123 -> 123.00
          1230.0 -> 1.23e3
      -1230000.0 -> -1.23e6

    With si=True:
          1230.0 -> "1.23k"
      -1230000.0 -> "-1.23M"

    With unit="N" and si=True:
          1230.0 -> "1.23 kN"
      -1230000.0 -> "-1.23 MN"
    """

    sign = ""
    if x < 0:
        x = -x
        sign = "-"
    elif x == 0:
        return format % 0
    elif np.isnan(x):
        return "NaN"

    exp = int(np.floor(np.log10(x)))
    exp3 = exp - (exp % 3)
    x3 = x / (10**exp3)

    if si and exp3 >= -24 and exp3 <= 24:
        if exp3 == 0:
            suffix = ""
        else:
            suffix = "yzafpnμm kMGTPEZY"[(exp3 + 24) // 3]

        if add_space_after_number is None:
            add_space_after_number = unit != ""

        if add_space_after_number:
            suffix = " " + suffix + unit
        else:
            suffix = suffix + unit

    else:
        suffix = f"e{exp3}"

        if add_space_after_number:
            add_space_after_number = unit != ""

        if add_space_after_number:
            suffix = suffix + " " + unit
        else:
            suffix = suffix + unit

    return f"{sign}{format % x3}{suffix}"


def bmatrix(T, degree):
    """Bernstein matrix for Bézier curves"""
    return np.matrix(
        [[bernstein_poly(i, degree, t) for i in range(degree + 1)] for t in T]
    )


def least_square_fit(points, M):
    M_ = np.linalg.pinv(M)
    return M_ * points


def bernstein_poly(i, n, t):
    """Polinômio de Bernstein"""
    return comb(n, i) * (t**i) * (1 - t) ** (n - i)


def generate_bezier_curve(points, nTimes=80):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [ [1,1],
              [2,3],
              [4,5], ..[Xn, Yn] ]
     nTimes is the number of time steps, defaults to 1000

     See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)]
    )

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def save_as_dat_from_bezier(
    bezier_upper: list, bezier_lower: list, name="generated_airfoil", header="Airfoil"
):
    """Salva o perfil de bezier como um arquivo .dat"""
    X_bezier_upper, Y_bezier_upper = generate_bezier_curve(bezier_upper)
    X_bezier_lower, Y_bezier_lower = generate_bezier_curve(bezier_lower)

    data_upper = np.array(
        [
            np.around(X_bezier_upper, 6).astype("str"),
            np.around(Y_bezier_upper, 6).astype("str"),
        ]
    ).transpose()
    data_lower = np.array(
        [
            np.around(X_bezier_lower, 6).astype("str"),
            np.around(Y_bezier_lower, 6).astype("str"),
        ]
    ).transpose()

    if ".dat" not in name:
        name += ".dat"

    data = np.concatenate((data_upper, data_lower))

    np.savetxt(f"airfoils/{name}", data, header=header, comments="", fmt="%s")
