import matplotlib.pyplot as plt
import numpy as np

from detection.utils.visualization import plot_box


# if yaw change greater than 70 degrees?
# still linear interpolation but need to find intersect of both yaws afterwards and add that as a intermediate frame
def line_func(x_axis, start_x, start_y, yaw):
    m = np.tan(yaw)
    c = start_y - m * start_x
    return m * x_axis + c, -m, -c


def intersect_point(m1, m2, c1, c2):
    """using cramer's rule to return x, y"""
    return (c2 - c1) / (m1 - m2), ((c1 * m2) - (c2 * m1)) / (m1 - m2)


# if yaw change less than 70 degrees
# call interp
def get_yaw(x1, y1, x2, y2):
    m = (y1 - y2) / (x1 - x2)
    return np.arctan(m)


if __name__ == "__main__":
    # tensor(399.9770) tensor(119.1213) tensor(14.0265) tensor(4.9688) tensor(1.4291)
    # tensor(164.1316) tensor(355.1674) tensor(16.0632) tensor(7.0019) tensor(0.4412)
    figsize = (12, 8)
    dpi = 200
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plot_box(
        ax,
        399.9770,
        119.1213,
        1.4291,
        14.0265,
        4.9688,
        (0.0, 1.0, 0.0, 1.0),
        "Ground Truth",
    )

    plot_box(
        ax,
        164.1316,
        355.1674,
        0.4412,
        16.0632,
        7.0019,
        (1.0, 0.0, 0.0, 1.0),
        "Ground Truth",
    )

    x_axis = np.arange(150, 500)
    y_axis_1, m1, c1 = line_func(x_axis, 399.9770, 119.1213, 1.4291)
    y_axis_2, m2, c2 = line_func(x_axis, 164.1316, 355.1674, 0.4412)
    int_x, int_y = intersect_point(m1, m2, c1, c2)
    print(int_x, int_y)
    plt.scatter(int_x, int_y, c="black")
    plt.plot(x_axis, y_axis_1, c="r", linewidth=0.2)
    plt.plot(x_axis, y_axis_2, c="g", linewidth=0.2)
    plt.clf()

    xp = np.asarray([1, 3])
    fp = np.asarray([4, 2])

    interp_x = 2
    interp_y = np.interp(interp_x, xp, fp)
    interp_yaw = get_yaw(interp_x, interp_y, 1, 4)
    print(interp_yaw)
    print(-np.pi / 4)

    plt.scatter(interp_x, interp_y)
    plot_box(
        ax,
        interp_x,
        interp_y,
        interp_yaw,
        1,
        1,
        (1.0, 0.0, 0.0, 1.0),
        "Ground Truth",
    )
    plt.scatter(1, 4)
    plt.scatter(3, 2)
    # plt.show()

    lst = np.linspace(0, 1, 3)
    print(lst)
    lst2 = np.arange(26, 29, 1)
    lst = [1, 2, 3, 4]
    print(lst[1:-1])
