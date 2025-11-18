# plots.py (compatible with current S2.py)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import S2


def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3, 1)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


class SphereMesh:
    def __init__(self, radius=1.0):
        self.r = radius
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_box_aspect([1, 1, 1])
        self.ax.grid(False)
        self.ax.set_axis_off()

    def draw_mesh(self, nu=40, nv=20, linewidth=0.6, color='orange'):
        u = np.linspace(0, 2*np.pi, nu)
        v = np.linspace(0, np.pi, nv)
        x = self.r * np.outer(np.cos(u), np.sin(v))
        y = self.r * np.outer(np.sin(u), np.sin(v))
        z = self.r * np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_wireframe(x, y, z, color=color, linewidth=linewidth)

    def draw_axes(self, length=None, linewidth=1.0):
        if length is None:
            length = self.r * 0.25
        origin = np.zeros(3)
        self.ax.quiver(*origin, length, 0, 0, color='r', linewidth=linewidth, arrow_length_ratio=0.05)
        self.ax.quiver(*origin, 0, length, 0, color='g', linewidth=linewidth, arrow_length_ratio=0.05)
        self.ax.quiver(*origin, 0, 0, length, color='b', linewidth=linewidth, arrow_length_ratio=0.05)
        offset = 0.1 * length
        self.ax.text(length + offset, 0, 0, "X", color='r')
        self.ax.text(0, length + offset, 0, "Y", color='g')
        self.ax.text(0, 0, length + offset, "Z", color='b')

    def plot_arc(self, x, y, u, steps=100, color='gray', linewidth=2):
        x = _unit(x)
        y = _unit(y)
        u = np.asarray(u, dtype=float).reshape(2, 1)

        # total geodesic angle between x and y
        angle = float(np.arccos(np.clip(float(x.T @ y), -1.0, 1.0)))

        # Sample along the arc using scaled tangent coords
        t = np.linspace(0.0, 1.0, steps)
        pts = np.zeros((steps, 3))
        for i, tau in enumerate(t):
            ui = u * (tau * angle)  # scale tangent coords linearly
            pi = S2.Exp(x, ui)      # 3x1
            pts[i, :] = pi.flatten()

        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=linewidth)

    def plot_vector(self, a, b=(0, 0, 0), color='c', linewidth=1.):
        a = np.asarray(a, dtype=float).reshape(3,)
        b = np.asarray(b, dtype=float).reshape(3,)
        self.ax.scatter(*a, color='orange', s=40)
        self.ax.quiver(*b, *a, color=color, linewidth=linewidth, arrow_length_ratio=0.05)

    def plot_exp(self, x, u, color_base='k', color_tan='m', color_exp='c'):
        x = _unit(x)
        u = np.asarray(u, dtype=float).reshape(2, 1)

        Bx = S2.B(x)                        # 3x2
        Px = np.eye(3) - x @ x.T            # projector to T_x S^2
        tan_vec = (Px @ (Bx @ u)).flatten() # 3D tangent vector at x
        y = S2.Exp(x, u).flatten()          # endpoint

        # base vector and endpoint from origin
        self.ax.scatter(*x.flatten(), color=color_base, s=40)
        self.ax.quiver(0, 0, 0, *x.flatten(), color=color_base, linewidth=1.2, arrow_length_ratio=0.05)

        # tangent arrow at x
        self.ax.quiver(*x.flatten(), *tan_vec, color=color_tan, linewidth=1.5, arrow_length_ratio=0.05)

        self.ax.scatter(*y, color=color_exp, s=40)
        self.ax.quiver(0, 0, 0, *y, color=color_exp, linewidth=1.2, arrow_length_ratio=0.05)

        self.plot_arc(x, y.reshape(3, 1), u)

    def plot_log(self, x, y, color_base='k', color_log='m', color_target='c'):
        x = _unit(x)
        y = _unit(y)

        u = S2.Log(x, y).reshape(2, 1)      # 2D tangent coords at x
        Bx = S2.B(x)
        Px = np.eye(3) - x @ x.T
        tan_vec = (Px @ (Bx @ u)).flatten()

        self.plot_arc(x, y, u)

        # base point and target
        self.ax.scatter(*x.flatten(), color=color_base, s=40)
        self.ax.quiver(0, 0, 0, *x.flatten(), color=color_base, linewidth=1.2, arrow_length_ratio=0.05)

        self.ax.scatter(*y.flatten(), color=color_target, s=40)
        self.ax.quiver(0, 0, 0, *y.flatten(), color=color_target, linewidth=1.2, arrow_length_ratio=0.05)

        # log vector at x
        self.ax.quiver(*x.flatten(), *tan_vec, color=color_log, linewidth=1.6, arrow_length_ratio=0.05)

    def show(self):
        plt.tight_layout()
        plt.show()


# --- Simple helper using S2.Re for composition visualization (optional) ---
def compose_via_Re(x, y):
    """Return z = Re(y) Re(x) e_z (geodesic composition proxy)."""
    ez = np.array([[0., 0., 1.]]).T
    R_x = S2.Re(_unit(x))
    R_y = S2.Re(_unit(y))
    return (R_y @ R_x @ ez).reshape(3, 1)


# ----------------- Tests -----------------
def testExp():
    sp = SphereMesh(radius=1.0)
    sp.draw_mesh()
    sp.draw_axes()

    x = np.array([[1., 0., 0.]]).T
    x /= np.linalg.norm(x)

    u = np.array([[0., np.pi/2.]]).T  # 2x1
    
    sp.plot_exp(x, u)
    sp.show()


def testLog():
    sp = SphereMesh(radius=1.0)
    sp.draw_mesh()
    sp.draw_axes()

    x = np.array([[0., 0., 1.]]).T
    u = np.array([[np.pi/2, 0.0]]).T
    y = S2.Exp(x, u)

    sp.plot_log(x, y)
    sp.show()

    print("x: ", x.flatten())
    print("u: ", u.flatten())
    print("y: ", y.flatten())
    print("Log(x, Exp(x, u)): ", S2.Log(x, y).flatten())


def testCompose():
    sp = SphereMesh(radius=1.0)
    sp.draw_mesh()
    sp.draw_axes()

    ez = np.array([[0., 0., 1.]]).T
    x = np.array([[1., 0., 0.]]).T
    x /= np.linalg.norm(x)
    u = np.array([[np.sqrt(2)/2, np.sqrt(2)/2]]).T * (np.pi/4)

    # visualize Exp from ez and from x
    sp.plot_exp(ez, u)
    sp.plot_exp(x, u)

    # visualize a simple composition using Re
    z = compose_via_Re(S2.Exp(ez, u), x)
    sp.plot_vector(z)

    sp.show()


if __name__ == "__main__":
    testExp()
    # testLog()
    # testCompose()
