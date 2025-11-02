import numpy as np
from manifpy import SO3Tangent

EPS = 1e-12

def skew(v):
    vx, vy, vz = v.flatten()
    return np.array([[0, -vz,  vy],
                     [vz,  0, -vx],
                     [-vy, vx,  0]], dtype=float)

def R(x):
    return SO3Tangent(x).exp().rotation()
def _inv_sqrt_2x2(S, tol=1e-12):
    # Eigen-based inverse square root for 2x2 SPD
    # S = V diag(w) V^T  ->  S^{-1/2} = V diag(1/sqrt(w)) V^T
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    # Clamp tiny/negative eigenvalues (numerical safety)
    w = np.maximum(w, tol)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

def B(x):
    """
    Equivalent of the C++ function:
    inline Matrix3x2d B(const Vector3d& x)
    Returns a 3x2 matrix forming an orthonormal basis for the tangent plane at x ∈ R^3.
    """
    e_i = np.array([0.0, 0.0, 1.0])  # UnitZ
    E_jk = np.eye(3)[:, :2]          # first two canonical basis vectors e1, e2

    norm_x = np.linalg.norm(x)
    if norm_x < 1e-12:
        return E_jk

    cross = np.cross(e_i.flatten(), x.flatten()).reshape(-1, 1)
    cross_norm = np.linalg.norm(cross)
    dot = np.dot(e_i, x)

    if cross_norm < 1e-12:
        # x parallel to e_i (either aligned or opposite)
        if dot >= 0:
            out = E_jk
        else:
            out = R(np.array([np.pi, 0.0, 0.0])) @ E_jk
    else:
        theta = np.arctan2(cross_norm, dot)
        axis = cross / cross_norm
        out = R(axis * theta) @ E_jk

    return out / norm_x


def Exp(x, u):
    if np.linalg.norm(u) < EPS:
        return x.copy()
    
    Bu = B(x) @ u
    angleAxis = np.cross(x.flatten(), Bu.flatten())
    return R(Bu) @ x

def ExpRot(x, u):
    if np.linalg.norm(u) < EPS:
        return x.copy()
    
    Bu = B(x) @ u
    angleAxis = np.cross(x.flatten(), Bu.flatten())
    return R(angleAxis)


def uExp(u):
    # The identity is e_i, the reference axis
    e_i = np.array([[0, 0, 1]]).T
    return Exp(e_i, u)

def Log(x, y):
    a = np.cross(x.flatten(), y.flatten()) 
    b = np.linalg.norm(a)
    c = float(x.T @ y)
    if b >= EPS:
        theta = np.arctan2(b, c)
        aa = np.cross(a.flatten(), x.flatten())
        axis = aa.reshape(-1, 1) / np.linalg.norm(aa)
        return B(x).T @ (theta * axis)
    
    return np.zeros((2, 1)) if c >= 0 else np.array([[np.pi, 0]]).T

def uLog(x):
    # The identity is e_i, the reference axis
    e_i = np.array([[0, 0, 1]]).T
    return Log(e_i, x)

def compose(x, y):
    e_i = np.array([[0, 0, 1]]).T
    R = ExpRot(e_i, uLog(y))
    return R @ x


def M(x, u):
    phi = B(x) @ u
    return - (R(phi) @ skew(x) @ SO3Tangent(phi).rjac() @ B(x))

def P(x, y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if abs(nx - ny) >= EPS:
        raise ValueError("P(x,y): x and y must have equal norm")
    r2 = float(y @ y)
    r4 = r2 * r2
    a = np.cross(x, y)
    b = np.linalg.norm(a)
    c = float(x @ y)
    theta = np.arctan2(b, c)
    sk_y = skew(y)
    sk_y2 = sk_y @ sk_y
    if b < EPS:
        b = EPS
    coeff = (-c * b + r4 * theta) / (b ** 3)
    row = (coeff * (x @ sk_y2) - y) / r4
    return row

def N(x, y):
    a = np.cross(x, y)
    b = np.linalg.norm(a)
    c = float(x @ y)
    theta = np.arctan2(b, c)
    sk_y = skew(y)
    if b < EPS:
        if c >= 0:
            denom = float(y @ y)
            denom = denom if denom >= EPS else EPS
            return (1.0 / denom) * (B(y).T @ sk_y)
        else:
            raise ValueError("singular at antipodal points")
    outer = np.outer(a, P(x, y))
    return B(y).T @ ((theta / b) * sk_y + outer)




if __name__ == "__main__":
    # Test J_r analogous
    x = np.array([[0., 3., 1.]]).T
    x /= np.linalg.norm(x)
    u = np.array([[np.pi, np.pi/2]]).T * 1e-2
    y = Exp(x, u)

    y1 = x + M(x, u) @ u  
    y2 = x + B(x) @ u  

    print("IkFom jacobian distance: ", np.linalg.norm(y-y1))
    print("LIMOncello jacobian distance: ", np.linalg.norm(y-y2))

    # Test Adjoint analogous
    e_i = np.array([[0, 0, 1]]).T

    x = np.array([[1., 0., 0.]]).T
    x /= np.linalg.norm(x)
    u = np.array([[np.sqrt(2)/2, np.sqrt(2)/2]]).T * np.pi/2

    right = Exp(x, u.copy())
    left = compose(uExp(u.copy()), x.copy())

    print("left: ", left)
    print("right: ", right)

