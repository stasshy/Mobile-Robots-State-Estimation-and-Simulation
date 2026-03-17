import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_unicycle(q, u, dt):
    x, y, th = q
    v, w = u
    return np.array([
        x + v * np.cos(th) * dt,
        y + v * np.sin(th) * dt,
        wrap_angle(th + w * dt)
    ])


def random_landmarks(n=4, xmin=-5, xmax=5, ymin=-5, ymax=5, min_dist=2.0):
    pts = []
    rng = np.random.default_rng()
    while len(pts) < n:
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)


def is_landmark_in_fov(q, landmark, fov_deg=180.0, max_range=7.0):
    x, y, th = q
    lx, ly = landmark
    dx, dy = lx - x, ly - y
    r = np.hypot(dx, dy)
    b = wrap_angle(np.arctan2(dy, dx) - th)
    half_fov = np.deg2rad(fov_deg / 2.0)
    return (-half_fov <= b <= half_fov) and (r <= max_range), r, b


def rb_to_xy(q, r, b):
    x, y, th = q
    ang = th + b
    return x + r * np.cos(ang), y + r * np.sin(ang)


def simulate_ground_truth_noiseless(q0, N, dt, u):
    q_hist = np.zeros((N, 3))
    q = q0.copy()
    for k in range(N):
        q = step_unicycle(q, u, dt)
        q_hist[k] = q
    return q_hist


class EKFSLAM:
    def __init__(self, R_motion, Q_meas, dt, n_landmarks, q0):
        self.dt = dt
        self.n_landmarks = n_landmarks
        self.n = 3 + 2 * n_landmarks

        self.mu = np.zeros((self.n, 1))
        self.mu[:3, 0] = q0

        self.Sigma = np.zeros((self.n, self.n))
        self.Sigma[0, 0] = 0.1**2
        self.Sigma[1, 1] = 0.1**2
        self.Sigma[2, 2] = np.deg2rad(5)**2
        self.Sigma[3:, 3:] = 1e6 * np.eye(self.n - 3)

        self.R = R_motion
        self.Q = Q_meas
        self.observed = [False] * n_landmarks

    def lm_idx(self, j):
        return 3 + 2 * j

    def predict(self, u):
        v, w = u
        x, y, th = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]

        self.mu[0, 0] = x + v * np.cos(th) * self.dt
        self.mu[1, 0] = y + v * np.sin(th) * self.dt
        self.mu[2, 0] = wrap_angle(th + w * self.dt)

        G = np.eye(self.n)
        G[0, 2] = -v * np.sin(th) * self.dt
        G[1, 2] =  v * np.cos(th) * self.dt

        R_full = np.zeros((self.n, self.n))
        R_full[:3, :3] = self.R
        self.Sigma = G @ self.Sigma @ G.T + R_full

    def init_landmark(self, j, z):
        r, b = z
        x, y, th = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]
        idx = self.lm_idx(j)
        self.mu[idx, 0] = x + r * np.cos(th + b)
        self.mu[idx + 1, 0] = y + r * np.sin(th + b)
        self.observed[j] = True

    def measurement_model(self, j):
        x, y, th = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]
        idx = self.lm_idx(j)
        lx, ly = self.mu[idx, 0], self.mu[idx + 1, 0]
        dx, dy = lx - x, ly - y
        r = np.hypot(dx, dy)
        b = wrap_angle(np.arctan2(dy, dx) - th)
        return np.array([[r], [b]]), dx, dy

    def measurement_jacobian(self, j, dx, dy):
        q = max(dx**2 + dy**2, 1e-9)
        r = max(np.sqrt(q), 1e-9)
        H = np.zeros((2, self.n))

        H[0, 0] = -dx / r
        H[0, 1] = -dy / r
        H[1, 0] =  dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0

        idx = self.lm_idx(j)
        H[0, idx] = dx / r
        H[0, idx + 1] = dy / r
        H[1, idx] = -dy / q
        H[1, idx + 1] = dx / q
        return H

    def correct_one(self, j, z):
        z = np.asarray(z).reshape(2)

        if not self.observed[j]:
            self.init_landmark(j, z)
            return

        z_pred, dx, dy = self.measurement_model(j)
        H = self.measurement_jacobian(j, dx, dy)

        S = H @ self.Sigma @ H.T + self.Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        innovation = z.reshape(2, 1) - z_pred
        innovation[1, 0] = wrap_angle(innovation[1, 0])

        self.mu = self.mu + K @ innovation
        self.mu[2, 0] = wrap_angle(self.mu[2, 0])

        I = np.eye(self.n)
        self.Sigma = (I - K @ H) @ self.Sigma @ (I - K @ H).T + K @ self.Q @ K.T

    def step(self, u, measurements):
        self.predict(u)
        for j, z in measurements:
            self.correct_one(j, z)


def simulate_slam(q0, N, dt, u, landmarks, R_motion, Q_meas, fov_deg=180.0, max_range=7.0, seed=7):
    rng = np.random.default_rng(seed)
    q = q0.copy()

    n_landmarks = len(landmarks)
    n_state = 3 + 2 * n_landmarks

    q_true = np.zeros((N, 3))
    mu_hist = np.zeros((N, n_state))
    observed_hist = np.zeros((N, n_landmarks), dtype=bool)

    ekf = EKFSLAM(R_motion, Q_meas, dt, n_landmarks, q0)

    for k in range(N):
        q_mean = step_unicycle(q, u, dt)
        q = q_mean + rng.multivariate_normal(np.zeros(3), R_motion)
        q[2] = wrap_angle(q[2])

        measurements = []
        for j, lm in enumerate(landmarks):
            ok, r_true, b_true = is_landmark_in_fov(q, lm, fov_deg=fov_deg, max_range=max_range)
            if ok:
                z = np.array([r_true, b_true]) + rng.multivariate_normal(np.zeros(2), Q_meas)
                z[1] = wrap_angle(z[1])
                measurements.append((j, z))

        ekf.step(u, measurements)

        q_true[k] = q
        mu_hist[k] = ekf.mu.flatten()
        observed_hist[k] = np.array(ekf.observed)

    return q_true, mu_hist, observed_hist


def animate_slam(q_true, mu_hist, q_gt, landmarks, observed_hist, out_gif,
                 fov_deg=180.0, max_range=7.0, fps=25):
    N = len(q_true)
    n_landmarks = len(landmarks)

    est_lms = np.zeros((N, n_landmarks, 2))
    for j in range(n_landmarks):
        idx = 3 + 2 * j
        est_lms[:, j, 0] = mu_hist[:, idx]
        est_lms[:, j, 1] = mu_hist[:, idx + 1]

    xs = np.concatenate([q_true[:, 0], mu_hist[:, 0], q_gt[:, 0], landmarks[:, 0], est_lms[:, :, 0].ravel()])
    ys = np.concatenate([q_true[:, 1], mu_hist[:, 1], q_gt[:, 1], landmarks[:, 1], est_lms[:, :, 1].ravel()])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xs.min() - 1, xs.max() + 1)
    ax.set_ylim(ys.min() - 1, ys.max() + 1)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("EKF-SLAM with random landmarks")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker="s", color="black", s=70, label="true landmarks")

    (true_ln,) = ax.plot([], [], color="black", lw=2, label="true")
    (ekf_ln,) = ax.plot([], [], "--", color="purple", lw=2, label="EKF")
    (gt_ln,) = ax.plot([], [], "--", color="lightgray", lw=2, label="noiseless")
    (true_pt,) = ax.plot([], [], "o", color="black")
    (ekf_pt,) = ax.plot([], [], "o", color="purple")
    (fov_l,) = ax.plot([], [], "-.", color="green", lw=1, label="FOV")
    (fov_r,) = ax.plot([], [], "-.", color="green", lw=1)

    est_pts = []
    colors = ["blue", "magenta", "cyan", "orange"]
    for j in range(n_landmarks):
        (pt,) = ax.plot([], [], "D", color=colors[j % len(colors)], ms=7, label=f"LM{j+1} est")
        est_pts.append(pt)

    ax.legend(loc="upper right")
    half_fov = np.deg2rad(fov_deg / 2.0)

    def init():
        true_ln.set_data([], [])
        ekf_ln.set_data([], [])
        gt_ln.set_data([], [])
        true_pt.set_data([], [])
        ekf_pt.set_data([], [])
        fov_l.set_data([], [])
        fov_r.set_data([], [])
        for pt in est_pts:
            pt.set_data([], [])
        return [true_ln, ekf_ln, gt_ln, true_pt, ekf_pt, fov_l, fov_r, *est_pts]

    def update(k):
        true_ln.set_data(q_true[:k+1, 0], q_true[:k+1, 1])
        ekf_ln.set_data(mu_hist[:k+1, 0], mu_hist[:k+1, 1])
        gt_ln.set_data(q_gt[:k+1, 0], q_gt[:k+1, 1])

        tx, ty, th = q_true[k]
        ex, ey = mu_hist[k, 0], mu_hist[k, 1]
        true_pt.set_data([tx], [ty])
        ekf_pt.set_data([ex], [ey])

        left_ang = th + half_fov
        right_ang = th - half_fov
        fov_l.set_data([tx, tx + max_range * np.cos(left_ang)], [ty, ty + max_range * np.sin(left_ang)])
        fov_r.set_data([tx, tx + max_range * np.cos(right_ang)], [ty, ty + max_range * np.sin(right_ang)])

        for j in range(n_landmarks):
            if observed_hist[k, j]:
                idx = 3 + 2 * j
                est_pts[j].set_data([mu_hist[k, idx]], [mu_hist[k, idx + 1]])
            else:
                est_pts[j].set_data([], [])

        return [true_ln, ekf_ln, gt_ln, true_pt, ekf_pt, fov_l, fov_r, *est_pts]

    anim = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=int(1000 / fps))
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {out_gif}")


if __name__ == "__main__":
    dt = 0.05
    T = 100.0
    N = int(T / dt)

    q0 = np.array([0.0, 0.0, 0.0])
    u = (1.0, 0.2)

    R_motion = np.diag([0.005**2, 0.005**2, np.deg2rad(0.2)**2])
    Q_meas = np.diag([0.01**2, np.deg2rad(0.25)**2])

    landmarks = random_landmarks(n=15, xmin=-5, xmax=5, ymin=-5, ymax=5, min_dist=2.0)

    fov_deg = 90.0
    max_range = 2.0

    q_true, mu_hist, observed_hist = simulate_slam(
        q0, N, dt, u, landmarks, R_motion, Q_meas,
        fov_deg=fov_deg, max_range=max_range, seed=7
    )

    q_gt = simulate_ground_truth_noiseless(q0, N, dt, u)

    animate_slam(
        q_true, mu_hist, q_gt, landmarks, observed_hist,
        out_gif="ekf_slam_random_4_landmarks.gif",
        fov_deg=fov_deg, max_range=max_range, fps=25
    )