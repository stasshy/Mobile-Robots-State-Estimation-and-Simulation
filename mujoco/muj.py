import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def step_unicycle(q, u, dt):
    x, y, th = q
    v, w = u
    return np.array([
        x + v * np.cos(th) * dt,
        y + v * np.sin(th) * dt,
        wrap_angle(th + w * dt),
    ])

def random_landmarks(n=8, xmin=-4.5, xmax=4.5, ymin=-4.5, ymax=4.5,
                     min_dist=2.0, seed=None):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)

def landmark_measurement(q, landmark):
    x, y, th = q
    lx, ly = landmark
    dx, dy = lx - x, ly - y
    r = np.hypot(dx, dy)
    b = wrap_angle(np.arctan2(dy, dx) - th)
    return r, b

def landmark_visible(q, landmark, fov_deg=360.0, max_range=3.0):
    r, b = landmark_measurement(q, landmark)
    if fov_deg >= 360.0:
        return r <= max_range, r, b
    half_fov = np.deg2rad(fov_deg / 2)
    return (-half_fov <= b <= half_fov) and (r <= max_range), r, b

def compute_control(q, target, landmarks, bounds, rng,
                    base_v=0.95, v_jitter=0.12,
                    wander_w=0.22, target_gain=1.10,
                    obstacle_dist=1.60, avoid_dist=1.05,
                    boundary_margin=0.80, max_w=1.35):
    x, y, th = q
    xmin, xmax, ymin, ymax = bounds

    v = base_v + rng.uniform(-v_jitter, v_jitter)
    w = rng.uniform(-wander_w, wander_w)

    # steer toward current target
    if target is not None:
        desired = np.arctan2(target[1] - y, target[0] - x)
        w += target_gain * wrap_angle(desired - th)

    # boundary avoidance
    if x > xmax - boundary_margin:
        w += wrap_angle(np.pi - th)
    elif x < xmin + boundary_margin:
        w += wrap_angle(0.0 - th)

    if y > ymax - boundary_margin:
        w += wrap_angle(-np.pi / 2 - th)
    elif y < ymin + boundary_margin:
        w += wrap_angle(np.pi / 2 - th)

    if (x > xmax - 0.30 or x < xmin + 0.30 or
        y > ymax - 0.30 or y < ymin + 0.30):
        v *= 0.45

    # obstacle avoidance
    nearest_dist = np.inf
    nearest_bearing = None
    for lm in landmarks:
        r, b = landmark_measurement(q, lm)
        if r < obstacle_dist and abs(b) < np.deg2rad(85):
            if r < nearest_dist:
                nearest_dist = r
                nearest_bearing = b

    if nearest_bearing is not None:
        w += -1.15 if nearest_bearing >= 0 else 1.15
        closeness = (obstacle_dist - nearest_dist) / obstacle_dist
        closeness = max(0.0, closeness)
        w *= (1.0 + 1.5 * closeness)
        v *= 0.25 if nearest_dist < avoid_dist else 0.60

    return np.array([
        np.clip(v, 0.30, 1.20),
        np.clip(w, -max_w, max_w)
    ])


def build_control_sequence(q0, N, dt, landmarks, bounds,
                           fov_deg=360.0, max_range=3.0, seed=11):
    rng = np.random.default_rng(seed)
    q = q0.copy()

    U = np.zeros((N, 2))
    discovered = np.zeros(len(landmarks), dtype=bool)
    current_target = None

    for k in range(N):
        for j, lm in enumerate(landmarks):
            ok, _, _ = landmark_visible(q, lm, fov_deg, max_range)
            if ok:
                discovered[j] = True

        remaining = np.where(~discovered)[0]
        target = None

        if len(remaining) > 0:
            if current_target is None or discovered[current_target]:
                dists = [np.linalg.norm(landmarks[j] - q[:2]) for j in remaining]
                current_target = remaining[int(np.argmin(dists))]
            target = landmarks[current_target]
            if np.linalg.norm(target - q[:2]) < 0.8:
                current_target = None

        u = compute_control(q, target, landmarks, bounds, rng)
        q = step_unicycle(q, u, dt)

        xmin, xmax, ymin, ymax = bounds
        q[0] = np.clip(q[0], xmin + 0.08, xmax - 0.08)
        q[1] = np.clip(q[1], ymin + 0.08, ymax - 0.08)
        q[2] = wrap_angle(q[2])

        U[k] = u

    return U

class EKFSLAM:
    def __init__(self, dt, n_landmarks, q0, R_motion, Q_meas):
        self.dt = dt
        self.n_landmarks = n_landmarks
        self.n = 3 + 2 * n_landmarks

        self.mu = np.zeros((self.n, 1))
        self.mu[:3, 0] = q0

        self.Sigma = np.zeros((self.n, self.n))
        self.Sigma[:3, :3] = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])
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
        q = max(dx**2 + dy**2, 1e-9)
        r = max(np.sqrt(q), 1e-9)
        zhat = np.array([[r], [wrap_angle(np.arctan2(dy, dx) - th)]])
        return zhat, dx, dy, q, r

    def measurement_jacobian(self, j, dx, dy, q, r):
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

        zhat, dx, dy, q, r = self.measurement_model(j)
        H = self.measurement_jacobian(j, dx, dy, q, r)

        S = H @ self.Sigma @ H.T + self.Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        innovation = z.reshape(2, 1) - zhat
        innovation[1, 0] = wrap_angle(innovation[1, 0])

        self.mu = self.mu + K @ innovation
        self.mu[2, 0] = wrap_angle(self.mu[2, 0])

        I = np.eye(self.n)
        self.Sigma = (I - K @ H) @ self.Sigma @ (I - K @ H).T + K @ self.Q @ K.T

    def step(self, u, measurements):
        self.predict(u)
        mu_pred = self.mu.copy()
        Sigma_pred = self.Sigma.copy()
        for j, z in measurements:
            self.correct_one(j, z)
        return mu_pred, Sigma_pred

def simulate_ground_truth(q0, dt, U, R_motion=None, seed=7):
    rng = np.random.default_rng(seed)
    q = q0.copy()
    hist = np.zeros((len(U), 3))

    for k, u in enumerate(U):
        q = step_unicycle(q, u, dt)
        if R_motion is not None:
            q = q + rng.multivariate_normal(np.zeros(3), R_motion)
            q[2] = wrap_angle(q[2])
        hist[k] = q

    return hist

def simulate_slam(q0, dt, U, landmarks, R_motion, Q_meas,
                  fov_deg=360.0, max_range=3.0, seed=7):
    rng = np.random.default_rng(seed)
    q = q0.copy()

    N = len(U)
    n_landmarks = len(landmarks)
    n_state = 3 + 2 * n_landmarks

    q_true = np.zeros((N, 3))
    mu_hist = np.zeros((N, n_state))
    mu_pred_hist = np.zeros((N, 3))
    sigma_trace_hist = np.zeros(N)
    observed_hist = np.zeros((N, n_landmarks), dtype=bool)

    ekf = EKFSLAM(dt, n_landmarks, q0, R_motion, Q_meas)

    for k, u in enumerate(U):
        q = step_unicycle(q, u, dt) + rng.multivariate_normal(np.zeros(3), R_motion)
        q[2] = wrap_angle(q[2])

        measurements = []
        for j, lm in enumerate(landmarks):
            ok, r, b = landmark_visible(q, lm, fov_deg, max_range)
            if ok:
                z = np.array([r, b]) + rng.multivariate_normal(np.zeros(2), Q_meas)
                z[1] = wrap_angle(z[1])
                measurements.append((j, z))

        mu_pred, Sigma_pred = ekf.step(u, measurements)

        q_true[k] = q
        mu_hist[k] = ekf.mu.flatten()
        mu_pred_hist[k] = mu_pred[:3, 0]
        sigma_trace_hist[k] = np.trace(Sigma_pred[:3, :3])
        observed_hist[k] = np.array(ekf.observed)

    return q_true, mu_hist, mu_pred_hist, sigma_trace_hist, observed_hist


def cube_xml_from_landmarks(landmarks):
    rng = np.random.default_rng(123)
    parts = []
    for i, (x, y) in enumerate(landmarks):
        sx, sy, sz = rng.uniform(0.12, 0.22, size=3)
        parts.append(f"""
    <body name="obstacle_{i}" pos="{x:.3f} {y:.3f} {sz:.3f}">
      <geom type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" rgba="0.2 0.2 0.8 1"/>
    </body>""")
    return "\n".join(parts)

def est_landmark_xml(n_landmarks):
    return "\n".join(
        f"""
    <body name="est_landmark_{j}" mocap="true" pos="1000 1000 0.45">
      <geom type="sphere" size="0.09" material="ekf_lm_mat" contype="0" conaffinity="0"/>
    </body>"""
        for j in range(n_landmarks)
    )

def boundary_walls_xml(bounds, wall_thickness=0.06, wall_height=0.35):
    xmin, xmax, ymin, ymax = bounds
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    lx, ly = xmax - xmin, ymax - ymin
    z = wall_height

    return f"""
    <body name="wall_bottom" pos="{cx:.3f} {ymin:.3f} {z:.3f}">
      <geom type="box" size="{lx/2:.3f} {wall_thickness:.3f} {wall_height:.3f}" rgba="0.8 0.2 0.2 1"/>
    </body>
    <body name="wall_top" pos="{cx:.3f} {ymax:.3f} {z:.3f}">
      <geom type="box" size="{lx/2:.3f} {wall_thickness:.3f} {wall_height:.3f}" rgba="0.8 0.2 0.2 1"/>
    </body>
    <body name="wall_left" pos="{xmin:.3f} {cy:.3f} {z:.3f}">
      <geom type="box" size="{wall_thickness:.3f} {ly/2:.3f} {wall_height:.3f}" rgba="0.8 0.2 0.2 1"/>
    </body>
    <body name="wall_right" pos="{xmax:.3f} {cy:.3f} {z:.3f}">
      <geom type="box" size="{wall_thickness:.3f} {ly/2:.3f} {wall_height:.3f}" rgba="0.8 0.2 0.2 1"/>
    </body>
    """

def build_scene(template_path, out_path, landmarks, bounds):
    text = template_path.read_text(encoding="utf-8")
    if "<!-- OBSTACLES_GO_HERE -->" not in text:
        raise ValueError("Λείπει το placeholder <!-- OBSTACLES_GO_HERE -->")

    generated = text.replace(
        "<!-- OBSTACLES_GO_HERE -->",
        cube_xml_from_landmarks(landmarks)
        + "\n"
        + est_landmark_xml(len(landmarks))
        + "\n"
        + boundary_walls_xml(bounds)
    )
    out_path.write_text(generated, encoding="utf-8")

def yaw_to_quat(th):
    return np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])


def set_free_body_pose(model, data, joint_name, x, y, th, z=0.0):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = [x, y, z]
    data.qpos[adr + 3:adr + 7] = yaw_to_quat(th)


def set_mocap_body(model, data, body_name, pos):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mid = model.body_mocapid[bid]
    data.mocap_pos[mid] = np.array(pos)
    data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0])


def setup_live_plot(landmarks, t, fov_deg=360.0, max_range=3.0):
    plt.ion()

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0])

    ax_map = fig.add_subplot(gs[:, 0])
    ax_mu = fig.add_subplot(gs[0, 1])
    ax_sigma = fig.add_subplot(gs[1, 1])

    ax_map.set_aspect("equal")
    ax_map.grid(True)
    ax_map.set_title("Live 2D EKF-SLAM")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.scatter(landmarks[:, 0], landmarks[:, 1], marker="s", color="blue", s=70, label="true landmarks")

    true_ln, = ax_map.plot([], [], color="blue", lw=2, label="true noisy")
    ekf_ln, = ax_map.plot([], [], "--", color="purple", lw=2, label="EKF")
    gt_ln, = ax_map.plot([], [], "--", color="lightgray", lw=2, label="noiseless")
    true_pt, = ax_map.plot([], [], "o", color="blue")
    ekf_pt, = ax_map.plot([], [], "o", color="purple")
    gt_pt, = ax_map.plot([], [], "o", color="gray")

    lidar_circle, = ax_map.plot([], [], "-", color="green", lw=1.2, label="lidar range")
    lidar_hits, = ax_map.plot([], [], "ro", ms=4, label="lidar detections")
    heading_arrow, = ax_map.plot([], [], "-", color="orange", lw=2, label="heading")

    est_pts = [ax_map.plot([], [], "D", color="magenta", ms=7)[0] for _ in range(len(landmarks))]

    ax_map.set_xlim(landmarks[:, 0].min() - 3, landmarks[:, 0].max() + 3)
    ax_map.set_ylim(landmarks[:, 1].min() - 3, landmarks[:, 1].max() + 3)
    ax_map.legend(loc="upper right")

    ax_mu.grid(True)
    ax_mu.set_title("EKF prediction mean")
    ax_mu.set_xlabel("t [s]")
    ax_mu.set_ylabel("mu")
    mu_x_ln, = ax_mu.plot([], [], label="mu_x")
    mu_y_ln, = ax_mu.plot([], [], label="mu_y")
    mu_th_ln, = ax_mu.plot([], [], label="mu_theta")
    ax_mu.set_xlim(t[0], t[-1])
    ax_mu.legend()

    ax_sigma.grid(True)
    ax_sigma.set_title("trace(Sigma_pose)")
    ax_sigma.set_xlabel("t [s]")
    sigma_ln, = ax_sigma.plot([], [])
    ax_sigma.set_xlim(t[0], t[-1])

    return {
        "fig": fig, "ax_mu": ax_mu, "ax_sigma": ax_sigma,
        "true_ln": true_ln, "ekf_ln": ekf_ln, "gt_ln": gt_ln,
        "true_pt": true_pt, "ekf_pt": ekf_pt, "gt_pt": gt_pt,
        "lidar_circle": lidar_circle, "lidar_hits": lidar_hits,
        "heading_arrow": heading_arrow, "est_pts": est_pts,
        "mu_x_ln": mu_x_ln, "mu_y_ln": mu_y_ln, "mu_th_ln": mu_th_ln,
        "sigma_ln": sigma_ln, "fov_deg": fov_deg, "max_range": max_range,
    }


def update_live_plot(plotters, t, q_true_hist, mu_hist, mu_pred_hist,
                     sigma_trace_hist, q_gt_hist, observed_hist, landmarks, k):
    tx, ty, th = q_true_hist[k]
    ex, ey = mu_hist[k, 0], mu_hist[k, 1]
    gx, gy = q_gt_hist[k, 0], q_gt_hist[k, 1]

    plotters["true_ln"].set_data(q_true_hist[:k+1, 0], q_true_hist[:k+1, 1])
    plotters["ekf_ln"].set_data(mu_hist[:k+1, 0], mu_hist[:k+1, 1])
    plotters["gt_ln"].set_data(q_gt_hist[:k+1, 0], q_gt_hist[:k+1, 1])

    plotters["true_pt"].set_data([tx], [ty])
    plotters["ekf_pt"].set_data([ex], [ey])
    plotters["gt_pt"].set_data([gx], [gy])

    # lidar range
    fov_deg = plotters["fov_deg"]
    max_range = plotters["max_range"]
    if fov_deg >= 360.0:
        ang = np.linspace(0, 2*np.pi, 200)
    else:
        half = np.deg2rad(fov_deg / 2)
        ang = np.linspace(th - half, th + half, 120)
    plotters["lidar_circle"].set_data(
        tx + max_range * np.cos(ang),
        ty + max_range * np.sin(ang),
    )

    plotters["heading_arrow"].set_data(
        [tx, tx + 0.8 * np.cos(th)],
        [ty, ty + 0.8 * np.sin(th)],
    )

    # visible landmarks
    hit_x, hit_y = [], []
    for lm in landmarks:
        ok, _, _ = landmark_visible(q_true_hist[k], lm, fov_deg, max_range)
        if ok:
            hit_x.append(lm[0])
            hit_y.append(lm[1])
    plotters["lidar_hits"].set_data(hit_x, hit_y)

    # estimated landmarks
    for j, pt in enumerate(plotters["est_pts"]):
        if observed_hist[k, j]:
            idx = 3 + 2 * j
            pt.set_data([mu_hist[k, idx]], [mu_hist[k, idx + 1]])
        else:
            pt.set_data([], [])

    # mu plots
    plotters["mu_x_ln"].set_data(t[:k+1], mu_pred_hist[:k+1, 0])
    plotters["mu_y_ln"].set_data(t[:k+1], mu_pred_hist[:k+1, 1])
    plotters["mu_th_ln"].set_data(t[:k+1], mu_pred_hist[:k+1, 2])

    mu_block = mu_pred_hist[:k+1]
    mu_min, mu_max = np.min(mu_block), np.max(mu_block)
    if abs(mu_max - mu_min) < 1e-8:
        mu_min, mu_max = mu_min - 1.0, mu_max + 1.0
    pad_mu = 0.08 * (mu_max - mu_min)
    plotters["ax_mu"].set_ylim(mu_min - pad_mu, mu_max + pad_mu)

    # sigma plot
    plotters["sigma_ln"].set_data(t[:k+1], sigma_trace_hist[:k+1])
    smin, smax = np.min(sigma_trace_hist[:k+1]), np.max(sigma_trace_hist[:k+1])
    if abs(smax - smin) < 1e-12:
        smin, smax = 0.0, smax + 1.0
    pad_s = 0.08 * (smax - smin)
    plotters["ax_sigma"].set_ylim(max(0.0, smin - pad_s), smax + pad_s)

    plotters["fig"].canvas.draw_idle()
    plotters["fig"].canvas.flush_events()
    plt.pause(0.001)

def main():
    dt = 0.05
    T = 300.0
    N = int(T / dt)
    t = np.arange(N) * dt

    q0 = np.array([0.0, 0.0, 0.0])
    R_motion = np.diag([0.005**2, 0.005**2, np.deg2rad(0.2)**2])
    Q_meas = np.diag([0.01**2, np.deg2rad(0.25)**2])

    fov_deg = 360.0
    max_range = 3.0
    bounds = (-5.0, 5.0, -5.0, 5.0)

    landmarks = random_landmarks(n=8, xmin=-4.5, xmax=4.5, ymin=-4.5, ymax=4.5,
                                 min_dist=2.0, seed=5)

    U = build_control_sequence(q0, N, dt, landmarks, bounds, fov_deg, max_range, seed=11)

    base = Path(__file__).parent
    template = base / "TB3-WafflePi scene.xml"
    scene = base / "generated_scene.xml"
    build_scene(template, scene, landmarks, bounds)

    q_true_hist, mu_hist, mu_pred_hist, sigma_trace_hist, observed_hist = simulate_slam(
        q0, dt, U, landmarks, R_motion, Q_meas, fov_deg, max_range, seed=7
    )
    q_gt_hist = simulate_ground_truth(q0, dt, U)

    plotters = setup_live_plot(landmarks, t, fov_deg, max_range)

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
            viewer.cam.distance = 12.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -50

        k = 0
        while viewer.is_running() and plt.fignum_exists(plotters["fig"].number):
            i = k % N
            q_true = q_true_hist[i]
            q_gt = q_gt_hist[i]
            ekf_x, ekf_y, ekf_th = mu_hist[i, 0], mu_hist[i, 1], mu_hist[i, 2]

            with viewer.lock():
                set_free_body_pose(model, data, "base_joint", q_true[0], q_true[1], q_true[2])
                set_mocap_body(model, data, "true_robot_marker", [q_true[0], q_true[1], 0.08])
                set_mocap_body(model, data, "ekf_robot_marker", [ekf_x, ekf_y, 0.10])
                set_mocap_body(model, data, "gt_robot_marker", [q_gt[0], q_gt[1], 0.12])

                # hide old FOV rays for 360 lidar
                set_mocap_body(model, data, "fov_left_marker", [1000, 1000, 0.04])
                set_mocap_body(model, data, "fov_right_marker", [1000, 1000, 0.04])

                for j in range(len(landmarks)):
                    if observed_hist[i, j]:
                        idx = 3 + 2 * j
                        lx, ly = mu_hist[i, idx], mu_hist[i, idx + 1]
                        set_mocap_body(model, data, f"est_landmark_{j}", [lx, ly, 0.45])
                    else:
                        set_mocap_body(model, data, f"est_landmark_{j}", [1000, 1000, 0.45])

                mujoco.mj_forward(model, data)

            if i % 10 == 0:
                print(f"\nStep {i}")
                print(f"Control u = ({U[i,0]:.3f}, {U[i,1]:.3f})")
                print(f"True robot = ({q_true[0]:.3f}, {q_true[1]:.3f}, {q_true[2]:.3f})")
                print(f"EKF  robot = ({ekf_x:.3f}, {ekf_y:.3f}, {ekf_th:.3f})")
                print(f"trace(Sigma_pose_pred) = {sigma_trace_hist[i]:.6f}")
                print("Estimated landmarks:")
                for j in range(len(landmarks)):
                    if observed_hist[i, j]:
                        idx = 3 + 2 * j
                        print(f"  LM{j}: ({mu_hist[i, idx]:.3f}, {mu_hist[i, idx + 1]:.3f})")
                    else:
                        print(f"  LM{j}: not observed yet")

            update_live_plot(
                plotters, t, q_true_hist, mu_hist, mu_pred_hist,
                sigma_trace_hist, q_gt_hist, observed_hist, landmarks, i
            )

            viewer.sync()
            time.sleep(dt)
            k += 1

    plt.close("all")


if __name__ == "__main__":
    main()