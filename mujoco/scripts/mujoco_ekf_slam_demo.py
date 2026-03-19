import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

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

def compute_control(q, target, landmarks, bounds, rng, explorer_state,
                    base_speed=7.8, speed_jitter=0.0,
                    wander_bias=0.22, target_gain=1.10,
                    obstacle_dist=1.60, avoid_dist=1.05,
                    boundary_margin=0.80, max_diff=2.4,
                    forced_turn_diff=6.5, forced_turn_speed=1.8):
    x, y, th = q
    xmin, xmax, ymin, ymax = bounds

    speed_center = base_speed + rng.uniform(-speed_jitter, speed_jitter)
    diff = rng.uniform(-wander_bias, wander_bias)

    if target is not None:
        desired = np.arctan2(target[1] - y, target[0] - x)
        diff += target_gain * wrap_angle(desired - th)

    if x > xmax - boundary_margin:
        diff += wrap_angle(np.pi - th)
    elif x < xmin + boundary_margin:
        diff += wrap_angle(0.0 - th)

    if y > ymax - boundary_margin:
        diff += wrap_angle(-np.pi / 2 - th)
    elif y < ymin + boundary_margin:
        diff += wrap_angle(np.pi / 2 - th)

    if (x > xmax - 0.30 or x < xmin + 0.30 or
        y > ymax - 0.30 or y < ymin + 0.30):
        speed_center *= 0.75

    nearest_dist = np.inf
    nearest_bearing = None
    for lm in landmarks:
        r, b = landmark_measurement(q, lm)
        if r < obstacle_dist and abs(b) < np.deg2rad(85):
            if r < nearest_dist:
                nearest_dist = r
                nearest_bearing = b

    if nearest_bearing is not None:
        diff += -1.15 if nearest_bearing >= 0 else 1.15
        closeness = (obstacle_dist - nearest_dist) / obstacle_dist
        closeness = max(0.0, closeness)
        diff *= (1.0 + 1.5 * closeness)
        speed_center *= 0.65 if nearest_dist < avoid_dist else 0.80

    if explorer_state["forced_turn_steps"] > 0:
        diff = explorer_state["turn_sign"] * forced_turn_diff
        speed_center = forced_turn_speed
        explorer_state["forced_turn_steps"] -= 1

    diff = np.clip(diff, -max_diff, max_diff)

    v_left = speed_center - diff
    v_right = speed_center + diff

    v_left = np.clip(v_left, -7.88, 7.88)
    v_right = np.clip(v_right, -7.88, 7.88)

    return np.array([v_left, v_right])

def yaw_to_quat(th):
    return np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])

def quat_to_yaw(quat):
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def set_free_body_pose(model, data, joint_name, x, y, th, z=0.0):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = [x, y, z]
    data.qpos[adr + 3:adr + 7] = yaw_to_quat(th)

def get_free_body_pose(model, data, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    x, y, _ = data.qpos[adr:adr + 3]
    quat = data.qpos[adr + 3:adr + 7]
    th = wrap_angle(quat_to_yaw(quat))
    return np.array([x, y, th])

def set_mocap_body(model, data, body_name, pos):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mid = model.body_mocapid[bid]
    data.mocap_pos[mid] = np.array(pos)
    data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0])

def build_mujoco_sim_handles(model):
    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right")

    if left_act == -1 or right_act == -1:
        raise ValueError("Δεν βρέθηκαν τα actuators wheel_left / wheel_right")

    return {
        "base_joint": "base_joint",
        "left_act": left_act,
        "right_act": right_act,
    }

def mujoco_motion_step(model, data, handles, u_lr):
    ctrl_min, ctrl_max = -7.88, 7.88

    data.ctrl[handles["left_act"]] = np.clip(u_lr[0], ctrl_min, ctrl_max)
    data.ctrl[handles["right_act"]] = np.clip(u_lr[1], ctrl_min, ctrl_max)

    mujoco.mj_step(model, data)

    q_next = get_free_body_pose(model, data, handles["base_joint"])
    q_next[2] = wrap_angle(q_next[2])
    return q_next

def add_motion_noise(q, R_motion, rng):
    qn = q.copy()
    qn = qn + rng.multivariate_normal(np.zeros(3), R_motion)
    qn[2] = wrap_angle(qn[2])
    return qn

def mujoco_measurements(q, landmarks, Q_meas, rng,
                        fov_deg=360.0, max_range=3.0):
    measurements = []
    for j, lm in enumerate(landmarks):
        ok, r, b = landmark_visible(q, lm, fov_deg, max_range)
        if ok:
            z = np.array([r, b]) + rng.multivariate_normal(np.zeros(2), Q_meas)
            z[1] = wrap_angle(z[1])
            measurements.append((j, z))
    return measurements

class EKFSLAM:
    def __init__(self, dt, n_landmarks, q0, R_motion, Q_meas, wheel_base):
        self.dt = dt
        self.n_landmarks = n_landmarks
        self.n = 3 + 2 * n_landmarks
        self.wheel_base = wheel_base

        self.mu = np.zeros((self.n, 1))
        self.mu[:3, 0] = q0

        self.Sigma = np.zeros((self.n, self.n))
        self.Sigma[:3, :3] = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])
        self.Sigma[3:, 3:] = 1e6 * np.eye(self.n - 3)

        self.R = R_motion
        self.Q = Q_meas
        self.observed = [False] * n_landmarks

    def init_landmark(self, j, z):
        r, b = z
        x, y, th = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]
        idx = self.lm_idx(j)
        self.mu[idx, 0] = x + r * np.cos(th + b)
        self.mu[idx + 1, 0] = y + r * np.sin(th + b)
        self.observed[j] = True

    def lm_idx(self, j):
        return 3 + 2 * j

    def predict(self, delta):
        dx, dy, dth = delta

        self.mu[0, 0] = self.mu[0, 0] + dx
        self.mu[1, 0] = self.mu[1, 0] + dy
        self.mu[2, 0] = wrap_angle(self.mu[2, 0] + dth)

        G = np.eye(self.n)

        R_full = np.zeros((self.n, self.n))
        R_full[:3, :3] = self.R

        self.Sigma = G @ self.Sigma @ G.T + R_full

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

    def step(self, delta, measurements):
        self.predict(delta)

        mu_pred = self.mu.copy()
        Sigma_pred = self.Sigma.copy()

        for j, z in measurements:
            self.correct_one(j, z)

        return mu_pred, Sigma_pred


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

def setup_live_plot(landmarks, t, fov_deg=360.0, max_range=3.0):
    plt.ion()

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85])

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

    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry("900x650+20+80")
    except Exception:
        try:
            manager.window.setGeometry(20, 80, 900, 650)
        except Exception:
            pass

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
    tx_true, ty_true, th_true = q_true_hist[k]
    tx_gt, ty_gt, _ = q_gt_hist[k]
    ex, ey = mu_hist[k, 0], mu_hist[k, 1]

    plotters["true_ln"].set_data(q_true_hist[:k+1, 0], q_true_hist[:k+1, 1])
    plotters["ekf_ln"].set_data(mu_hist[:k+1, 0], mu_hist[:k+1, 1])
    plotters["gt_ln"].set_data(q_gt_hist[:k+1, 0], q_gt_hist[:k+1, 1])

    plotters["true_pt"].set_data([tx_true], [ty_true])
    plotters["ekf_pt"].set_data([ex], [ey])
    plotters["gt_pt"].set_data([tx_gt], [ty_gt])

    fov_deg = plotters["fov_deg"]
    max_range = plotters["max_range"]
    if fov_deg >= 360.0:
        ang = np.linspace(0, 2 * np.pi, 200)
    else:
        half = np.deg2rad(fov_deg / 2)
        ang = np.linspace(th_true - half, th_true + half, 120)

    plotters["lidar_circle"].set_data(
        tx_true + max_range * np.cos(ang),
        ty_true + max_range * np.sin(ang),
    )

    plotters["heading_arrow"].set_data(
        [tx_true, tx_true + 0.8 * np.cos(th_true)],
        [ty_true, ty_true + 0.8 * np.sin(th_true)],
    )

    hit_x, hit_y = [], []
    for lm in landmarks:
        ok, _, _ = landmark_visible(q_true_hist[k], lm, fov_deg, max_range)
        if ok:
            hit_x.append(lm[0])
            hit_y.append(lm[1])
    plotters["lidar_hits"].set_data(hit_x, hit_y)

    for j, pt in enumerate(plotters["est_pts"]):
        if observed_hist[k, j]:
            idx = 3 + 2 * j
            pt.set_data([mu_hist[k, idx]], [mu_hist[k, idx + 1]])
        else:
            pt.set_data([], [])

    plotters["mu_x_ln"].set_data(t[:k+1], mu_pred_hist[:k+1, 0])
    plotters["mu_y_ln"].set_data(t[:k+1], mu_pred_hist[:k+1, 1])
    plotters["mu_th_ln"].set_data(t[:k+1], mu_pred_hist[:k+1, 2])

    mu_block = mu_pred_hist[:k+1]
    mu_min, mu_max = np.min(mu_block), np.max(mu_block)
    if abs(mu_max - mu_min) < 1e-8:
        mu_min, mu_max = mu_min - 1.0, mu_max + 1.0
    pad_mu = 0.08 * (mu_max - mu_min)
    plotters["ax_mu"].set_ylim(mu_min - pad_mu, mu_max + pad_mu)

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
    dt = 0.15
    T = 150.0
    N = int(T / dt)
    t = np.arange(N + 1) * dt

    q0 = np.array([0.0, 0.0, 0.0])
    R_motion = np.diag([0.03**2, 0.03**2, np.deg2rad(1.0)**2])
    Q_meas = np.diag([0.05**2, np.deg2rad(1.0)**2])

    fov_deg = 360.0
    max_range = 3.0
    bounds = (-5.0, 5.0, -5.0, 5.0)
    wheel_base = 0.288

    rng_ctrl = np.random.default_rng(11)
    rng_noise = np.random.default_rng(7)

    current_target = None

    explorer_state = {
        "no_landmark_steps": 0,
        "forced_turn_steps": 0,
        "turn_sign": 1.0,
    }

    landmarks = random_landmarks(
        n=8, xmin=-4.5, xmax=4.5, ymin=-4.5, ymax=4.5,
        min_dist=2.0
    )

    base = Path(__file__).resolve().parent.parent
    template = base / "models" / "TB3-WafflePi scene.xml"
    scene = base / "models" / "generated_scene.xml"
    build_scene(template, scene, landmarks, bounds)

    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = dt

    handles = build_mujoco_sim_handles(model)

    data_gt = mujoco.MjData(model)
    data_true = mujoco.MjData(model)
    data_view = mujoco.MjData(model)

    set_free_body_pose(model, data_gt, handles["base_joint"], q0[0], q0[1], q0[2])
    set_free_body_pose(model, data_view, handles["base_joint"], q0[0], q0[1], q0[2])
    set_free_body_pose(model, data_true, handles["base_joint"], q0[0], q0[1], q0[2])

    mujoco.mj_forward(model, data_gt)
    mujoco.mj_forward(model, data_view)
    mujoco.mj_forward(model, data_true)

    q_gt = q0.copy()
    q_true = q0.copy()

    n_landmarks = len(landmarks)
    n_state = 3 + 2 * n_landmarks

    q_gt_hist = np.zeros((N + 1, 3))
    q_true_hist = np.zeros((N + 1, 3))
    mu_hist = np.zeros((N + 1, n_state))
    mu_pred_hist = np.zeros((N + 1, 3))
    sigma_trace_hist = np.zeros(N + 1)
    observed_hist = np.zeros((N + 1, n_landmarks), dtype=bool)
    u_hist = np.zeros((N + 1, 2))

    ekf = EKFSLAM(dt, n_landmarks, q0, R_motion, Q_meas, wheel_base)

    q_gt_hist[0] = q0
    q_true_hist[0] = q0
    mu_hist[0] = ekf.mu.flatten()
    mu_pred_hist[0] = ekf.mu[:3, 0]
    sigma_trace_hist[0] = np.trace(ekf.Sigma[:3, :3])
    observed_hist[0] = np.array(ekf.observed)
    u_hist[0] = np.array([0.0, 0.0])

    xmin, xmax, ymin, ymax = bounds
    discovered = np.zeros(n_landmarks, dtype=bool)
    current_target = None

    plotters = setup_live_plot(landmarks, t, fov_deg, max_range)

    with mujoco.viewer.launch_passive(model, data_view, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
            viewer.cam.distance = 12.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -50

        k = 0
        while viewer.is_running() and plt.fignum_exists(plotters["fig"].number) and k < N:
            visible_now = False
            for j, lm in enumerate(landmarks):
                ok, _, _ = landmark_visible(q_gt, lm, fov_deg, max_range)
                if ok:
                    discovered[j] = True
                    visible_now = True

            if visible_now:
                explorer_state["no_landmark_steps"] = 0
            else:
                explorer_state["no_landmark_steps"] += 1

            if explorer_state["forced_turn_steps"] == 0 and explorer_state["no_landmark_steps"] >= 20:
                explorer_state["turn_sign"] = rng_ctrl.choice([-1.0, 1.0])
                explorer_state["forced_turn_steps"] = 8
                explorer_state["no_landmark_steps"] = 0

            remaining = np.where(~discovered)[0]
            target = None

            if len(remaining) > 0:
                if current_target is None or discovered[current_target]:
                    dists = [np.linalg.norm(landmarks[j] - q_gt[:2]) for j in remaining]
                    current_target = remaining[int(np.argmin(dists))]
                target = landmarks[current_target]
                if np.linalg.norm(target - q_gt[:2]) < 0.8:
                    current_target = None

            u_lr = compute_control(q_gt, target, landmarks, bounds, rng_ctrl, explorer_state)

            q_gt_prev = q_gt.copy()
            q_gt = mujoco_motion_step(model, data_gt, handles, u_lr)
            q_gt[2] = wrap_angle(q_gt[2])

            delta_gt = np.array([
                q_gt[0] - q_gt_prev[0],
                q_gt[1] - q_gt_prev[1],
                wrap_angle(q_gt[2] - q_gt_prev[2]),
            ])   

            q_true = q_gt.copy()
            q_true = add_motion_noise(q_true, R_motion, rng_noise)
            q_true[0] = np.clip(q_true[0], xmin + 0.08, xmax - 0.08)
            q_true[1] = np.clip(q_true[1], ymin + 0.08, ymax - 0.08)
            q_true[2] = wrap_angle(q_true[2])

            measurements = mujoco_measurements(
                q_true, landmarks, Q_meas, rng_noise,
                fov_deg=fov_deg, max_range=max_range
            )

            mu_pred, Sigma_pred = ekf.step(delta_gt, measurements)

            q_gt_hist[k + 1] = q_gt
            q_true_hist[k + 1] = q_true
            mu_hist[k + 1] = ekf.mu.flatten()
            mu_pred_hist[k + 1] = mu_pred[:3, 0]
            sigma_trace_hist[k + 1] = np.trace(Sigma_pred[:3, :3])
            observed_hist[k + 1] = np.array(ekf.observed)
            # u_hist[k + 1] = u_lr_exec

            i = k + 1
            ekf_x, ekf_y, ekf_th = mu_hist[i, 0], mu_hist[i, 1], mu_hist[i, 2]

            with viewer.lock():
                set_free_body_pose(model, data_view, "base_joint", q_gt[0], q_gt[1], q_gt[2])

                set_mocap_body(model, data_view, "ekf_robot_marker", [ekf_x, ekf_y, 0.10])

                # set_mocap_body(model, data_view, "fov_left_marker", [1000, 1000, 0.04])
                # set_mocap_body(model, data_view, "fov_right_marker", [1000, 1000, 0.04])

                for jj in range(len(landmarks)):
                    if observed_hist[i, jj]:
                        idx = 3 + 2 * jj
                        lx, ly = mu_hist[i, idx], mu_hist[i, idx + 1]
                        set_mocap_body(model, data_view, f"est_landmark_{jj}", [lx, ly, 0.45])
                    else:
                        set_mocap_body(model, data_view, f"est_landmark_{jj}", [1000, 1000, 0.45])

                mujoco.mj_forward(model, data_view)

            if i % 10 == 0:
                print(f"\nStep {i}")
                print(f"Control delta = ({delta_gt[0]:.3f}, {delta_gt[1]:.3f}, {delta_gt[2]:.3f})")
                print(f"Ground truth = ({q_gt[0]:.3f}, {q_gt[1]:.3f}, {q_gt[2]:.3f})")
                print(f"Noisy true   = ({q_true[0]:.3f}, {q_true[1]:.3f}, {q_true[2]:.3f})")
                print(f"EKF estimate = ({ekf_x:.3f}, {ekf_y:.3f}, {ekf_th:.3f})")
                print(f"trace(Sigma_pose_pred) = {sigma_trace_hist[i]:.6f}")
                print("Estimated landmarks:")
                for jj in range(len(landmarks)):
                    if observed_hist[i, jj]:
                        idx = 3 + 2 * jj
                        print(f"  LM{jj}: ({mu_hist[i, idx]:.3f}, {mu_hist[i, idx + 1]:.3f})")
                    else:
                        print(f"  LM{jj}: not observed yet")

            update_live_plot(
                plotters, t, q_true_hist, mu_hist, mu_pred_hist,
                sigma_trace_hist, q_gt_hist, observed_hist, landmarks, i
            )

            viewer.sync()
            time.sleep(0.0)
            k += 1

    plt.close("all")

if __name__ == "__main__":
    main()