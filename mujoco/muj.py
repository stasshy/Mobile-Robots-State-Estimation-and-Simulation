import time
from pathlib import Path
import mujoco
import mujoco.viewer
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


def random_landmarks(n=8, xmin=-4.5, xmax=4.5, ymin=-4.5, ymax=4.5, min_dist=2.0, seed=None):
    pts = []
    rng = np.random.default_rng(seed)
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


def simulate_slam(q0, N, dt, u, landmarks, R_motion, Q_meas,
                  fov_deg=180.0, max_range=7.0, seed=7):
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
            ok, r_true, b_true = is_landmark_in_fov(
                q, lm, fov_deg=fov_deg, max_range=max_range
            )
            if ok:
                z = np.array([r_true, b_true]) + rng.multivariate_normal(np.zeros(2), Q_meas)
                z[1] = wrap_angle(z[1])
                measurements.append((j, z))

        ekf.step(u, measurements)

        q_true[k] = q
        mu_hist[k] = ekf.mu.flatten()
        observed_hist[k] = np.array(ekf.observed)

    return q_true, mu_hist, observed_hist


def cube_xml_from_landmarks(landmarks, cube_half_sizes=None):
    """
    Uses your old obstacle logic:
    <body name="obstacle_{ID}" pos="{X} {Y} {Z}">
      <geom type="box" size="{SX} {SY} {SZ}" rgba="0.2 0.2 0.8 1"/>
    </body>
    """
    parts = []
    n = len(landmarks)

    if cube_half_sizes is None:
        rng = np.random.default_rng(123)
        cube_half_sizes = []
        for _ in range(n):
            sx = rng.uniform(0.12, 0.22)
            sy = rng.uniform(0.12, 0.22)
            sz = rng.uniform(0.12, 0.22)
            cube_half_sizes.append((sx, sy, sz))

    for i, (lm, szs) in enumerate(zip(landmarks, cube_half_sizes)):
        x, y = lm
        sx, sy, sz = szs
        z = sz
        parts.append(f"""
    <body name="obstacle_{i}" pos="{x:.3f} {y:.3f} {z:.3f}">
      <geom type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" rgba="0.2 0.2 0.8 1"/>
    </body>""")

    return "\n".join(parts), cube_half_sizes


def make_est_landmark_xml(n_landmarks):
    parts = []
    for j in range(n_landmarks):
        parts.append(f"""
    <body name="est_landmark_{j}" mocap="true" pos="1000 1000 0.45">
      <geom type="sphere" size="0.09" material="ekf_lm_mat" contype="0" conaffinity="0"/>
    </body>""")
    return "\n".join(parts)


def build_scene_from_landmarks(scene_template_path, generated_scene_path, landmarks):
    scene_template = scene_template_path.read_text(encoding="utf-8")

    if "<!-- OBSTACLES_GO_HERE -->" not in scene_template:
        raise ValueError("Λείπει το placeholder <!-- OBSTACLES_GO_HERE -->")

    true_cube_xml, cube_half_sizes = cube_xml_from_landmarks(landmarks)
    est_lm_xml = make_est_landmark_xml(len(landmarks))

    final_scene = scene_template.replace(
        "<!-- OBSTACLES_GO_HERE -->",
        true_cube_xml + "\n" + est_lm_xml
    )

    generated_scene_path.write_text(final_scene, encoding="utf-8")
    print(f"Generated scene saved to: {generated_scene_path}")
    return cube_half_sizes


def yaw_to_quat(th):
    return np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])


def set_free_body_pose(model, data, joint_name, x, y, th, z=0.0):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_adr = model.jnt_qposadr[jid]

    data.qpos[qpos_adr + 0] = x
    data.qpos[qpos_adr + 1] = y
    data.qpos[qpos_adr + 2] = z
    data.qpos[qpos_adr + 3:qpos_adr + 7] = yaw_to_quat(th)


def set_mocap_body(model, data, body_name, pos):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = model.body_mocapid[bid]
    data.mocap_pos[mocap_id] = np.array(pos)
    data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

def setup_live_plot(landmarks, fov_deg=180.0, max_range=7.0):
    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Live 2D EKF-SLAM")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(
        landmarks[:, 0], landmarks[:, 1],
        marker="s", color="blue", s=70, label="true landmarks"
    )

    (true_ln,) = ax.plot([], [], color="blue", lw=2, label="true noisy")
    (ekf_ln,) = ax.plot([], [], "--", color="purple", lw=2, label="EKF")
    (gt_ln,) = ax.plot([], [], "--", color="lightgray", lw=2, label="noiseless")

    (true_pt,) = ax.plot([], [], "o", color="blue")
    (ekf_pt,) = ax.plot([], [], "o", color="purple")
    (gt_pt,) = ax.plot([], [], "o", color="gray")

    (fov_l,) = ax.plot([], [], "-.", color="green", lw=1, label="FOV")
    (fov_r,) = ax.plot([], [], "-.", color="green", lw=1)

    est_pts = []
    for _ in range(len(landmarks)):
        (pt,) = ax.plot([], [], "D", color="magenta", ms=7)
        est_pts.append(pt)

    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    ax.set_xlim(xs.min() - 3, xs.max() + 3)
    ax.set_ylim(ys.min() - 3, ys.max() + 3)

    ax.legend(loc="upper right")
    fig.canvas.draw()
    fig.canvas.flush_events()

    artists = {
        "fig": fig,
        "ax": ax,
        "true_ln": true_ln,
        "ekf_ln": ekf_ln,
        "gt_ln": gt_ln,
        "true_pt": true_pt,
        "ekf_pt": ekf_pt,
        "gt_pt": gt_pt,
        "fov_l": fov_l,
        "fov_r": fov_r,
        "est_pts": est_pts,
        "half_fov": np.deg2rad(fov_deg / 2.0),
        "max_range": max_range,
    }
    return artists


def update_live_plot(plotters, q_true_hist, mu_hist, q_gt_hist, observed_hist, k):
    true_ln = plotters["true_ln"]
    ekf_ln = plotters["ekf_ln"]
    gt_ln = plotters["gt_ln"]
    true_pt = plotters["true_pt"]
    ekf_pt = plotters["ekf_pt"]
    gt_pt = plotters["gt_pt"]
    fov_l = plotters["fov_l"]
    fov_r = plotters["fov_r"]
    est_pts = plotters["est_pts"]
    half_fov = plotters["half_fov"]
    max_range = plotters["max_range"]
    fig = plotters["fig"]

    true_ln.set_data(q_true_hist[:k+1, 0], q_true_hist[:k+1, 1])
    ekf_ln.set_data(mu_hist[:k+1, 0], mu_hist[:k+1, 1])
    gt_ln.set_data(q_gt_hist[:k+1, 0], q_gt_hist[:k+1, 1])

    tx, ty, th = q_true_hist[k]
    ex, ey = mu_hist[k, 0], mu_hist[k, 1]
    gx, gy = q_gt_hist[k, 0], q_gt_hist[k, 1]

    true_pt.set_data([tx], [ty])
    ekf_pt.set_data([ex], [ey])
    gt_pt.set_data([gx], [gy])

    left_ang = th + half_fov
    right_ang = th - half_fov

    fov_l.set_data(
        [tx, tx + max_range * np.cos(left_ang)],
        [ty, ty + max_range * np.sin(left_ang)]
    )
    fov_r.set_data(
        [tx, tx + max_range * np.cos(right_ang)],
        [ty, ty + max_range * np.sin(right_ang)]
    )

    for j in range(len(est_pts)):
        if observed_hist[k, j]:
            idx = 3 + 2 * j
            lx = mu_hist[k, idx]
            ly = mu_hist[k, idx + 1]
            est_pts[j].set_data([lx], [ly])
        else:
            est_pts[j].set_data([], [])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)

def main():
    dt = 0.05
    T = 300.0
    N = int(T / dt)

    q0 = np.array([0.0, 0.0, 0.0])
    u = (0.6, 0.2)

    R_motion = np.diag([0.005**2, 0.005**2, np.deg2rad(0.2)**2])
    Q_meas = np.diag([0.01**2, np.deg2rad(0.25)**2])

    fov_deg = 180.0
    max_range = 7.0
    half_fov = np.deg2rad(fov_deg / 2.0)

    landmarks = random_landmarks(
        n=8,
        xmin=-4.5,
        xmax=4.5,
        ymin=-4.5,
        ymax=4.5,
        min_dist=2.0,
        seed=None,
    )

    base = Path(__file__).parent
    scene_template_path = base / "TB3-WafflePi scene.xml"
    generated_scene_path = base / "generated_scene.xml"

    build_scene_from_landmarks(
        scene_template_path,
        generated_scene_path,
        landmarks
    )

    q_true_hist, mu_hist, observed_hist = simulate_slam(
        q0, N, dt, u, landmarks, R_motion, Q_meas,
        fov_deg=fov_deg, max_range=max_range, seed=7
    )

    q_gt_hist = simulate_ground_truth_noiseless(q0, N, dt, u)

    plotters = setup_live_plot(
        landmarks,
        fov_deg=fov_deg,
        max_range=max_range,
    )

    model = mujoco.MjModel.from_xml_path(str(generated_scene_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
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

            ekf_x = mu_hist[i, 0]
            ekf_y = mu_hist[i, 1]
            ekf_th = mu_hist[i, 2]

            with viewer.lock():
                set_free_body_pose(
                    model, data, "base_joint",
                    q_true[0], q_true[1], q_true[2], z=0.0
                )

                set_mocap_body(model, data, "true_robot_marker", [q_true[0], q_true[1], 0.08])
                set_mocap_body(model, data, "ekf_robot_marker", [ekf_x, ekf_y, 0.10])
                set_mocap_body(model, data, "gt_robot_marker", [q_gt[0], q_gt[1], 0.12])

                left_ang = q_true[2] + half_fov
                right_ang = q_true[2] - half_fov

                set_mocap_body(
                    model, data, "fov_left_marker",
                    [
                        q_true[0] + max_range * np.cos(left_ang),
                        q_true[1] + max_range * np.sin(left_ang),
                        0.04,
                    ]
                )
                set_mocap_body(
                    model, data, "fov_right_marker",
                    [
                        q_true[0] + max_range * np.cos(right_ang),
                        q_true[1] + max_range * np.sin(right_ang),
                        0.04,
                    ]
                )

                if i % 10 == 0:
                    print(f"\nStep {i}")
                    print(f"True robot = ({q_true[0]:.3f}, {q_true[1]:.3f}, {q_true[2]:.3f})")
                    print(f"EKF  robot = ({ekf_x:.3f}, {ekf_y:.3f}, {ekf_th:.3f})")
                    print("Estimated landmarks:")

                for j in range(len(landmarks)):
                    if observed_hist[i, j]:
                        idx = 3 + 2 * j
                        lx = mu_hist[i, idx]
                        ly = mu_hist[i, idx + 1]
                        z_est = 0.45

                        set_mocap_body(model, data, f"est_landmark_{j}", [lx, ly, z_est])

                        if i % 10 == 0:
                            print(f"  LM{j}: ({lx:.3f}, {ly:.3f})")
                    else:
                        set_mocap_body(model, data, f"est_landmark_{j}", [1000, 1000, 0.45])

                        if i % 10 == 0:
                            print(f"  LM{j}: not observed yet")

                mujoco.mj_forward(model, data)

            update_live_plot(
                plotters,
                q_true_hist,
                mu_hist,
                q_gt_hist,
                observed_hist,
                i
            )

            viewer.sync()
            time.sleep(dt)
            k += 1

    plt.close("all")

if __name__ == "__main__":
    main()