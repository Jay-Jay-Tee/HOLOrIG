import cv2
import math
import time
import random
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# ---------------- WINDOW & HAND SETUP ----------------
win_w, win_h = 960, 720
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.7)

# ---------------- APP STATE ----------------
mode = "intro"      
shape_kind = "rings"
PINCH_THRESHOLD = 40

mode_change_time = time.time()

# ---------------- OPTIONAL SFX----------------
use_sfx = False
try:
    import pygame
    pygame.mixer.init()
    sfx_select = pygame.mixer.Sound("select.wav") 
    sfx_back   = pygame.mixer.Sound("back.wav")  
    sfx_grab   = pygame.mixer.Sound("grab.wav")    
    use_sfx = True
except Exception:
    sfx_select = sfx_back = sfx_grab = None
    use_sfx = False

def play_sfx(sound):
    if use_sfx and sound is not None:
        try:
            sound.play()
        except Exception:
            pass

# ---------------- 3D OBJECTS (GYRO RINGS + ORBIT CORE) ----------------

def get_shape_points(kind="rings"):
    if kind == "rings":
        pts = []
        edges = []
        N = 32
        r = 1.5

        start = 0
        for i in range(N):
            ang = 2 * math.pi * i / N
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            z = 0.0
            pts.append([x, y, z])
        for i in range(N):
            edges.append((start + i, start + (i + 1) % N))

        start = len(pts)
        for i in range(N):
            ang = 2 * math.pi * i / N
            x = 0.0
            y = r * math.cos(ang)
            z = r * math.sin(ang)
            pts.append([x, y, z])
        for i in range(N):
            edges.append((start + i, start + (i + 1) % N))

        start = len(pts)
        for i in range(N):
            ang = 2 * math.pi * i / N
            x = r * math.cos(ang)
            y = 0.0
            z = r * math.sin(ang)
            pts.append([x, y, z])
        for i in range(N):
            edges.append((start + i, start + (i + 1) % N))

        return np.array(pts, dtype=np.float32), edges

    else:
        s = 1.0
        t = 1.8
        inner = np.array([
            [-s, -s, -s],
            [ s, -s, -s],
            [ s,  s, -s],
            [-s,  s, -s],
            [-s, -s,  s],
            [ s, -s,  s],
            [ s,  s,  s],
            [-s,  s,  s],
        ], dtype=np.float32)
        outer = np.array([
            [-t, -t, -t],
            [ t, -t, -t],
            [ t,  t, -t],
            [-t,  t, -t],
            [-t, -t,  t],
            [ t, -t,  t],
            [ t,  t,  t],
            [-t,  t,  t],
        ], dtype=np.float32)

        pts = np.vstack([inner, outer])

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),

            (8, 9), (9, 10), (10, 11), (11, 8),
            (12,13), (13,14), (14,15), (15,12),
            (8,12), (9,13), (10,14), (11,15),

            (0, 8), (1, 9), (2,10), (3,11),
            (4,12), (5,13), (6,14), (7,15),
        ]
        return pts, edges

pts3d, edges3d = get_shape_points(shape_kind)
rot_x = 0.2
rot_y = -0.3
rot_z = 0.0
zoom = 1.2

pinching_3d = False
prev_ix_3d, prev_iy_3d = None, None
zoom_base_dist = None
zoom_base_zoom = None

ROT_SENS = 0.005
ZOOM_SENS = 0.003
SPIN_SENS = 0.02

def project_points(points3d, rx, ry, rz, zoom_factor):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0],[0, cx, -sx],[0, sx, cx]], np.float32)
    Ry = np.array([[cy, 0, sy],[0, 1, 0],[-sy, 0, cy]], np.float32)
    Rz = np.array([[cz, -sz, 0],[sz, cz, 0],[0, 0, 1]], np.float32)
    R = Rz @ Ry @ Rx

    rotated = (R @ points3d.T).T
    scale = 200 * zoom_factor
    rotated_scaled = rotated * scale

    cx_screen, cy_screen = win_w // 2, win_h // 2
    pts2d = []
    depths = []
    for x, y, z in rotated_scaled:
        sx2d = int(cx_screen + x)
        sy2d = int(cy_screen - y)
        pts2d.append((sx2d, sy2d))
        depths.append(z)
    return pts2d, depths

# ---------------- ROBOT ARM ----------------
base_x = win_w // 2
base_y = int(win_h * 0.82)
L1, L2 = 220, 170
gripper_closed = False

def ik_2link(target_x, target_y):
    dx = target_x - base_x
    dy = target_y - base_y
    d = math.hypot(dx, dy)
    max_reach = L1 + L2 - 10
    d = min(d, max_reach)

    cos2 = (d*d - L1*L1 - L2*L2) / (2*L1*L2)
    cos2 = max(-1.0, min(1.0, cos2))
    angle2 = math.acos(cos2)

    k1 = L1 + L2 * math.cos(angle2)
    k2 = L2 * math.sin(angle2)
    angle1 = math.atan2(dy, dx) - math.atan2(k2, k1)

    ex = base_x + L1 * math.cos(angle1)
    ey = base_y + L1 * math.sin(angle1)
    end_x = ex + L2 * math.cos(angle1 + angle2)
    end_y = ey + L2 * math.sin(angle1 + angle2)

    return angle1, angle2, int(ex), int(ey), int(end_x), int(end_y)

def draw_gripper(img, end_x, end_y, angle, closed):
    if closed:
        spread = 0.12; color = (0, 0, 255)
    else:
        spread = 0.5;  color = (0, 255, 0)

    claw_len = 45
    a1 = angle + spread
    a2 = angle - spread
    x1 = int(end_x + claw_len * math.cos(a1))
    y1 = int(end_y + claw_len * math.sin(a1))
    x2 = int(end_x + claw_len * math.cos(a2))
    y2 = int(end_y + claw_len * math.sin(a2))

    glow_color = (color[0] // 4, color[1] // 4, color[2] // 4)
    cv2.line(img, (end_x, end_y), (x1, y1), glow_color, 10)
    cv2.line(img, (end_x, end_y), (x2, y2), glow_color, 10)

    cv2.line(img, (end_x, end_y), (x1, y1), color, 4)
    cv2.line(img, (end_x, end_y), (x2, y2), color, 4)
    cv2.circle(img, (end_x, end_y), 10, color, cv2.FILLED)

def map_hand_to_workspace(ix, iy):
    min_x = int(win_w * 0.25)
    max_x = int(win_w * 0.75)
    min_y = int(win_h * 0.18)
    max_y = int(win_h * 0.78)
    x = int(ix / win_w * (max_x - min_x) + min_x)
    y = int(iy / win_h * (max_y - min_y) + min_y)
    x = max(min_x, min(max_x, x))
    y = max(min_y, min(max_y, y))
    return x, y, (min_x, min_y, max_x, max_y)

# ---------------- UI HELPERS ----------------

menu_cards = [
    (80, 220, 300, 420, "GYRO RINGS", "3d_rings"),
    (340, 220, 560, 420, "ORBIT CORE", "3d_core"),
    (620, 220, 840, 420, "ROBOT ARM", "arm"),
]

def draw_grid(img):
    step = 40
    for x in range(0, win_w, step):
        cv2.line(img, (x, 0), (x, win_h), (40, 40, 40), 1)
    for y in range(0, win_h, step):
        cv2.line(img, (0, y), (win_w, y), (40, 40, 40), 1)

def draw_hud(img, fps):
    cx, cy = win_w // 2, win_h // 2
    cv2.circle(img, (cx, cy), 18, (80, 80, 120), 1)
    cv2.line(img, (cx - 30, cy), (cx - 8, cy), (80, 80, 120), 1)
    cv2.line(img, (cx + 8, cy), (cx + 30, cy), (80, 80, 120), 1)
    cv2.line(img, (cx, cy - 30), (cx, cy - 8), (80, 80, 120), 1)
    cv2.line(img, (cx, cy + 8), (cx, cy + 30), (80, 80, 120), 1)

    cv2.putText(img, f"{int(fps)} FPS", (win_w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 2)

    cv2.putText(img, "ORIGO 2025 | NITC",
                (20, win_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 255), 2)
    cv2.putText(img, "Gesture Interaction System",
                (20, win_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

def draw_menu(img, t_since_mode):
    draw_grid(img)
    alpha = min(1.0, t_since_mode / 1.2)
    title_color = (0, int(255 * alpha), 255)

    cv2.putText(img, "ORIGO Holographic Hub",
                (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, title_color, 3)
    cv2.putText(img, "Pinch on a tile to launch a demo",
                (190, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)

    for (x1, y1, x2, y2, label, _) in menu_cards:
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(img, label,
                    (x1 + 20, y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img

def draw_back_button(img):
    x1, y1, x2, y2 = 20, 20, 120, 70
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(img, "BACK", (x1 + 15, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return (x1, y1, x2, y2)

def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2

# ---------------- INTRO SCREEN ----------------
def draw_intro(img, t_since_mode):
    draw_grid(img)
    cx, cy = win_w // 2, win_h // 2

    radius = 140 + 10 * math.sin(t_since_mode * 2)
    cv2.circle(img, (cx, cy), int(radius), (30, 80, 120), 2)

    alpha = min(1.0, t_since_mode / 1.5)
    main_color = (0, int(255 * alpha), 255)
    sub_color = (200, 255, 255)

    cv2.putText(img, "ORIGO 2025",
                (cx - 170, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, main_color, 3)
    cv2.putText(img, "Holographic Gesture Control",
                (cx - 260, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, sub_color, 2)
    cv2.putText(img, "by Joshua Jacob Thomas",
                (cx - 220, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 230, 255), 2)
    cv2.putText(img, "Pinch anywhere to continue",
                (cx - 220, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    return img

# ---------------- FPS TRACKING ----------------
prev_time = time.time()
fps = 0.0

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (win_w, win_h))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (win_w, win_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    hands, img = detector.findHands(frame, flipType=False)

    index_pt = None
    thumb_pt = None
    pinch_len = None
    fingers_state = None
    wrist_pt = None
    mid_knuckle_pt = None

    if hands:
        hand0 = hands[0]
        lm0 = hand0["lmList"]
        ix, iy, _ = lm0[8]
        tx, ty, _ = lm0[4]
        wx, wy, _ = lm0[0]
        mx, my, _ = lm0[9]

        index_pt = (ix, iy)
        thumb_pt = (tx, ty)
        wrist_pt = (wx, wy)
        mid_knuckle_pt = (mx, my)
        fingers_state = detector.fingersUp(hand0)

        cv2.circle(img, index_pt, 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, thumb_pt, 7, (255, 0, 255), cv2.FILLED)

        pinch_len, _, img = detector.findDistance(index_pt, thumb_pt, img)

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

    t_since_mode = now - mode_change_time

    status_text = ""
    legend = ""

    # -------- INTRO MODE --------
    if mode == "intro":
        img = draw_intro(img, t_since_mode)
        legend = "Pinch to continue | ESC = Quit"
        if index_pt is not None and pinch_len is not None and pinch_len < PINCH_THRESHOLD:
            mode = "menu"
            mode_change_time = now
            play_sfx(sfx_select)

    # -------- MENU MODE --------
    elif mode == "menu":
        img = draw_menu(img, t_since_mode)
        status_text = "MODE: MENU"
        legend = "Gestures: PINCH on any tile = Open that demo | ESC = Quit"

        if index_pt is not None and pinch_len is not None and pinch_len < PINCH_THRESHOLD:
            for (x1, y1, x2, y2, label, mode_name) in menu_cards:
                if point_in_rect(index_pt[0], index_pt[1], (x1, y1, x2, y2)):
                    if mode_name.startswith("3d"):
                        shape_kind = "rings" if mode_name == "3d_rings" else "core"
                        pts3d, edges3d = get_shape_points(shape_kind)
                        rot_x, rot_y, rot_z = 0.2, -0.3, 0.0
                        zoom = 1.2
                    mode = mode_name
                    mode_change_time = now
                    play_sfx(sfx_select)
                    break

    # -------- 3D MODES --------
    elif mode in ["3d_rings", "3d_core"]:
        draw_grid(img)
        back_rect = draw_back_button(img)
        mode_label = "GYRO RINGS" if mode == "3d_rings" else "ORBIT CORE"
        status_text = f"MODE: {mode_label}"

        if index_pt is not None and pinch_len is not None and pinch_len < PINCH_THRESHOLD:
            if point_in_rect(index_pt[0], index_pt[1], back_rect):
                mode = "menu"
                mode_change_time = now
                pinching_3d = False
                zoom_base_dist = None
                zoom_base_zoom = None
                play_sfx(sfx_back)
            else:
                if not pinching_3d:
                    pinching_3d = True
                    prev_ix_3d, prev_iy_3d = index_pt
                else:
                    dx = index_pt[0] - prev_ix_3d
                    dy = index_pt[1] - prev_iy_3d
                    rot_y += dx * ROT_SENS
                    rot_x += dy * ROT_SENS
                    prev_ix_3d, prev_iy_3d = index_pt
        else:
            pinching_3d = False
            prev_ix_3d, prev_iy_3d = None, None

        if wrist_pt is not None and mid_knuckle_pt is not None:
            dx_tilt = mid_knuckle_pt[0] - wrist_pt[0]
            if abs(dx_tilt) > 25:
                direction = 1 if dx_tilt > 0 else -1
                rot_z += direction * SPIN_SENS

        if len(hands) >= 2:
            h1, h2 = hands[:2]
            i1 = h1["lmList"][8]
            i2 = h2["lmList"][8]
            d = math.hypot(i2[0] - i1[0], i2[1] - i1[1])
            if zoom_base_dist is None:
                zoom_base_dist = d
                zoom_base_zoom = zoom
            else:
                delta = d - zoom_base_dist
                zoom = zoom_base_zoom + delta * ZOOM_SENS
                zoom = max(0.4, min(2.5, zoom))
        else:
            zoom_base_dist = None
            zoom_base_zoom = None

        pts2d, depths = project_points(pts3d, rot_x, rot_y, rot_z, zoom)


        cv2.circle(img, (win_w//2, win_h//2), int(130*zoom), (20, 20, 60), 2)

        base_color = (0, 255, 255) if mode == "3d_rings" else (0, 200, 255)
        for a, b in edges3d:
            x1, y1 = pts2d[a]
            x2, y2 = pts2d[b]
            z_avg = (depths[a] + depths[b]) / 2.0
            depth_factor = 1.0 / (1.0 + abs(z_avg) / 600.0)
            depth_factor = max(0.2, min(1.0, depth_factor))
            col = (
                int(base_color[0] * depth_factor),
                int(base_color[1] * depth_factor),
                int(base_color[2] * depth_factor),
            )
            glow_color = (col[0] // 4, col[1] // 4, col[2] // 4)
            cv2.line(img, (x1, y1), (x2, y2), glow_color, 6)
            cv2.line(img, (x1, y1), (x2, y2), col, 2)

        for x, y in pts2d:
            cv2.circle(img, (x, y), 3, (255, 255, 255), cv2.FILLED)

        legend = "Gestures: PINCH+DRAG = Rotate | Two Hands = Zoom | Palm Tilt = Spin | PINCH on BACK = Menu"

    # -------- ROBOT ARM MODE --------
    elif mode == "arm":
        draw_grid(img)
        back_rect = draw_back_button(img)
        status_text = "MODE: ROBOT ARM"

        target_x, target_y = base_x, int(win_h * 0.4)
        workspace_box = (int(win_w * 0.25), int(win_h * 0.18),
                         int(win_w * 0.75), int(win_h * 0.78))

        if index_pt is not None:
            tx, ty, workspace_box = map_hand_to_workspace(index_pt[0], index_pt[1])
            target_x, target_y = tx, ty

            if pinch_len is not None and pinch_len < PINCH_THRESHOLD:
                if point_in_rect(index_pt[0], index_pt[1], back_rect):
                    mode = "menu"
                    mode_change_time = now
                    play_sfx(sfx_back)
                else:
                    if not gripper_closed:
                        play_sfx(sfx_grab)
                    gripper_closed = True
            elif fingers_state is not None and sum(fingers_state) >= 4:
                gripper_closed = False

        angle1, angle2, ex, ey, end_x, end_y = ik_2link(target_x, target_y)

        cv2.rectangle(img,
                      (base_x - 80, base_y + 10),
                      (base_x + 80, base_y + 35),
                      (30, 30, 30), -1)
        cv2.rectangle(img,
                      (base_x - 80, base_y + 10),
                      (base_x + 80, base_y + 35),
                      (0, 255, 255), 2)

        cv2.line(img, (base_x, base_y + 10), (base_x, base_y - 40),
                 (80, 80, 80), 8)

        cv2.line(img, (base_x, base_y), (ex, ey), (80, 40, 0), 18)
        cv2.line(img, (ex, ey), (end_x, end_y), (0, 60, 40), 18)
        cv2.line(img, (base_x, base_y), (ex, ey), (255, 120, 0), 10)
        cv2.line(img, (ex, ey), (end_x, end_y), (0, 220, 180), 10)

        cv2.circle(img, (base_x, base_y), 16, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (ex, ey), 14, (230, 230, 230), cv2.FILLED)
        cv2.circle(img, (end_x, end_y), 12, (255, 255, 255), cv2.FILLED)

        x1, y1, x2, y2 = workspace_box
        center = (base_x, base_y)
        cv2.ellipse(img, center,
                    (L1 + L2 - 5, L1 + L2 - 5),
                    0, 210, 330, (60, 60, 100), 1)

        draw_gripper(img, end_x, end_y, angle1 + angle2, gripper_closed)

        legend = "Gestures: Move Hand = Move Arm | PINCH = Close Gripper | Open Palm = Open Gripper | PINCH on BACK = Menu"

    grade = np.full_like(img, (20, 40, 80))
    img = cv2.addWeighted(img, 0.88, grade, 0.12, 0)

    draw_hud(img, fps)

    if status_text:
        cv2.putText(img, status_text,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if legend:
        cv2.putText(img, legend,
                    (20, win_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("ORIGO Multi-Demo Holographic Hub", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
