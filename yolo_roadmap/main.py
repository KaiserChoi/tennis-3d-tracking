import cv2
import pandas as pd
import numpy as np

def smooth_trajectory(df, max_gap=3, window=3):
    print(f">>> 姝ｅ湪鎵ц杞ㄨ抗棰勫鐞?(鎻掑€间笂闄? {max_gap}甯? 骞虫粦绐楀彛: {window}甯?...")
    df_smoothed = df.copy()
    df_smoothed = df_smoothed.sort_values('frame_index')
    df_smoothed.set_index('frame_index', inplace=True)

    df_smoothed['x'] = df_smoothed['x'].interpolate(method='linear', limit=max_gap)
    df_smoothed['y'] = df_smoothed['y'].interpolate(method='linear', limit=max_gap)

    df_smoothed['x'] = df_smoothed['x'].rolling(window=window, center=True, min_periods=1).mean()
    df_smoothed['y'] = df_smoothed['y'].rolling(window=window, center=True, min_periods=1).mean()
    df_smoothed.reset_index(inplace=True)

    return df_smoothed

def evaluate_bounces_fuzzy(lookup_table, window, angle_thresh, momentum_thresh, tolerance=2):
    candidate_bounces = set()
    frame_stats = {}

    valid_frames = [i for i, pt in enumerate(lookup_table) if pt is not None]

    for i in range(len(valid_frames)):
        curr_idx = valid_frames[i]

        if i < window or i >= len(valid_frames) - window:
            continue

        prev_idx = valid_frames[i - window]
        next_idx = valid_frames[i + window]

        if (curr_idx - prev_idx) > window * 3 or (next_idx - curr_idx) > window * 3:
            continue

        p_prev = lookup_table[prev_idx]
        p_curr = lookup_table[curr_idx]
        p_next = lookup_table[next_idx]

        v_in = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
        v_out = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])

        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)

        angle = 0.0
        y_reversal = False
        delta_v = 0.0

        if norm_in > 1e-5 and norm_out > 1e-5:
            cos_theta = np.clip(np.dot(v_in, v_out) / (norm_in * norm_out), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_theta))
            y_reversal = (v_in[1] > 0 and v_out[1] < 0)

            speed_in = norm_in / (curr_idx - prev_idx)
            speed_out = norm_out / (next_idx - curr_idx)
            delta_v = abs(speed_in - speed_out)

        frame_stats[curr_idx] = {
            'angle': angle,
            'y_reversal': y_reversal,
            'delta_v': delta_v,
            'angle_ok': angle >= angle_thresh,
            'y_ok': y_reversal,
            'mom_ok': delta_v >= momentum_thresh
        }

    for curr_idx, stats in frame_stats.items():
        if not stats['angle_ok']:
            continue

        local_y_ok = False
        local_mom_ok = False

        for j in range(curr_idx - tolerance, curr_idx + tolerance + 1):
            if j in frame_stats:
                if frame_stats[j]['y_ok']: local_y_ok = True
                if frame_stats[j]['mom_ok']: local_mom_ok = True

        if local_y_ok or local_mom_ok:
            candidate_bounces.add(curr_idx)

    bounces = {}
    sorted_bounces = sorted(list(candidate_bounces))
    if not sorted_bounces:
        return bounces, frame_stats

    cluster = [sorted_bounces[0]]
    for j in range(1, len(sorted_bounces)):
        if sorted_bounces[j] - cluster[-1] <= window * 2 + tolerance:
            cluster.append(sorted_bounces[j])
        else:
            best_idx = max(cluster, key=lambda idx: lookup_table[idx][1])
            bounces[best_idx] = lookup_table[best_idx]
            cluster = [sorted_bounces[j]]

    if cluster:
        best_idx = max(cluster, key=lambda idx: lookup_table[idx][1])
        bounces[best_idx] = lookup_table[best_idx]

    return bounces, frame_stats

def overlay_tracking_interactive(video_path, csv_path, original_res=(1920, 1080)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: unable to open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # 鑾峰彇鍘熷瑙嗛鍒嗚鲸鐜?
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    PANEL_W = 420 # 鍙充晶 X-Ray 鎺у埗鍙板搴?

    # =======================================================
    # 闃叉尋鍘嬭璁★細涓哄簳閮ㄧ殑 5 涓?Trackbar 棰勭暀浜嗗ぇ绾?200px 鐨勯珮搴?
    # =======================================================
    MAX_SCREEN_W = 1500
    MAX_SCREEN_H = 700 # 楂樺害鏀剁揣鍒?00锛屼繚璇佸嵆渚垮姞涓婃媺鏉嗕篃涓嶄細鎾戠垎 1080p 灞忓箷

    MAX_VIDEO_W = MAX_SCREEN_W - PANEL_W
    MAX_VIDEO_H = MAX_SCREEN_H

    scale_factor = min(MAX_VIDEO_W / orig_w, MAX_VIDEO_H / orig_h)
    if scale_factor > 1.0:
        scale_factor = 1.0

    video_disp_w = int(orig_w * scale_factor)
    video_disp_h = int(orig_h * scale_factor)

    total_w = video_disp_w + PANEL_W
    total_h = max(video_disp_h, 400)
    # =======================================================

    print("姝ｅ湪鍔犺浇杞ㄨ抗鏁版嵁...")
    df = pd.read_csv(csv_path)
    df = smooth_trajectory(df, max_gap=3, window=3)

    max_frame = int(df['frame_index'].max()) if not df.empty else 0
    total_frames = max(max_frame + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    csv_scale_x = orig_w / original_res[0]
    csv_scale_y = orig_h / original_res[1]

    lookup_table = [None] * total_frames
    df_valid = df.dropna(subset=['x', 'y'])
    for _, row in df_valid.iterrows():
        f_idx = int(row['frame_index'])
        if f_idx < total_frames:
            lookup_table[f_idx] = (int(float(row['x']) * csv_scale_x), int(float(row['y']) * csv_scale_y))

    window_name = 'Tennis Tuner (Pixel Perfect)'

    # =======================================================
    # 鏍稿績淇敼锛氫娇鐢?AUTOSIZE銆侽penCV 浼氫弗鏍间娇鐢ㄤ綘鐨勭敾甯冨ぇ灏?
    # 鐢诲竷澶栭潰浼氳嚜鍔ㄥ寘瑁规媺鏉嗭紝鏉滅粷鎷変几瑙嗛锛?
    # =======================================================
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # (绉婚櫎浜嗕細瀵艰嚧鎸ゅ帇鐨?cv2.resizeWindow)

    cv2.createTrackbar('Frame', window_name, 0, total_frames - 1, lambda x: None)
    cv2.createTrackbar('Angle Thr', window_name, 10, 90, lambda x: None)
    cv2.createTrackbar('Window Size', window_name, 3, 10, lambda x: None)
    cv2.createTrackbar('Momentum Thr', window_name, 15, 60, lambda x: None)
    cv2.createTrackbar('Sync Tol (f)', window_name, 2, 5, lambda x: None)

    delay = int(1000 / fps) if fps > 0 else 40
    frame_idx = 0
    paused = False

    COLOR_GREEN = (0, 255, 0)
    COLOR_PURPLE = (255, 0, 255)
    COLOR_YELLOW = (0, 255, 255)

    p_ang, p_win, p_mom, p_tol = -1, -1, -1, -1
    bounces_dict, frame_stats = {}, {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()

    def map_pt(pt):
        return (int(pt[0] * scale_factor), int(pt[1] * scale_factor))

    while True:
        tb_frame = cv2.getTrackbarPos('Frame', window_name)
        if tb_frame != frame_idx:
            frame_idx = tb_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            paused = True

        c_ang = max(1, cv2.getTrackbarPos('Angle Thr', window_name))
        c_win = max(1, cv2.getTrackbarPos('Window Size', window_name))
        c_mom = max(1, cv2.getTrackbarPos('Momentum Thr', window_name))
        c_tol = cv2.getTrackbarPos('Sync Tol (f)', window_name)

        if c_ang != p_ang or c_win != p_win or c_mom != p_mom or c_tol != p_tol:
            bounces_dict, frame_stats = evaluate_bounces_fuzzy(lookup_table, c_win, c_ang, c_mom, c_tol)
            p_ang, p_win, p_mom, p_tol = c_ang, c_win, c_mom, c_tol

        display_frame = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        if ret and frame is not None:
            # 瀹岀編绛夋瘮缂╂斁
            resized_video = cv2.resize(frame, (video_disp_w, video_disp_h), interpolation=cv2.INTER_AREA)
            display_frame[0:video_disp_h, 0:video_disp_w] = resized_video

        # 娓叉煋杞ㄨ抗
        history_pts = []
        for i in range(max(0, frame_idx - 30), frame_idx + 1):
            if i < total_frames and lookup_table[i] is not None:
                history_pts.append(map_pt(lookup_table[i]))
        if len(history_pts) > 1:
            for i in range(1, len(history_pts)):
                cv2.line(display_frame, history_pts[i-1], history_pts[i], COLOR_PURPLE, thickness=2)

        for b_idx, b_pos in bounces_dict.items():
            if 0 <= frame_idx - b_idx < 15:
                bp = map_pt(b_pos)
                cv2.circle(display_frame, bp, radius=12, color=COLOR_YELLOW, thickness=2)
                cv2.circle(display_frame, bp, radius=4, color=COLOR_YELLOW, thickness=-1)
                cv2.putText(display_frame, "BOUNCE!", (bp[0]+15, bp[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)

        current_pt = lookup_table[frame_idx] if frame_idx < total_frames else None
        if current_pt is not None:
            cp = map_pt(current_pt)
            cv2.circle(display_frame, cp, radius=6, color=COLOR_GREEN, thickness=-1)

        # UI闈㈡澘鑳屾櫙娓叉煋
        panel_x = video_disp_w
        panel_y = 0
        cv2.rectangle(display_frame, (panel_x, panel_y), (total_w, total_h), (15, 15, 15), -1)

        cv2.putText(display_frame, f"[ FUZZY X-RAY : Frame {frame_idx} ]", (panel_x + 10, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(display_frame, f"Rule: Angle(0f) & [Y-Rev(卤{c_tol}f) | Mom(卤{c_tol}f)]", (panel_x + 10, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

        ang_str, yrev_str, mom_str = "---", "---", "---"
        status_text = ">> NO BALL <<"
        status_color = (0, 0, 255)
        a_col, y_col, m_col = COLOR_GREEN, COLOR_GREEN, COLOR_GREEN

        if current_pt is not None:
            status_text = "[WARNING] Track Broken"
            status_color = (0, 165, 255)

        if frame_idx in frame_stats:
            stats = frame_stats[frame_idx]

            local_y_ok = False
            local_mom_peak = 0.0
            for j in range(frame_idx - c_tol, frame_idx + c_tol + 1):
                if j in frame_stats:
                    if frame_stats[j]['y_ok']: local_y_ok = True
                    if frame_stats[j]['delta_v'] > local_mom_peak:
                        local_mom_peak = frame_stats[j]['delta_v']

            local_mom_ok = (local_mom_peak >= c_mom)
            angle_ok = stats['angle_ok']
            is_bounce_candidate = angle_ok and (local_y_ok or local_mom_ok)
            is_true_peak = frame_idx in bounces_dict

            a_col = (255, 255, 255) if angle_ok else COLOR_GREEN
            y_col = (255, 255, 255) if local_y_ok else COLOR_GREEN
            m_col = (255, 255, 255) if local_mom_ok else COLOR_GREEN

            ang_str = f"{stats['angle']:.2f} deg"
            yrev_str = f"{local_y_ok} (Curr: {stats['y_reversal']})"
            mom_str = f"{local_mom_peak:.1f} px/f"

            if is_true_peak:
                status_text = ">> BOUNCE (NMS PEAK) <<"
                status_color = COLOR_YELLOW
            elif is_bounce_candidate:
                status_text = ">> CANDIDATE (WAIT NMS) <<"
                status_color = (0, 165, 255)
            else:
                status_text = ">> NO BOUNCE <<"
                status_color = (0, 0, 255)

        y_base = panel_y + 110
        line_spacing = 25

        cv2.putText(display_frame, f"Current Angle : {ang_str}", (panel_x + 10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, a_col, 1)
        cv2.putText(display_frame, f"Local Y-Rev   : {yrev_str}", (panel_x + 10, y_base + line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.5, y_col, 1)
        cv2.putText(display_frame, f"Local Mom Peak: {mom_str}", (panel_x + 10, y_base + line_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, m_col, 1)

        cv2.putText(display_frame, "-------------------------------------------", (panel_x + 10, y_base + line_spacing * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)

        cv2.putText(display_frame, f"Angle Thr     : >= {c_ang}.0 deg", (panel_x + 10, y_base + line_spacing * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)
        cv2.putText(display_frame, f"Momentum Thr  : >= {c_mom}.0 px/f", (panel_x + 10, y_base + line_spacing * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)
        cv2.putText(display_frame, f"Sync Tol      : +- {c_tol} Frames", (panel_x + 10, y_base + line_spacing * 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)

        cv2.putText(display_frame, status_text, (panel_x + 10, y_base + line_spacing * 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow(window_name, display_frame)

        wait_time = delay if not paused else 30
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key in [ord('a'), ord('A')]:
            paused = True; frame_idx = max(0, frame_idx - 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)
        elif key in [ord('d'), ord('D')]:
            paused = True; frame_idx = min(total_frames - 1, frame_idx + 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)

        if not paused and ret:
            frame_idx += 1
            if frame_idx < total_frames:
                ret, frame = cap.read()
                cv2.setTrackbarPos('Frame', window_name, frame_idx)
            else:
                paused = True; frame_idx -= 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO_FILE = 'video.mp4'
    CSV_FILE = 'tracknet_extracted.csv'

    overlay_tracking_interactive(VIDEO_FILE, CSV_FILE)
