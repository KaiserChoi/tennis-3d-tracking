import cv2
import numpy as np
import os
import time
import math
from collections import deque
from ultralytics import YOLO

# =====================================================================
# 🛠️ [全局配置中心] - 所有核心参数调整都在这里完成
# =====================================================================
class Config:
    # ---------------------------------------------------------
    # 1. 文件与模型路径
    # ---------------------------------------------------------
    BASE_PATH = r"D:\Work\.Master Folders\MR_ShuttleStrike\yolo_tennis"
    MODEL_BALL = os.path.join(BASE_PATH, "runs", "detect", "yolo11_tennis_recall_optimized_v3-2", "weights", "best.pt")
    MODEL_PERSON = "yolo11n.pt"  
    VIDEO_INPUT = os.path.join(BASE_PATH, "test_4.mp4")

    # ---------------------------------------------------------
    # 2. YOLO 双检测器配置
    # ---------------------------------------------------------
    CONF_BALL = 0.15         
    IMGSZ_BALL = 960         
    CONF_PERSON = 0.15       
    IMGSZ_PERSON = 960       

    # ---------------------------------------------------------
    # 3. 前端视窗固定参数 
    # ---------------------------------------------------------
    DISPLAY_HEIGHT = 768     
    MINIMAP_RATIO = 0.3      
    MINIMAP_MARGIN = 6.0     

    # ---------------------------------------------------------
    # 4. 追踪与死球黑洞控制
    # ---------------------------------------------------------
    TRK_SEARCH_WINDOW = 10   
    TRK_CREATE_THR = 70.0    
    TRK_MAX_HISTORY = 60     
    TRK_EVAL_WINDOW = 7      
    TRK_STATIC_THR = 1.0     
    TRK_STATIC_RADIUS = 20.0 
    TRK_ZONE_PERSIST = 50    

    # ---------------------------------------------------------
    # 5. 全局轨迹缝合参数
    # ---------------------------------------------------------
    ENABLE_STITCHING = True  
    STITCH_MAX_GAP = 35      
    STITCH_MAX_ANGLE = 40.0  
    STITCH_MIN_SPEED = 2.0   

    # ---------------------------------------------------------
    # 6. 弹跳分析参数
    # ---------------------------------------------------------
    BOUNCE_WINDOW = 3        
    BOUNCE_ANG_THR = 10      
    BOUNCE_MOM_THR = 15      
    BOUNCE_TOLERANCE = 2     

    # ---------------------------------------------------------
    # 7. 全局时空联合清洗
    # ---------------------------------------------------------
    CLEAN_TIME_FRAMES = 25   
    CLEAN_SPACE_METERS = 1.5 

    # ---------------------------------------------------------
    # 8. ⚡ 击球分离检测 (Hit Separation)
    # ---------------------------------------------------------
    HIT_ANGLE_THR = 45.0      
    HIT_DIST_PX_NET = 100     
    HIT_DIST_PX_BASE = 250    
    
    TOP_HIT_LOOKBACK_FRAMES = 50  
    HIT_DIST_PX_TOP_MAX = 50     
    
    ROI_SIDE_MARGIN = 2.0     
    ROI_NET_MARGIN = 1.5      

    # ---------------------------------------------------------
    # 9. 网球固定基线跨网测速 (Speed Trap)
    # ---------------------------------------------------------
    NET_OFFSET_PX_DEFAULT = 50       
    SPEED_LINE_OFFSET_DEFAULT = 150  
    SPEED_COEF_UP   = 1.5     
    SPEED_COEF_DOWN = 1.5     

    # ---------------------------------------------------------
    # 10. 全局色彩主题 (B, G, R)
    # ---------------------------------------------------------
    VIS_RESIDUAL_FRAMES = 15 
    C_MOV_TRACK    = (255, 0, 255) 
    C_MOV_STITCHED = (0, 165, 255) 
    C_MOV_BOX      = (0, 255, 255) 
    C_STA_BOX      = (0, 0, 200)   
    C_BOUNCE       = (0, 255, 255) 
    C_HIT          = (0, 165, 255) 
    C_OUT_BOUNDS   = (0, 0, 255)   
    C_PERSON_BOX   = (0, 255, 100) 
    C_NET_LINE     = (255, 100, 255) 
    C_SPEED_LINE   = (255, 255, 0)   

    # ---------------------------------------------------------
    # 11. ⚡ 视频录制配置 (Video Recording)
    # ---------------------------------------------------------
    SAVE_VIDEO = False                           # [一键控制] 是否录制并保存最终的检测视频
    OUTPUT_VIDEO_NAME = "output_tracked_tape_4.mp4"    # 保存的文件名，将自动存放在 BASE_PATH 目录下


# =====================================================================
# [模块 1: 物理球场畸变与双基线测速标定器] 
# =====================================================================
class CourtCalibrator:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.src_pts = []
        
        self.dst_pts = np.array([
            [-4.115, 11.885],   
            [-4.115, -11.885],  
            [4.115, -11.885],   
            [4.115, 11.885]     
        ], dtype=np.float32)

        self.court_lines_3d = [
            ([-4.115, -11.885], [-4.115, 11.885]), 
            ([4.115, -11.885], [4.115, 11.885]),   
            ([-5.485, -11.885], [-5.485, 11.885]), 
            ([5.485, -11.885], [5.485, 11.885]),   
            ([-5.485, -11.885], [5.485, -11.885]), 
            ([-5.485, 11.885], [5.485, 11.885]),   
            ([-4.115, -6.4], [4.115, -6.4]),       
            ([-4.115, 6.4], [4.115, 6.4]),         
            ([0, -6.4], [0, 6.4]),                 
            ([-5.485, 0], [5.485, 0])              
        ]
        
        focal_length = self.w * 0.8 
        self.K = np.array([[focal_length, 0, self.w/2], [0, focal_length, self.h/2], [0, 0, 1]], dtype=np.float32)
        self.D = np.zeros(5, dtype=np.float32)
        
        self.map1, self.map2 = None, None
        self.H_real_to_pixel = None
        self.H_pixel_to_real = None
        
        self.net_offset_px = Config.NET_OFFSET_PX_DEFAULT
        self.speed_line_offset_px = Config.SPEED_LINE_OFFSET_DEFAULT
        
        self.net_line_eq = None 
        self.speed_line_eq = None
        self.static_minimap = None 

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.src_pts) < 4:
                disp_scale = Config.DISPLAY_HEIGHT / self.h
                self.src_pts.append([int(x / disp_scale), int(y / disp_scale)])

    def _draw_lines(self, img):
        if self.H_real_to_pixel is None: return
        for pt1_real, pt2_real in self.court_lines_3d:
            is_net = (pt1_real[1] == 0 and pt2_real[1] == 0) 
            pts_real = np.array([pt1_real, pt2_real], dtype=np.float32).reshape(-1, 1, 2)
            pts_pixel = cv2.perspectiveTransform(pts_real, self.H_real_to_pixel)
            
            if is_net:
                p1_net = [pts_pixel[0][0][0], pts_pixel[0][0][1] - self.net_offset_px]
                p2_net = [pts_pixel[1][0][0], pts_pixel[1][0][1] - self.net_offset_px]
                l_net = np.cross([p1_net[0], p1_net[1], 1], [p2_net[0], p2_net[1], 1])
                norm_net = np.linalg.norm(l_net[:2])
                if norm_net > 0: l_net = l_net / norm_net
                self.net_line_eq = l_net 
                
                if l_net[1] != 0:
                    y0 = int(-l_net[2]/l_net[1]) 
                    yw = int(-(l_net[0]*self.w + l_net[2])/l_net[1]) 
                    cv2.line(img, (0, y0), (self.w, yw), Config.C_NET_LINE, max(2, int(3*(self.h/720))))
                    cv2.putText(img, "NET TAPE", (10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.C_NET_LINE, 2)

                p1_spd = [p1_net[0], p1_net[1] + self.speed_line_offset_px]
                p2_spd = [p2_net[0], p2_net[1] + self.speed_line_offset_px]
                l_spd = np.cross([p1_spd[0], p1_spd[1], 1], [p2_spd[0], p2_spd[1], 1])
                norm_spd = np.linalg.norm(l_spd[:2])
                if norm_spd > 0: l_spd = l_spd / norm_spd
                self.speed_line_eq = l_spd
                
                if l_spd[1] != 0:
                    y0_s = int(-l_spd[2]/l_spd[1]) 
                    yw_s = int(-(l_spd[0]*self.w + l_spd[2])/l_spd[1]) 
                    cv2.line(img, (0, y0_s), (self.w, yw_s), Config.C_SPEED_LINE, max(2, int(3*(self.h/720))))
                    cv2.putText(img, "SPEED LINE", (10, y0_s + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.C_SPEED_LINE, 2)
            else:
                p1 = (int(pts_pixel[0][0][0]), int(pts_pixel[0][0][1]))
                p2 = (int(pts_pixel[1][0][0]), int(pts_pixel[1][0][1]))
                cv2.line(img, p1, p2, (255, 255, 0), max(2, int(3*(self.h/720))))

    def calibrate(self, first_frame):
        window_name = "Phase 1: Setup Distortion & Court"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) 
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        cv2.createTrackbar('Distortion', window_name, 500, 1000, lambda x: None)
        cv2.createTrackbar('Net Offset(px)', window_name, Config.NET_OFFSET_PX_DEFAULT, 300, lambda x: None)
        cv2.createTrackbar('Speed Line Pos', window_name, Config.SPEED_LINE_OFFSET_DEFAULT + 300, 600, lambda x: None)

        print("\n" + "="*50)
        print(">>> 标定向导 <<<")
        print("1. 滑动 [Distortion] 消除画面边缘弯曲。")
        print("2. 顺时针点击 4 个单打底线交角。")
        print("3. 滑动 [Net Offset] 与 [Speed Line Pos] 调节双测速基线。")
        print("4. 确认无误，按 ENTER 发车。")
        print("="*50 + "\n")

        prev_k1_val = 500
        disp_scale = Config.DISPLAY_HEIGHT / self.h
        target_w = int(self.w * disp_scale)

        while True:
            k1_val = cv2.getTrackbarPos('Distortion', window_name)
            self.net_offset_px = cv2.getTrackbarPos('Net Offset(px)', window_name)
            self.speed_line_offset_px = cv2.getTrackbarPos('Speed Line Pos', window_name) - 300

            if k1_val != prev_k1_val:
                self.src_pts = []
                prev_k1_val = k1_val

            k1 = (k1_val - 500) / 1000.0 
            self.D[0] = k1
            self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (self.w, self.h), cv2.CV_32FC1)
            display_orig = cv2.remap(first_frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

            if len(self.src_pts) == 4:
                src_arr = np.array(self.src_pts, dtype=np.float32)
                self.H_real_to_pixel, _ = cv2.findHomography(self.dst_pts, src_arr)
                self.H_pixel_to_real, _ = cv2.findHomography(src_arr, self.dst_pts)
                self._draw_lines(display_orig) 
                
            display_view = cv2.resize(display_orig, (target_w, Config.DISPLAY_HEIGHT))

            cv2.putText(display_view, "[1] Adjust Distortion  [2] Click 4 Corners  [3] Adjust Lines", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            for i, pt in enumerate(self.src_pts):
                vx, vy = int(pt[0] * disp_scale), int(pt[1] * disp_scale)
                cv2.circle(display_view, (vx, vy), 5, (0, 0, 255), -1)
                if i > 0:
                    px, py = int(self.src_pts[i-1][0] * disp_scale), int(self.src_pts[i-1][1] * disp_scale)
                    cv2.line(display_view, (px, py), (vx, vy), (0, 255, 0), 2)
            
            if len(self.src_pts) == 4:
                px, py = int(self.src_pts[3][0] * disp_scale), int(self.src_pts[3][1] * disp_scale)
                vx, vy = int(self.src_pts[0][0] * disp_scale), int(self.src_pts[0][1] * disp_scale)
                cv2.line(display_view, (px, py), (vx, vy), (0, 255, 0), 2)
                
                cv2.rectangle(display_view, (10, 60), (550, 100), (0, 0, 0), -1)
                cv2.putText(display_view, "Align Pink/Cyan Lines, then press [ENTER]!", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, display_view)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(self.src_pts) == 4: break
            elif key == ord('r') or key == ord('R'): self.src_pts = []

        cv2.destroyWindow(window_name)

    def draw_virtual_court(self, frame):
        self._draw_lines(frame)

    def pixel_to_real(self, px, py):
        if self.H_pixel_to_real is None: return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        real_pt = cv2.perspectiveTransform(pt, self.H_pixel_to_real)
        return real_pt[0][0][0], real_pt[0][0][1]

    def build_static_minimap(self, target_h, target_w):
        self.mm_h = target_h
        self.mm_w = target_w
        
        court_len, court_wid = 23.77, 10.97
        margin = Config.MINIMAP_MARGIN 
        self.mm_scale = min(target_h / (court_len + margin * 2), target_w / (court_wid + margin * 2))
        
        minimap = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        minimap[:] = (40, 70, 40) 

        cv2.rectangle(minimap, self._to_map(-5.485, -11.885), self._to_map(5.485, 11.885), (80, 110, 140), -1)
        for pt1, pt2 in self.court_lines_3d:
            cv2.line(minimap, self._to_map(*pt1), self._to_map(*pt2), (255, 255, 255), 2)
            
        cv2.putText(minimap, "2D TRACKING MAP", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.circle(minimap, (30, 80), 6, Config.C_BOUNCE, -1); cv2.putText(minimap, "IN (Bounce)", (45, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.circle(minimap, (30, 110), 6, Config.C_OUT_BOUNDS, -1); cv2.putText(minimap, "OUT (Bounce)", (45, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        cv2.drawMarker(minimap, (30, 140), Config.C_HIT, markerType=cv2.MARKER_STAR, markerSize=12, thickness=2)
        cv2.putText(minimap, "HIT (Player)", (45, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        panel_w = target_w - 30
        panel_h = 105
        panel_x, panel_y = 15, target_h - panel_h - 15 
        
        cv2.rectangle(minimap, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 50, 30), -1)
        cv2.rectangle(minimap, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 120, 80), 1)
        cv2.putText(minimap, "SPEED (km/h)", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        cv2.putText(minimap, "Mode: Two-Line Trap", (panel_x + 175, panel_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)
        cv2.line(minimap, (panel_x + 5, panel_y + 35), (panel_x + panel_w - 5, panel_y + 35), (80, 120, 80), 1)
        
        cv2.putText(minimap, f"UP   :", (panel_x + 10, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(minimap, f"(x{Config.SPEED_COEF_UP:.2f})", (panel_x + 180, panel_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(minimap, f"DOWN :", (panel_x + 10, panel_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(minimap, f"(x{Config.SPEED_COEF_DOWN:.2f})", (panel_x + 180, panel_y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        self.static_minimap = minimap

    def _to_map(self, rx, ry):
        mx = int(self.mm_w / 2 + rx * self.mm_scale)
        my = int(self.mm_h / 2 + ry * self.mm_scale) 
        return (mx, my)

    def generate_minimap(self, bounces_history, global_crossings, global_hits):
        minimap = self.static_minimap.copy()
        
        panel_x, panel_y = 15, self.mm_h - 105 - 15 
        cv2.putText(minimap, f"{global_crossings.get('bottom_up', 0.0):05.1f}", (panel_x + 95, panel_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.C_SPEED_LINE, 2)
        cv2.putText(minimap, f"{global_crossings.get('top_down', 0.0):05.1f}", (panel_x + 95, panel_y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.C_SPEED_LINE, 2)

        for b_id, b_data in bounces_history.items():
            mx, my = self._to_map(b_data['rx'], b_data['ry'])
            color = Config.C_BOUNCE if b_data['is_in'] else Config.C_OUT_BOUNDS 
            cv2.circle(minimap, (mx, my), 6, color, -1)
            cv2.circle(minimap, (mx, my), 7, (0, 0, 0), 1) 
            
        for h_id, h_data in global_hits.items():
            mx, my = self._to_map(h_data['rx'], h_data['ry'])
            cv2.drawMarker(minimap, (mx, my), Config.C_HIT, markerType=cv2.MARKER_STAR, markerSize=14, thickness=2)
            
        return minimap


# =====================================================================
# [模块 2: 物理轨迹缝合器] 
# =====================================================================
class TrajectoryStitcher:
    def __init__(self, max_time_gap=Config.STITCH_MAX_GAP, max_angle_deg=Config.STITCH_MAX_ANGLE, min_speed=Config.STITCH_MIN_SPEED):
        self.max_time_gap = max_time_gap
        self.max_angle_deg = max_angle_deg
        self.min_speed = min_speed

    def _get_velocity_vector(self, history_list, mode='out', sample_frames=5):
        if len(history_list) < 2: return None, None
        pts = history_list[-sample_frames:] if mode == 'out' else history_list[:sample_frames]
        if len(pts) < 2: return None, None
        start_frame, start_det = pts[0][:2]
        end_frame, end_det = pts[-1][:2]
        frame_span = end_frame - start_frame
        if frame_span <= 0: return None, None
        vec = np.array(end_det[0:2]) - np.array(start_det[0:2])
        speed = math.hypot(vec[0], vec[1]) / frame_span
        return vec, speed

    def _angle_between(self, v1, v2):
        norm1, norm2 = math.hypot(v1[0], v1[1]), math.hypot(v2[0], v2[1])
        if norm1 == 0 or norm2 == 0: return 180.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def stitch_queues(self, queues):
        stitched_queues = []
        skip_ids = set()
        active_queues = [q for q in queues if not q['is_static']]
        match_candidates = []

        for q_a in active_queues:
            for q_b in active_queues:
                if q_a['id'] == q_b['id']: continue
                hist_a, hist_b = list(q_a['history']), list(q_b['history'])
                time_a_end, time_b_start = hist_a[-1][0], hist_b[0][0]
                time_gap = time_b_start - time_a_end
                
                if 0 < time_gap <= self.max_time_gap:
                    v_out, speed_a = self._get_velocity_vector(hist_a, mode='out')
                    v_in, speed_b  = self._get_velocity_vector(hist_b, mode='in')
                    if v_out is None or v_in is None: continue
                    if speed_a < self.min_speed or speed_b < self.min_speed: continue
                        
                    vec_displacement = np.array(hist_b[0][1][0:2]) - np.array(hist_a[-1][1][0:2])
                    angle_a_to_d = self._angle_between(v_out, vec_displacement)
                    angle_b_to_d = self._angle_between(v_in, vec_displacement)
                    angle_a_to_b = self._angle_between(v_out, v_in)
                    
                    if (angle_a_to_d <= self.max_angle_deg and angle_b_to_d <= self.max_angle_deg and angle_a_to_b <= self.max_angle_deg):
                        cost = angle_a_to_d + angle_b_to_d + angle_a_to_b
                        match_candidates.append((cost, q_a, q_b))

        match_candidates.sort(key=lambda x: x[0])
        for cost, q_a, q_b in match_candidates:
            if q_a['id'] in skip_ids or q_b['id'] in skip_ids: continue
            hist_a, hist_b = list(q_a['history']), list(q_b['history'])
            frame_a_end, det_a = hist_a[-1][:2]
            frame_b_start, det_b = hist_b[0][:2]
            gap = frame_b_start - frame_a_end
            interpolated_pts = []
            
            if gap > 1:
                for i in range(1, gap):
                    alpha = i / gap
                    interp_frame = frame_a_end + i
                    interp_det = [det_a[j] + alpha * (det_b[j] - det_a[j]) for j in range(6)]
                    interpolated_pts.append((interp_frame, interp_det, True))
            
            new_history = hist_a + interpolated_pts + hist_b
            q_a['history'] = deque(new_history, maxlen=q_a['history'].maxlen)
            skip_ids.add(q_b['id'])
            skip_ids.add(q_a['id'])

        for q in queues:
            if q['id'] not in skip_ids or q in [c[1] for c in match_candidates]:
                if q['id'] not in [c[2]['id'] for c in match_candidates if c[2]['id'] in skip_ids]:
                    stitched_queues.append(q)
        return stitched_queues


# =====================================================================
# [模块 3: 轨迹分析器] 
# =====================================================================
class TrajectoryAnalyzer:
    def __init__(self, 
                 window=Config.BOUNCE_WINDOW, 
                 angle_thresh=Config.BOUNCE_ANG_THR, 
                 momentum_thresh=Config.BOUNCE_MOM_THR, 
                 tolerance=Config.BOUNCE_TOLERANCE):
        self.window = window
        self.angle_thresh = angle_thresh
        self.momentum_thresh = momentum_thresh
        self.tolerance = tolerance

    def smooth(self, pts):
        smoothed = []
        half_w = 1 
        n = len(pts)
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            subset = pts[start:end]
            avg_cx = sum(p[1][0] for p in subset) / len(subset)
            avg_cy = sum(p[1][1] for p in subset) / len(subset)
            new_det = list(pts[i][1])
            new_det[0] = avg_cx
            new_det[1] = avg_cy
            smoothed.append((pts[i][0], new_det, pts[i][2]))
        return smoothed

    def detect_bounces(self, pts):
        if len(pts) < self.window * 2 + 1: return []
            
        lookup = {p[0]: (p[1][0], p[1][1]) for p in pts}
        valid_frames = [p[0] for p in pts]
        candidate_bounces = set()
        frame_stats = {}
        
        for i in range(len(valid_frames)):
            curr_idx = valid_frames[i]
            if i < self.window or i >= len(valid_frames) - self.window: continue
            prev_idx = valid_frames[i - self.window]
            next_idx = valid_frames[i + self.window]
            if (curr_idx - prev_idx) > self.window * 3 or (next_idx - curr_idx) > self.window * 3: continue
                
            p_prev, p_curr, p_next = lookup[prev_idx], lookup[curr_idx], lookup[next_idx]
            v_in = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
            v_out = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])
            
            norm_in, norm_out = math.hypot(v_in[0], v_in[1]), math.hypot(v_out[0], v_out[1])
            angle, delta_v, y_reversal = 0.0, 0.0, False
            
            if norm_in > 1e-5 and norm_out > 1e-5:
                cos_theta = np.clip(np.dot(v_in, v_out) / (norm_in * norm_out), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_theta))
                y_reversal = (v_in[1] > 0 and v_out[1] < 0)
                speed_in = norm_in / (curr_idx - prev_idx)
                speed_out = norm_out / (next_idx - curr_idx)
                delta_v = abs(speed_in - speed_out)
                
            frame_stats[curr_idx] = {'angle': angle, 'angle_ok': angle >= self.angle_thresh, 'y_ok': y_reversal, 'mom_ok': delta_v >= self.momentum_thresh}

        for curr_idx, stats in frame_stats.items():
            if not stats['angle_ok']: continue
            local_y_ok = False; local_mom_ok = False
            for j in range(curr_idx - self.tolerance, curr_idx + self.tolerance + 1):
                if j in frame_stats:
                    if frame_stats[j]['y_ok']: local_y_ok = True
                    if frame_stats[j]['mom_ok']: local_mom_ok = True
            if local_y_ok or local_mom_ok: candidate_bounces.add(curr_idx)
                
        bounces_raw = []
        sorted_bounces = sorted(list(candidate_bounces))
        if not sorted_bounces: return bounces_raw
            
        cluster = [sorted_bounces[0]]
        for j in range(1, len(sorted_bounces)):
            if sorted_bounces[j] - cluster[-1] <= self.window * 2 + self.tolerance:
                cluster.append(sorted_bounces[j])
            else:
                best_idx = max(cluster, key=lambda idx: lookup[idx][1])
                bounces_raw.append((best_idx, lookup[best_idx][0], lookup[best_idx][1], frame_stats[best_idx]['angle']))
                cluster = [sorted_bounces[j]]
        if cluster:
            best_idx = max(cluster, key=lambda idx: lookup[idx][1])
            bounces_raw.append((best_idx, lookup[best_idx][0], lookup[best_idx][1], frame_stats[best_idx]['angle']))
            
        return bounces_raw

    def detect_net_crossings(self, pts, calibrator):
        crossings = []
        if len(pts) < 2 or calibrator.net_line_eq is None or calibrator.speed_line_eq is None: 
            return crossings
        
        l_net = calibrator.net_line_eq 
        l_spd = calibrator.speed_line_eq
        
        def check_crossing(p1, p2, line_eq):
            val1 = line_eq[0]*p1[1][0] + line_eq[1]*p1[1][1] + line_eq[2]
            val2 = line_eq[0]*p2[1][0] + line_eq[1]*p2[1][1] + line_eq[2]
            return val1 * val2 <= 0 and val1 != val2

        for i in range(1, len(pts)):
            p_prev, p_curr = pts[i-1], pts[i]
            
            if check_crossing(p_prev, p_curr, l_net):
                direction = "bottom_up" if p_prev[1][1] > p_curr[1][1] else "top_down"
                spd_cross_idx = -1
                search_range = 25 
                
                for j in range(i, max(0, i - search_range), -1):
                    if j - 1 >= 0 and check_crossing(pts[j-1], pts[j], l_spd):
                        spd_cross_idx = j
                        break
                        
                if spd_cross_idx == -1:
                    for j in range(i, min(len(pts) - 1, i + search_range)):
                        if check_crossing(pts[j], pts[j+1], l_spd):
                            spd_cross_idx = j
                            break

                if spd_cross_idx != -1:
                    p_net_cross = p_curr
                    p_spd_cross = pts[spd_cross_idx]
                    
                    pixel_dist = math.hypot(p_net_cross[1][0] - p_spd_cross[1][0], p_net_cross[1][1] - p_spd_cross[1][1])
                    frame_diff = abs(p_net_cross[0] - p_spd_cross[0])
                    
                    if frame_diff > 0:
                        speed_px = pixel_dist / frame_diff
                        crossings.append({'frame': p_curr[0], 'direction': direction, 'speed_px': speed_px})
        return crossings


# =====================================================================
# [模块 4: 核心追踪器]
# =====================================================================
class QueueTracker:
    def __init__(self, 
                 search_window=Config.TRK_SEARCH_WINDOW, 
                 create_thr=Config.TRK_CREATE_THR, 
                 eval_window=Config.TRK_EVAL_WINDOW, 
                 static_thr=Config.TRK_STATIC_THR, 
                 static_lock_radius=Config.TRK_STATIC_RADIUS, 
                 zone_persistence=Config.TRK_ZONE_PERSIST):
        self.base_search_window = search_window 
        self.base_create_thr = create_thr       
        self.eval_window = eval_window     
        self.static_thr = static_thr       
        self.static_lock_radius = static_lock_radius 
        self.zone_persistence = zone_persistence 
        self.max_search_window = 10   
        self.max_create_thr = 100.0   
        self.queues = []                   
        self.queue_id_counter = 0
        self.static_zones = {} 

    def detect_objects(self, model, frame, conf_thr, imgsz=960):
        results = model(frame, conf=conf_thr, imgsz=imgsz, half=True, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append((cx, cy, x1, y1, x2, y2))
        return detections

    def process_frame(self, frame_idx, detections):
        matched_detects = set()
        self.static_zones = {qid: zone for qid, zone in self.static_zones.items() if frame_idx - zone['last_seen'] <= self.zone_persistence}

        for q in self.queues:
            age = frame_idx - q['history'][0][0]
            if q['is_static'] and age >= self.eval_window:
                self.static_zones[q['id']] = {'pos': np.array(q['history'][-1][1][0:2]), 'last_seen': frame_idx, 'q_ref': q}

        for d_idx, det in enumerate(detections):
            det_pos = np.array(det[0:2])
            for z_id, zone in self.static_zones.items():
                if math.hypot(det_pos[0]-zone['pos'][0], det_pos[1]-zone['pos'][1]) <= self.static_lock_radius:
                    zone['last_seen'] = frame_idx; zone['pos'] = det_pos
                    zone['q_ref']['history'].append((frame_idx, det, False))
                    matched_detects.add(d_idx)
                    break 

        match_candidates = []
        matched_queues = set([zone['q_ref']['id'] for zone in self.static_zones.values() if zone['last_seen'] == frame_idx])
        
        for d_idx, det in enumerate(detections):
            if d_idx in matched_detects: continue 
            for q_idx, q in enumerate(self.queues):
                if q['id'] in matched_queues: continue
                speed = q.get('speed', 0.0)
                dynamic_window = min(self.max_search_window, self.base_search_window + int(speed * 1.0))
                dynamic_thr = min(self.max_create_thr, self.base_create_thr + (speed * dynamic_window * 0.8))
                
                min_dist = float('inf')
                for past_frame, past_det, _ in reversed(q['history']):
                    if frame_idx - past_frame <= dynamic_window:
                        dist = math.hypot(det[0] - past_det[0], det[1] - past_det[1])
                        if dist < min_dist: min_dist = dist
                    else: break 
                        
                if min_dist != float('inf'):
                    match_candidates.append((min_dist, dynamic_thr, q_idx, d_idx))

        match_candidates.sort(key=lambda x: x[0])
        queue_allocated = set([q_id for q_id in matched_queues]) 

        for dist, dynamic_thr, q_idx, d_idx in match_candidates:
            if self.queues[q_idx]['id'] in queue_allocated or d_idx in matched_detects: continue
            if dist <= dynamic_thr:
                self.queues[q_idx]['history'].append((frame_idx, detections[d_idx], False))
                queue_allocated.add(self.queues[q_idx]['id'])
                matched_detects.add(d_idx)
                
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detects:
                self.queues.append({'id': self.queue_id_counter, 'history': deque([(frame_idx, det, False)], maxlen=Config.TRK_MAX_HISTORY), 'is_static': True, 'speed': 0.0})
                self.queue_id_counter += 1

        for q in self.queues:
            history_list = list(q['history'])
            eval_items = [item for item in history_list if frame_idx - item[0] <= self.eval_window]
            if len(eval_items) >= 2:
                oldest_det = eval_items[0][1]
                newest_det = eval_items[-1][1]
                displacement = math.hypot(newest_det[0] - oldest_det[0], newest_det[1] - oldest_det[1])
                frame_span = eval_items[-1][0] - eval_items[0][0]
                if frame_span > 0:
                    avg_dist = displacement / frame_span
                    q['speed'] = avg_dist 
                    q['is_static'] = (avg_dist < self.static_thr)

        surviving_queues = []
        for q in self.queues:
            time_since_last_seen = frame_idx - q['history'][-1][0]
            speed = q.get('speed', 0.0)
            dynamic_window = min(self.max_search_window, self.base_search_window + int(speed * 1.0))
            max_allowed_gap = self.zone_persistence if q['is_static'] else dynamic_window
            if time_since_last_seen <= max_allowed_gap: surviving_queues.append(q)
                
        self.queues = surviving_queues

    def get_render_data(self, frame_idx, analyzer, calibrator):
        moving_dets, static_dets = [], []
        moving_segments = [] 
        bounces_to_render = []
        crossings_to_report = []

        for q in self.queues:
            if not q['is_static'] and len(q['history']) > 1:
                pts = [item for item in q['history'] if frame_idx - item[0] <= Config.TRK_MAX_HISTORY]
                smoothed_pts = analyzer.smooth(pts)
                
                bounces = analyzer.detect_bounces(smoothed_pts)
                for b_frame, bx, by, angle in bounces:
                    bounces_to_render.append((b_frame, bx, by, angle))
                    
                crossings = analyzer.detect_net_crossings(smoothed_pts, calibrator)
                for c in crossings:
                    crossings_to_report.append(c)

                render_pts = [p for p in smoothed_pts if frame_idx - p[0] <= 45]
                for i in range(len(render_pts) - 1):
                    moving_segments.append((render_pts[i][1][0:2], render_pts[i+1][1][0:2], render_pts[i+1][2]))

            if q['history'][-1][0] == frame_idx:
                det, is_stitched = q['history'][-1][1], q['history'][-1][2]
                if not is_stitched:
                    if q['is_static']: static_dets.append(det)
                    else: moving_dets.append(det)
                    
        return moving_dets, static_dets, moving_segments, bounces_to_render, crossings_to_report


# =====================================================================
# [模块 5: 主程序执行流] 
# =====================================================================
def overlay_tracking_queue_based(video_path, model_ball_path):
    print(f">>> 正在加载网球专属模型: {model_ball_path} ...")
    model_ball = YOLO(model_ball_path)
    
    print(">>> 正在加载人物检测模型 (通用 COCO yolo11n) ...")
    model_person = YOLO(Config.MODEL_PERSON) 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print("错误：无法打开视频！")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = cap.read()
    calibrator = CourtCalibrator(w, h)
    calibrator.calibrate(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

    target_h = Config.DISPLAY_HEIGHT
    disp_scale = target_h / h
    video_disp_w = int(w * disp_scale)
    minimap_width = int(video_disp_w * Config.MINIMAP_RATIO) 
    total_disp_w = video_disp_w + minimap_width
    
    calibrator.build_static_minimap(target_h, minimap_width)

    # ⚡ 初始化视频录制器
    out_writer = None
    if Config.SAVE_VIDEO:
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        if fps_input <= 0: fps_input = 30
        out_path = os.path.join(Config.BASE_PATH, Config.OUTPUT_VIDEO_NAME)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_input, (total_disp_w, target_h))
        print(f">>> 视频录制已开启: 将保存至 {out_path} ({total_disp_w}x{target_h} @ {fps_input}fps)")

    window_name = 'YOLO MR_ShuttleStrike (Final Record Version)'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    tracker = QueueTracker()
    stitcher = TrajectoryStitcher()
    analyzer = TrajectoryAnalyzer()
    
    enable_stitching = Config.ENABLE_STITCHING
    paused = False
    
    global_ball_history = {}      
    global_player_history = {}    
    global_bounces = {}
    global_hits = {}              
    global_crossings = {'bottom_up': 0.0, 'top_down': 0.0}
    processed_crossing_frames = set() 

    while True:
        if not paused:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.remap(frame, calibrator.map1, calibrator.map2, interpolation=cv2.INTER_LINEAR)
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            detections_ball = tracker.detect_objects(model_ball, frame, conf_thr=Config.CONF_BALL, imgsz=Config.IMGSZ_BALL) 
            results_person = model_person.track(
                frame, conf=Config.CONF_PERSON, imgsz=Config.IMGSZ_PERSON, classes=[0], 
                persist=True, tracker="bytetrack.yaml", half=True, verbose=False
            )
            
            person_boxes = []
            current_players = []
            if results_person[0].boxes.id is not None:
                boxes = results_person[0].boxes.xyxy.cpu().numpy()
                track_ids = results_person[0].boxes.id.int().cpu().tolist()
                for box, t_id in zip(boxes, track_ids):
                    px1, py1, px2, py2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    person_boxes.append((px1, py1, px2, py2, t_id))
                    
                    feet_x = (px1 + px2) / 2
                    feet_y = py2
                    real_coords = calibrator.pixel_to_real(feet_x, feet_y)
                    rx, ry = real_coords if real_coords else (0, 0)
                    
                    hit_anchor_x = feet_x
                    hit_anchor_y = py1 + (py2 - py1) * 0.3
                    
                    current_players.append({
                        'id': t_id, 'cx': hit_anchor_x, 'cy': hit_anchor_y, 'rx': rx, 'ry': ry
                    })
                    
            global_player_history[frame_idx] = current_players
            if frame_idx - 150 in global_player_history:
                del global_player_history[frame_idx - 150]

            tracker.process_frame(frame_idx=frame_idx, detections=detections_ball)
            if enable_stitching: tracker.queues = stitcher.stitch_queues(tracker.queues)
            
            for q in tracker.queues:
                if not q['is_static']:
                    for item in q['history']:
                        f_idx, det, _ = item
                        global_ball_history[f_idx] = (det[0], det[1])
                        
            if frame_idx - 150 in global_ball_history:
                del global_ball_history[frame_idx - 150]
            
            moving_dets, static_dets, moving_segments, frame_bounces, frame_crossings = tracker.get_render_data(frame_idx, analyzer, calibrator)

            # ----------------------------------------
            # 截击网跨越事件与上区回溯算法
            # ----------------------------------------
            for c in frame_crossings:
                c_frame = c['frame']
                if c_frame not in processed_crossing_frames:
                    processed_crossing_frames.add(c_frame)
                    direction = c['direction']
                    speed_px = c['speed_px']
                    
                    coef = Config.SPEED_COEF_UP if direction == 'bottom_up' else Config.SPEED_COEF_DOWN
                    global_crossings[direction] = speed_px * coef
                    
                    if direction == 'top_down':
                        best_hit_frame = None
                        min_dist = float('inf')
                        best_hit_data = None
                        
                        search_start = c_frame
                        search_end = max(0, c_frame - Config.TOP_HIT_LOOKBACK_FRAMES)
                        
                        for f_back in range(search_start, search_end, -1):
                            if f_back in global_ball_history and f_back in global_player_history:
                                bx, by = global_ball_history[f_back]
                                for p in global_player_history[f_back]:
                                    if abs(p['rx']) <= 4.115 + Config.ROI_SIDE_MARGIN and p['ry'] <= -Config.ROI_NET_MARGIN:
                                        dist = math.hypot(bx - p['cx'], by - p['cy'])
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_hit_frame = f_back
                                            best_hit_data = {'px': bx, 'py': by, 'rx': p['rx'], 'ry': p['ry']}
                        
                        if best_hit_frame and min_dist <= Config.HIT_DIST_PX_TOP_MAX:
                            global_hits[best_hit_frame] = best_hit_data
                            print(f"[TOP HIT DETECTED] 拓扑回溯成功! F:{best_hit_frame} | 距离: {min_dist:.1f}px")

            # ----------------------------------------
            # 下区击球 (Hit) 与弹跳 (Bounce) 分离逻辑
            # ----------------------------------------
            for b_frame, bx, by, angle in frame_bounces:
                if b_frame in global_bounces or b_frame in global_hits: continue 
                
                real_coords = calibrator.pixel_to_real(bx, by)
                if not real_coords: continue
                rx, ry = real_coords
                
                is_hit = False
                matched_player = None
                
                if ry > 0: 
                    if angle >= Config.HIT_ANGLE_THR:
                        search_frames = [b_frame, b_frame-1, b_frame+1, b_frame-2, b_frame+2]
                        for sf in search_frames:
                            if is_hit: break
                            if sf in global_player_history:
                                for p in global_player_history[sf]:
                                    if abs(p['rx']) <= 4.115 + Config.ROI_SIDE_MARGIN and p['ry'] >= -Config.ROI_NET_MARGIN:
                                        ry_ratio = min(max(p['ry'], 0.0), 11.885) / 11.885
                                        dynamic_dist_thr = Config.HIT_DIST_PX_NET + (Config.HIT_DIST_PX_BASE - Config.HIT_DIST_PX_NET) * ry_ratio
                                        
                                        dist = math.hypot(bx - p['cx'], by - p['cy'])
                                        if dist <= dynamic_dist_thr:
                                            is_hit = True
                                            matched_player = p
                                            print(f"[BOTTOM HIT DETECTED] F:{b_frame} | 角度: {angle:.1f}° | 距离: {dist:.1f}px (阈值:{dynamic_dist_thr:.1f}px)")
                                            break
                
                if is_hit and matched_player:
                    hit_rx = rx
                    hit_ry = matched_player['ry']
                    
                    keys_to_delete = []
                    for existing_frame, data in global_hits.items():
                        if abs(b_frame - existing_frame) <= Config.CLEAN_TIME_FRAMES:
                            if math.hypot(hit_rx - data['rx'], hit_ry - data['ry']) <= Config.CLEAN_SPACE_METERS:
                                keys_to_delete.append(existing_frame)
                    for k in keys_to_delete: del global_hits[k]
                    
                    global_hits[b_frame] = {'px': bx, 'py': by, 'rx': hit_rx, 'ry': hit_ry}
                else:
                    keys_to_delete = []
                    for existing_frame, data in global_bounces.items():
                        if abs(b_frame - existing_frame) <= Config.CLEAN_TIME_FRAMES:
                            if math.hypot(rx - data['rx'], ry - data['ry']) <= Config.CLEAN_SPACE_METERS:
                                keys_to_delete.append(existing_frame)
                    for k in keys_to_delete: del global_bounces[k]
                    
                    is_in = abs(rx) <= 4.115 and abs(ry) <= 11.885 
                    global_bounces[b_frame] = {'px': bx, 'py': by, 'rx': rx, 'ry': ry, 'is_in': is_in}

            # ---- 1. 主画面渲染 ----
            calibrator.draw_virtual_court(frame)
            
            for px1, py1, px2, py2, p_id in person_boxes:
                cv2.rectangle(frame, (px1, py1), (px2, py2), Config.C_PERSON_BOX, 2)
                cv2.rectangle(frame, (px1, py1 - 20), (px1 + 50, py1), Config.C_PERSON_BOX, -1)
                cv2.putText(frame, f"P{p_id}", (px1 + 5, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            for det in static_dets:
                cv2.rectangle(frame, (int(det[2]), int(det[3])), (int(det[4]), int(det[5])), Config.C_STA_BOX, 1)
            for pt1, pt2, is_stitched in moving_segments:
                color = Config.C_MOV_STITCHED if is_stitched else Config.C_MOV_TRACK
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 1 if is_stitched else 2)
            for det in moving_dets:
                cv2.circle(frame, (int(det[0]), int(det[1])), radius=6, color=(0, 255, 0), thickness=-1)
                cv2.rectangle(frame, (int(det[2]), int(det[3])), (int(det[4]), int(det[5])), Config.C_MOV_BOX, 2)

            for existing_frame, b_data in global_bounces.items():
                if 0 <= frame_idx - existing_frame <= Config.VIS_RESIDUAL_FRAMES: 
                    bp_x, bp_y = b_data['px'], b_data['py']
                    cv2.circle(frame, (int(bp_x), int(bp_y)), 15, Config.C_BOUNCE, 2)
                    cv2.circle(frame, (int(bp_x), int(bp_y)), 5, Config.C_BOUNCE, -1)
                    cv2.putText(frame, "BOUNCE!", (int(bp_x) + 20, int(bp_y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.C_BOUNCE, 2)

            for existing_frame, h_data in global_hits.items():
                if 0 <= frame_idx - existing_frame <= Config.VIS_RESIDUAL_FRAMES: 
                    hp_x, hp_y = h_data['px'], h_data['py']
                    cv2.circle(frame, (int(hp_x), int(hp_y)), 25, Config.C_HIT, 3) 
                    cv2.putText(frame, "HIT!", (int(hp_x) + 30, int(hp_y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, Config.C_HIT, 3)

            # ---- 2. 分辨率锁定拼接 ----
            frame_disp = cv2.resize(frame, (video_disp_w, target_h))
            minimap_disp = calibrator.generate_minimap(global_bounces, global_crossings, global_hits)
            final_display = np.hstack((frame_disp, minimap_disp))

            ms_total = (time.time() - t_start) * 1000
            fps = 1000.0 / (ms_total + 1e-5)
            
            stitch_status = "ON" if enable_stitching else "OFF"
            status_text = f"Players: {len(person_boxes)} | Stitching: {stitch_status} | FPS: {fps:.1f}"
            cv2.putText(final_display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ⚡ 将渲染好的最终画面写入视频文件
            if Config.SAVE_VIDEO and out_writer is not None:
                out_writer.write(final_display)

            cv2.imshow(window_name, final_display)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('s') or key == ord('S'): enable_stitching = not enable_stitching
        elif key == ord('c') or key == ord('C'): 
            global_bounces.clear()
            global_hits.clear()

    # ⚡ 释放录制资源
    cap.release()
    if Config.SAVE_VIDEO and out_writer is not None:
        out_writer.release()
        print(f">>> 录制完成，视频已成功保存至: {Config.OUTPUT_VIDEO_NAME}")
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    overlay_tracking_queue_based(Config.VIDEO_INPUT, Config.MODEL_BALL)