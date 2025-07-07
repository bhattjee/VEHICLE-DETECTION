import cv2
import os
import time
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import json

# ============================================================================
# ENHANCED CONFIGURATION - OPTIMIZED FOR VEHICLE COUNTING
# ============================================================================

# Video file path (use raw string for Windows paths)
VIDEO_FILE = r"C:/Users/SAMB 01/pythonauto/imagecounting/cardemo1.mp4"

# System Settings
FRAME_WIDTH = 800
FRAME_HEIGHT = 450
COUNTING_LINE_Y = int(FRAME_HEIGHT * 0.6)  # 60% of frame height
CONFIDENCE_THRESHOLD = 0.4  # Lower threshold for better detection
PIXEL_TO_METER_RATIO = 0.1  
OFFSET = 20  # Increased counting zone offset

# Vehicle classes to detect (COCO dataset)
VEHICLE_CLASSES = [2, 3, 5, 7, 6]  # car, motorcycle, bus, truck, (added train)

# Tracking parameters
DEEPSORT_MAX_AGE = 50  # Increased from 30 for better tracking
DEEPSORT_N_INIT = 5    # Increased from 3 for more confirmation frames
MIN_CROSSING_PIXELS = 10  # Minimum movement to count as crossing

# ============================================================================

try:
    from ultralytics import YOLO
    print("✅ YOLO imported successfully")
except ImportError:
    print("❌ Error: Install ultralytics with: pip install ultralytics")
    exit(1)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    print("✅ DeepSort imported successfully")
except ImportError:
    print("❌ Error: Install deep-sort-realtime with: pip install deep-sort-realtime")
    exit(1)

class VideoVehicleMonitor:
    def __init__(self, video_path):
        """Initialize the enhanced Video Vehicle Monitor."""
        self.video_path = video_path
        self.setup_parameters()
        self.setup_models()
        self.setup_tracking()
        self.setup_video()
        
    def setup_parameters(self):
        """Setup enhanced system parameters."""
        self.FRAME_WIDTH = FRAME_WIDTH
        self.FRAME_HEIGHT = FRAME_HEIGHT
        self.CENTER_X = self.FRAME_WIDTH // 2
        self.COUNT_LINE_Y = COUNTING_LINE_Y
        self.OFFSET = OFFSET
        self.VEHICLE_CLASSES = VEHICLE_CLASSES
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.PIXEL_TO_METER = PIXEL_TO_METER_RATIO
        self.MIN_CROSSING_PIXELS = MIN_CROSSING_PIXELS
        
    def setup_models(self):
        """Load enhanced YOLO and DeepSort models."""
        print("📦 Loading YOLO model...")
        self.model = YOLO("yolov8s.pt")  # Using small model for better accuracy
        print("✅ YOLO model loaded")
        
        print("📦 Loading DeepSort tracker...")
        self.tracker = DeepSort(max_age=DEEPSORT_MAX_AGE, n_init=DEEPSORT_N_INIT)
        print("✅ DeepSort tracker loaded")
        
    def setup_tracking(self):
        """Initialize enhanced tracking variables."""
        self.left_lane_count = 0
        self.right_lane_count = 0
        self.counted_ids = set()
        self.centroid_history = {}
        self.speed_history = defaultdict(deque)
        self.vehicle_speeds = {}
        self.frame_timestamps = {}
        self.vehicle_log = []
        self.detection_log = []
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_count = 0
        self.total_detections = 0
        
    def setup_video(self):
        """Setup video capture with enhanced checks."""
        if not os.path.exists(self.video_path):
            print(f"❌ Video file not found: {self.video_path}")
            print("💡 Make sure the video file exists in the correct location")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"❌ Cannot open video file: {self.video_path}")
            raise RuntimeError("Cannot open video file")
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        
        print(f"📹 Video loaded: {self.video_path}")
        print(f"📊 Video properties:")
        print(f"   - Total frames: {self.total_frames}")
        print(f"   - FPS: {self.video_fps:.1f}")
        print(f"   - Duration: {self.video_duration:.1f} seconds")
        print(f"   - Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
    def calculate_speed(self, track_id, current_pos, timestamp):
        """Enhanced speed calculation with smoothing."""
        if track_id not in self.speed_history:
            self.speed_history[track_id] = deque(maxlen=10)
            self.frame_timestamps[track_id] = deque(maxlen=10)
        
        self.speed_history[track_id].append(current_pos)
        self.frame_timestamps[track_id].append(timestamp)
        
        if len(self.speed_history[track_id]) >= 3:
            positions = list(self.speed_history[track_id])
            timestamps = list(self.frame_timestamps[track_id])
            
            # Calculate total distance traveled with smoothing
            total_distance = 0
            valid_frames = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                distance = np.sqrt(dx*dx + dy*dy) * self.PIXEL_TO_METER
                if distance > 0:  # Ignore zero movement
                    total_distance += distance
                    valid_frames += 1
            
            # Calculate time elapsed
            time_elapsed = timestamps[-1] - timestamps[0]
            
            if time_elapsed > 0 and valid_frames > 0:
                speed_ms = total_distance / time_elapsed
                speed_kmh = speed_ms * 3.6
                # Filter out unrealistic speeds with more conservative thresholds
                if 5 < speed_kmh < 180:  # Adjusted realistic speed range
                    self.vehicle_speeds[track_id] = speed_kmh
                    return speed_kmh
        
        return 0.0
    
    def draw_interface(self, frame):
        """Enhanced user interface with more information."""
        # Draw detection zones
        cv2.line(frame, (self.CENTER_X, 0), (self.CENTER_X, self.FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(frame, (0, self.COUNT_LINE_Y), (self.FRAME_WIDTH, self.COUNT_LINE_Y), (0, 255, 255), 3)
        
        # Draw counting zones with larger offset
        cv2.rectangle(frame, (0, self.COUNT_LINE_Y - self.OFFSET), 
                     (self.CENTER_X, self.COUNT_LINE_Y + self.OFFSET), (255, 0, 0), 2)
        cv2.rectangle(frame, (self.CENTER_X, self.COUNT_LINE_Y - self.OFFSET), 
                     (self.FRAME_WIDTH, self.COUNT_LINE_Y + self.OFFSET), (255, 0, 0), 2)
        
        # Enhanced info panel
        cv2.rectangle(frame, (10, 10), (500, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, 180), (255, 255, 255), 2)
        
        # Display enhanced statistics
        total_count = self.left_lane_count + self.right_lane_count
        progress = (self.frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0
        
        cv2.putText(frame, f'Left Lane: {self.left_lane_count}', (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Right Lane: {self.right_lane_count}', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Total: {total_count}', (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Progress: {progress:.1f}%', (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Tracking: {len(self.centroid_history)}', (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Right side info
        active_vehicles = len([v for v in self.vehicle_speeds.values() if v > 0])
        avg_speed = np.mean([v for v in self.vehicle_speeds.values() if v > 0]) if active_vehicles > 0 else 0
        cv2.putText(frame, f'Active: {active_vehicles}', (300, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'Avg Speed: {avg_speed:.1f} km/h', (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'Detections: {self.total_detections}', (300, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Enhanced lane labels
        cv2.putText(frame, 'LEFT LANE', (50, self.COUNT_LINE_Y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, 'RIGHT LANE', (self.FRAME_WIDTH - 170, self.COUNT_LINE_Y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Progress bar with frame count
        bar_width = 400
        bar_height = 10
        bar_x = 20
        bar_y = self.FRAME_HEIGHT - 30
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        progress_width = int((progress / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        cv2.putText(frame, f'Frame: {self.frame_count}/{self.total_frames}', 
                    (bar_x + bar_width + 10, bar_y + bar_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    def draw_vehicle_info(self, frame, track_id, x1, y1, x2, y2, cx, cy):
        """Enhanced vehicle information drawing."""
        # Choose color based on lane with more contrast
        color = (0, 255, 0) if cx < self.CENTER_X else (255, 100, 0)
        
        # Draw enhanced bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)
        
        # Draw vehicle ID with background for better visibility
        (text_width, text_height), _ = cv2.getTextSize(f'ID:{track_id}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width + 2, y1 - 30 + text_height + 2), (0, 0, 0), -1)
        cv2.putText(frame, f'ID:{track_id}', (x1 + 1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw speed if available
        if track_id in self.vehicle_speeds:
            speed = self.vehicle_speeds[track_id]
            if speed > 1:
                speed_text = f'{speed:.1f}km/h'
                (speed_width, speed_height), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 10), (x1 + speed_width + 2, y1 - 10 + speed_height + 2), (0, 0, 0), -1)
                cv2.putText(frame, speed_text, (x1 + 1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw enhanced trajectory line
        if track_id in self.centroid_history:
            prev_cy = self.centroid_history[track_id]
            if abs(cy - prev_cy) > self.MIN_CROSSING_PIXELS:  # Only draw significant movements
                cv2.line(frame, (cx, prev_cy), (cx, cy), color, 2)
    
    def process_frame(self, frame):
        """Enhanced frame processing with better counting logic."""
        self.frame_count += 1
        timestamp = time.time()
        
        # Resize frame
        frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        
        # Enhanced YOLO detection
        results = self.model(frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)[0]
        
        # Prepare detections with class filtering
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in self.VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1
                
                # Filter out too small detections
                if w > 20 and h > 20:  # Minimum vehicle size threshold
                    detections.append(([x1, y1, w, h], conf, cls))
                    self.total_detections += 1
        
        # Update tracker with enhanced parameters
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Process tracked vehicles with improved counting logic
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Calculate speed with enhanced method
            speed = self.calculate_speed(track_id, (cx, cy), timestamp)
            
            # Enhanced counting logic
            if track_id in self.centroid_history:
                prev_cy = self.centroid_history[track_id]
                
                # Check if vehicle crossed the line with significant movement
                crossing_condition = (
                    (prev_cy < self.COUNT_LINE_Y - self.OFFSET and 
                     cy >= self.COUNT_LINE_Y - self.OFFSET and
                     abs(cy - prev_cy) > self.MIN_CROSSING_PIXELS) or
                    (prev_cy > self.COUNT_LINE_Y + self.OFFSET and 
                     cy <= self.COUNT_LINE_Y + self.OFFSET and
                     abs(cy - prev_cy) > self.MIN_CROSSING_PIXELS)
                )
                
                if crossing_condition and track_id not in self.counted_ids:
                    lane = "Left" if cx < self.CENTER_X else "Right"
                    
                    if lane == "Left":
                        self.left_lane_count += 1
                    else:
                        self.right_lane_count += 1
                    
                    self.counted_ids.add(track_id)
                    
                    # Enhanced logging
                    final_speed = self.vehicle_speeds.get(track_id, 0)
                    log_entry = {
                        'id': track_id,
                        'speed_kmh': round(final_speed, 1),
                        'lane': lane,
                        'frame': self.frame_count,
                        'timestamp': datetime.now().isoformat(),
                        'direction': "Down" if cy > prev_cy else "Up",
                        'bbox': [x1, y1, x2, y2],
                        'centroid': [cx, cy]
                    }
                    self.vehicle_log.append(log_entry)
                    
                    print(f"🚗 Vehicle {track_id} crossed {lane} lane at {final_speed:.1f} km/h (Frame: {self.frame_count})")
            
            # Update history even if not counted for better tracking
            self.centroid_history[track_id] = cy
            self.draw_vehicle_info(frame, track_id, x1, y1, x2, y2, cx, cy)
        
        # Log detection information for debugging
        self.detection_log.append({
            'frame': self.frame_count,
            'detections': len(detections),
            'tracks': len(tracks),
            'timestamp': datetime.now().isoformat()
        })
        
        self.draw_interface(frame)
        return frame
    
    def update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
    
    def save_data(self):
        """Enhanced data saving with more information."""
        if self.vehicle_log:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"vehicle_data_{timestamp}.json"
            detection_filename = f"detection_log_{timestamp}.json"
            
            # Save vehicle data
            with open(filename, 'w') as f:
                json.dump({
                    'video_file': self.video_path,
                    'total_vehicles': len(self.vehicle_log),
                    'left_lane_count': self.left_lane_count,
                    'right_lane_count': self.right_lane_count,
                    'frame_count': self.frame_count,
                    'total_frames': self.total_frames,
                    'processing_date': datetime.now().isoformat(),
                    'config': {
                        'frame_width': self.FRAME_WIDTH,
                        'frame_height': self.FRAME_HEIGHT,
                        'counting_line_y': self.COUNT_LINE_Y,
                        'confidence_threshold': self.CONFIDENCE_THRESHOLD,
                        'vehicle_classes': self.VEHICLE_CLASSES
                    },
                    'vehicles': self.vehicle_log
                }, f, indent=2)
            
            # Save detection log
            with open(detection_filename, 'w') as f:
                json.dump({
                    'video_file': self.video_path,
                    'total_detections': self.total_detections,
                    'detection_log': self.detection_log
                }, f, indent=2)
                
            print(f"💾 Data saved to {filename} and {detection_filename}")
    
    def run(self):
        """Enhanced main execution loop with better controls."""
        print("🎬 Starting Enhanced Video Vehicle Monitor...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save data")
        print("  - Press 'r' to restart video")
        print("  - Press SPACE to pause/resume")
        print("  - Press 'd' to toggle debug info")
        print("=" * 50)
        
        paused = False
        show_debug = False
        processed_frame = None
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("📹 Video ended")
                        break
                    
                    processed_frame = self.process_frame(frame)
                    self.update_fps()
                else:
                    if processed_frame is not None:
                        cv2.putText(processed_frame, "PAUSED", (self.FRAME_WIDTH//2 - 100, self.FRAME_HEIGHT//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                if processed_frame is not None:
                    if show_debug:
                        debug_frame = processed_frame.copy()
                        cv2.putText(debug_frame, "DEBUG MODE", (self.FRAME_WIDTH - 200, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Video Vehicle Monitor - DEBUG", debug_frame)
                    else:
                        cv2.imshow("Video Vehicle Monitor", processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_data()
                elif key == ord('r'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_count = 0
                    self.counted_ids = set()
                    self.centroid_history = {}
                    print("🔄 Video restarted - Counters reset")
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    print(f"{'🔍 Debug mode ON' if show_debug else '👁️ Debug mode OFF'}")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup with more detailed results."""
        self.save_data()
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Final Results:")
        print(f"=" * 50)
        print(f"Video: {self.video_path}")
        print(f"Frames processed: {self.frame_count}/{self.total_frames} ({self.frame_count/self.total_frames:.1%})")
        print(f"Left lane vehicles: {self.left_lane_count}")
        print(f"Right lane vehicles: {self.right_lane_count}")
        print(f"Total vehicles: {self.left_lane_count + self.right_lane_count}")
        print(f"Total detections: {self.total_detections}")
        
        if self.vehicle_log:
            speeds = [v['speed_kmh'] for v in self.vehicle_log if v['speed_kmh'] > 0]
            if speeds:
                print(f"Average speed: {np.mean(speeds):.1f} km/h")
                print(f"Max speed: {max(speeds):.1f} km/h")
                print(f"Min speed: {min(speeds):.1f} km/h")
        
        print(f"=" * 50)
        print("💾 Data saved to JSON files")
        print("🛑 Program terminated")

if __name__ == "__main__":
    # Enhanced file checking
    if not os.path.exists(VIDEO_FILE):
        print(f"❌ Video file not found: {VIDEO_FILE}")
        print("💡 Please verify the file path and try again")
        print("Current working directory:", os.getcwd())
        print("Files in directory:", os.listdir())
        input("Press Enter to exit...")
        exit(1)
    
    try:
        print("🚀 Initializing Enhanced Vehicle Counting System...")
        monitor = VideoVehicleMonitor(VIDEO_FILE)
        monitor.run()
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")