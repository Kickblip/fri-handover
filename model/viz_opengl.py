"""
OpenGL visualization to compare actual vs predicted receiving-hand landmarks in 3D.
Outputs a video file showing the comparison over time with timestamps.

Usage:
    python -m model.viz_opengl <stem> [--pred path/to/predictions.csv] [--output path/to/output.mp4]

Visualization:
    - Actual hand landmarks   → GREEN points with connections
    - Predicted hand landmarks → RED points with connections
    - Error lines (if enabled) → YELLOW lines connecting actual to predicted
    - Timestamp displayed in seconds
    - Legend displayed with actual and predicted hands
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import cv2

try:
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_LINES,
        GL_POINTS,
        GL_PROJECTION,
        GL_MODELVIEW,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        glBegin,
        glClear,
        glClearColor,
        glColor3f,
        glEnable,
        glEnd,
        glLoadIdentity,
        glMatrixMode,
        glPointSize,
        glReadPixels,
        glRotatef,
        glTranslatef,
        glVertex3f,
        glViewport,
    )
except ImportError:
    print("Error: PyOpenGL is required. Install with: pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)

try:
    import glfw
except ImportError:
    print("Error: GLFW is required. Install with: pip install glfw")
    sys.exit(1)

from .config import PRED_DIR, HANDS_DIR, VIDEO_DIR
from .data import load_receiving_hand_world, _read_csv, _pick_col


# Hand landmark connections (MediaPipe hand skeleton)
HAND_CONNECTIONS = [
    # Wrist to thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Wrist to index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Wrist to middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Wrist to ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Wrist to pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


class OpenGLViewer:
    def __init__(self, stem: str, pred_path: Optional[Path] = None, output_path: Optional[Path] = None):
        self.stem = stem
        self.window = None
        self.width = 1280
        self.height = 720
        self.output_path = output_path
        
        # Camera state
        self.rot_x = 30.0
        self.rot_y = -45.0
        self.zoom = 1.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        
        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        
        # Animation state
        self.current_frame_idx = 0
        self.playing = False
        self.frame_time = 0.0
        self.fps = 30.0
        
        # Data
        self.actual_map: Dict[int, np.ndarray] = {}
        self.pred_map: Dict[int, np.ndarray] = {}
        self.shared_frames: list[int] = []
        self.frame_to_time: Dict[int, float] = {}  # Map frame index to time in seconds
        
        # Visualization options
        self.show_connections = True
        self.show_error_lines = True
        
        # Load data
        self.load_data(pred_path)
        
    def load_data(self, pred_path: Optional[Path] = None):
        """Load actual and predicted hand coordinates."""
        print("Loading actual receiving hand coordinates...")
        actual_coords, actual_frames = load_receiving_hand_world(self.stem)
        n_landmarks = actual_coords.shape[1] // 3
        actual_coords = actual_coords.reshape(len(actual_frames), n_landmarks, 3)
        self.actual_map = {f: actual_coords[i] for i, f in enumerate(actual_frames)}
        
        print("Loading predicted coordinates...")
        if pred_path is None:
            pred_path = PRED_DIR / f"{self.stem}_future_predictions.csv"
        
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions CSV: {pred_path}")
        
        df = pd.read_csv(pred_path)
        required = {"frame", "future_frame_idx"}
        if not required.issubset(df.columns):
            raise ValueError(f"Predictions CSV must contain {required}, got {df.columns[:8].tolist()}")
        
        lm_cols = [c for c in df.columns if c.startswith("lm_") and c.endswith(("_x", "_y", "_z"))]
        if not lm_cols:
            raise ValueError(f"No landmark columns (lm_*_{x|y|z}) found in {pred_path}")
        
        lm_cols = sorted(lm_cols)
        arr = df[lm_cols].to_numpy(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        n_landmarks = arr.shape[1] // 3
        
        pred_map: Dict[int, Tuple[np.ndarray, int]] = {}
        for row, coords in zip(df.itertuples(index=False), arr):
            base_frame = int(row.frame)
            horizon = int(row.future_frame_idx)
            target_frame = base_frame + horizon + 1
            pts = coords.reshape(n_landmarks, 3)
            
            # Keep the prediction generated from the most recent base frame
            if target_frame in pred_map and base_frame <= pred_map[target_frame][1]:
                continue
            pred_map[target_frame] = (pts, base_frame)
        
        self.pred_map = {frame: data[0] for frame, data in pred_map.items()}
        
        # Find shared frames
        self.shared_frames = sorted(set(self.actual_map.keys()) & set(self.pred_map.keys()))
        if not self.shared_frames:
            raise RuntimeError("No overlapping frames between predictions and ground truth.")
        
        print(f"Loaded {len(self.shared_frames)} frames with both actual and predicted data")
        
        # Create frame to time mapping (assuming 30 FPS)
        for i, frame in enumerate(self.shared_frames):
            self.frame_to_time[frame] = i / self.fps
        
        # Calculate bounding box for camera setup
        all_points = []
        for frame in self.shared_frames:
            all_points.append(self.actual_map[frame])
            all_points.append(self.pred_map[frame])
        all_points = np.concatenate(all_points, axis=0)
        self.center = np.mean(all_points, axis=0)
        self.scale = np.max(np.abs(all_points - self.center)) * 1.5
        
    def init_gl(self):
        """Initialize OpenGL state."""
        gl.glEnable(GL_DEPTH_TEST)
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glPointSize(5.0)
        
    def setup_projection(self):
        """Set up projection matrix."""
        gl.glMatrixMode(GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = self.width / self.height if self.height > 0 else 1.0
        glu.gluPerspective(45.0, aspect, 0.1, 1000.0)
        gl.glMatrixMode(GL_MODELVIEW)
        
    def draw_hand(self, landmarks: np.ndarray, color: Tuple[float, float, float], show_connections: bool = True):
        """Draw a hand as points and connections."""
        if landmarks.shape[0] != 21:
            return
        
        # Draw points
        gl.glColor3f(*color)
        gl.glBegin(GL_POINTS)
        for landmark in landmarks:
            if not np.any(np.isnan(landmark)):
                gl.glVertex3f(landmark[0], landmark[1], landmark[2])
        gl.glEnd()
        
        # Draw connections
        if show_connections:
            gl.glBegin(GL_LINES)
            for i, j in HAND_CONNECTIONS:
                if i < len(landmarks) and j < len(landmarks):
                    p1, p2 = landmarks[i], landmarks[j]
                    if not (np.any(np.isnan(p1)) or np.any(np.isnan(p2))):
                        gl.glVertex3f(p1[0], p1[1], p1[2])
                        gl.glVertex3f(p2[0], p2[1], p2[2])
            gl.glEnd()
    
    def draw_error_lines(self, actual: np.ndarray, predicted: np.ndarray):
        """Draw lines connecting actual to predicted landmarks."""
        if actual.shape[0] != predicted.shape[0]:
            return
        
        gl.glColor3f(1.0, 1.0, 0.0)  # Yellow
        gl.glBegin(GL_LINES)
        for i in range(min(len(actual), len(predicted))):
            a, p = actual[i], predicted[i]
            if not (np.any(np.isnan(a)) or np.any(np.isnan(p))):
                gl.glVertex3f(a[0], a[1], a[2])
                gl.glVertex3f(p[0], p[1], p[2])
        gl.glEnd()
    
    def render(self):
        """Render the current frame."""
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        
        # Camera transform
        gl.glTranslatef(self.trans_x, self.trans_y, -self.zoom * self.scale)
        gl.glRotatef(self.rot_x, 1, 0, 0)
        gl.glRotatef(self.rot_y, 0, 1, 0)
        gl.glTranslatef(-self.center[0], -self.center[1], -self.center[2])
        
        # Get current frame data
        if 0 <= self.current_frame_idx < len(self.shared_frames):
            frame = self.shared_frames[self.current_frame_idx]
            actual = self.actual_map[frame]
            predicted = self.pred_map.get(frame)
            
            if predicted is not None:
                # Draw actual hand (green)
                self.draw_hand(actual, (0.0, 1.0, 0.0), self.show_connections)
                
                # Draw predicted hand (red)
                self.draw_hand(predicted, (1.0, 0.0, 0.0), self.show_connections)
                
                # Draw error lines (yellow)
                if self.show_error_lines:
                    self.draw_error_lines(actual, predicted)
        
        # Draw coordinate axes
        gl.glBegin(GL_LINES)
        gl.glColor3f(1.0, 0.0, 0.0)  # X - Red
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0.1, 0, 0)
        gl.glColor3f(0.0, 1.0, 0.0)  # Y - Green
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0.1, 0)
        gl.glColor3f(0.0, 0.0, 1.0)  # Z - Blue
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0, 0.1)
        gl.glEnd()
    
    def read_pixels(self) -> np.ndarray:
        """Read pixels from OpenGL framebuffer and return as numpy array (BGR format for OpenCV)."""
        pixels = gl.glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8)
        img = img.reshape((self.height, self.width, 3))
        # Flip vertically (OpenGL origin is bottom-left, OpenCV is top-left)
        img = np.flipud(img)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr
    
    def add_text_overlay(self, img: np.ndarray, text: str, position: Tuple[int, int] = (10, 30)):
        """Add text overlay to image."""
        cv2.putText(
            img, text, position,
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )
        return img
    
    def render_video(self):
        """Render all frames to a video file."""
        if self.output_path is None:
            self.output_path = VIDEO_DIR / f"{self.stem}_opengl_comparison.mp4"
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Rendering video to: {self.output_path}")
        print(f"Total frames: {len(self.shared_frames)}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")
        
        # Initialize OpenGL context (headless)
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Hide window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        
        self.window = glfw.create_window(self.width, self.height, "", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        self.init_gl()
        
        # Auto-rotate camera for better visualization
        auto_rotate = True
        rotation_speed = 0.5  # degrees per frame
        
        try:
            for frame_idx in range(len(self.shared_frames)):
                self.current_frame_idx = frame_idx
                frame = self.shared_frames[frame_idx]
                time_sec = self.frame_to_time.get(frame, frame_idx / self.fps)
                
                # Auto-rotate camera
                if auto_rotate:
                    self.rot_y = (self.rot_y + rotation_speed) % 360.0
                
                # Render frame
                gl.glViewport(0, 0, self.width, self.height)
                self.setup_projection()
                self.render()
                
                # Read pixels
                img = self.read_pixels()
                
                # Add text overlay with timestamp
                time_text = f"Time: {time_sec:.2f}s | Frame: {frame}"
                self.add_text_overlay(img, time_text, (10, 30))
                
                # Add legend
                legend_y = 60
                self.add_text_overlay(img, "Actual (Green)", (10, legend_y))
                self.add_text_overlay(img, "Predicted (Red)", (10, legend_y + 30))
                
                # Write frame
                writer.write(img)
                
                if (frame_idx + 1) % 10 == 0:
                    print(f"Rendered {frame_idx + 1}/{len(self.shared_frames)} frames")
        
        finally:
            writer.release()
            glfw.destroy_window(self.window)
            glfw.terminate()
        
        print(f"✓ Video saved: {self.output_path.resolve()}")
        
    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button events."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_down = (action == glfw.PRESS)
    
    def cursor_pos_callback(self, window, x, y):
        """Handle mouse movement."""
        if self.mouse_down:
            dx = x - self.mouse_x
            dy = y - self.mouse_y
            self.rot_y += dx * 0.5
            self.rot_x += dy * 0.5
        self.mouse_x = x
        self.mouse_y = y
    
    def scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll (zoom)."""
        self.zoom = max(0.1, min(5.0, self.zoom - yoffset * 0.1))
    
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        if action != glfw.PRESS and action != glfw.REPEAT:
            return
        
        if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_LEFT:
            self.current_frame_idx = max(0, self.current_frame_idx - 1)
            print(f"Frame: {self.shared_frames[self.current_frame_idx] if self.current_frame_idx < len(self.shared_frames) else 'N/A'}")
        elif key == glfw.KEY_RIGHT:
            self.current_frame_idx = min(len(self.shared_frames) - 1, self.current_frame_idx + 1)
            print(f"Frame: {self.shared_frames[self.current_frame_idx] if self.current_frame_idx < len(self.shared_frames) else 'N/A'}")
        elif key == glfw.KEY_SPACE:
            self.playing = not self.playing
            print("Playing" if self.playing else "Paused")
        elif key == glfw.KEY_R:
            self.rot_x = 30.0
            self.rot_y = -45.0
            self.zoom = 1.0
            self.trans_x = 0.0
            self.trans_y = 0.0
            print("Camera reset")
        elif key == glfw.KEY_C:
            self.show_connections = not self.show_connections
            print(f"Connections: {'ON' if self.show_connections else 'OFF'}")
        elif key == glfw.KEY_E:
            self.show_error_lines = not self.show_error_lines
            print(f"Error lines: {'ON' if self.show_error_lines else 'OFF'}")
    
    def run(self):
        """Main render loop."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        
        self.window = glfw.create_window(
            self.width, self.height, f"Handover Compare - {self.stem}", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_window_size_callback(self.window, self.window_size_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        
        self.init_gl()
        
        last_time = glfw.get_time()
        
        print("\n=== Controls ===")
        print("Mouse drag: Rotate camera")
        print("Mouse wheel: Zoom")
        print("Left/Right arrows: Navigate frames")
        print("Space: Play/pause")
        print("R: Reset camera")
        print("C: Toggle connections")
        print("E: Toggle error lines")
        print("Q/ESC: Quit")
        print("===============\n")
        
        while not glfw.window_should_close(self.window):
            current_time = glfw.get_time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update animation
            if self.playing:
                self.frame_time += dt
                if self.frame_time >= 1.0 / self.fps:
                    self.frame_time = 0.0
                    self.current_frame_idx = (self.current_frame_idx + 1) % len(self.shared_frames)
            
            glfw.poll_events()
            
            gl.glViewport(0, 0, self.width, self.height)
            self.setup_projection()
            self.render()
            
            glfw.swap_buffers(self.window)
        
        glfw.terminate()
    
    def window_size_callback(self, window, width, height):
        """Handle window resize."""
        self.width = width
        self.height = height


def main():
    ap = argparse.ArgumentParser(description="Visualize actual vs predicted hand landmarks in 3D using OpenGL.")
    ap.add_argument("stem", help="Video/data stem, e.g., 1_video")
    ap.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="Optional custom predictions CSV (defaults to dataset/model_output/predictions/<stem>_future_predictions.csv)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (defaults to dataset/model_output/videos/<stem>_opengl_comparison.mp4). If not provided, opens interactive viewer.",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive viewer instead of rendering video (default: render video if --output is provided)",
    )
    args = ap.parse_args()
    
    viewer = OpenGLViewer(args.stem, args.pred, args.output)
    
    if args.interactive or args.output is None:
        # Interactive mode
        viewer.run()
    else:
        # Video rendering mode
        viewer.render_video()


if __name__ == "__main__":
    main()

