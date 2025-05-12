import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))


# Import LIDA components
from lidapy.agent import Environment
import traceback
import queue
import time
import threading
import gymnasium as gym 
import cv2

from film_agent.utils import print_error
from film_agent.clip_utils import clip_image_encoder, clip_text_encoder

#region Environment
class FilmEnvironment(Environment):
    def __init__(self, video_source=0, 
                 reference_image_folders=None,
                 output_dir="recordings", fps=30.0, 
                 display_frames=True, display_frequency=1):
        # Ensure output_dir is not None and is a valid path
        if output_dir is None:
            raise ValueError("output_dir cannot be None. Please provide a valid directory path.")
        
        super(FilmEnvironment, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.is_recording = False
        self.cap = cv2.VideoCapture(video_source)
        self.current_frame = None
        self.output_dir = str(Path(output_dir))  # Convert to string to ensure compatibility
        self.video_writer = None
        self.current_video_path = None
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        
        # Add parameters for frame display
        self.display_frames = display_frames
        self.display_frequency = display_frequency
        self.frame_count = 0
        self.window_name = "Film Environment"
        
        # Initialize display_texts for custom text overlays
        self.display_texts = []
        
        if self.display_frames:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)

        self.recording_thread = None
        self.recording_queue = queue.Queue(maxsize=60)
        self.should_record = False
        self.thread_active = False
        self.thread_lock = threading.Lock()
        self.last_heartbeat_time = 0

        # Create output directory using pathlib
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        if reference_image_folders is not None:
            self.reference_images = {folder: self.load_images_from_folder(folder) for folder in reference_image_folders}
          
        self.text_id = self.add_display_text("", position=(10, 90), font_scale=0.7, color=(255, 255, 255), thickness=1, outline=True)

    def collect_embeddings(self, folder):
        # Process throwing references
        embeddings = []
        print(f"Processing {len(self.reference_images[folder])} throwing reference images...")
        for i, ref_img in enumerate(self.reference_images[folder]):
            try:
                emb = clip_image_encoder(ref_img)
                if emb:
                    embeddings.append(emb)
            except Exception as e:
                print_error(f"Failed to process reference image {i}: {e}")
                # Continue with other reference images
        
        return embeddings

    def load_images_from_folder(self, folder_path):
        if not folder_path:
            raise ValueError("Folder path cannot be None")
        
        folder = Path('film_agent/frames')/folder_path
        if not folder.is_dir():
            raise ValueError("Folder path is not a directory")
            
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(folder.glob(f"**/{ext}")))
            
        if len(image_paths) == 0:
            raise ValueError(f"No images found in folder: {folder_path}")

        return [self.load_reference_image(p) for p in image_paths if self.load_reference_image(p) is not None]

    def load_reference_image(self, image_path):
        try:
            image = cv2.imread(str(image_path))
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
        except Exception:
            return None

    def record_frame(self):
        if not self.is_recording or self.current_frame is None:
            return
        now = time.time()
        if now - self.last_frame_time >= self.frame_interval:
            try:
                self.recording_queue.put_nowait(self.current_frame.copy())
                self.last_frame_time = now
            except queue.Full:
                pass

    def stop_recording(self):
        with self.thread_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self.should_record = False

        self.recording_queue.put(None)

        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5)
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved to {self.current_video_path}")
            self.video_writer = None

    def start_recording(self):
        with self.thread_lock:
            if self.is_recording:
                return
            if self.recording_thread and self.recording_thread.is_alive():
                self.thread_active = False
                self.recording_thread.join()

            if self.current_frame is None:
                raise Exception("No current frame available to start recording")

            h, w = self.current_frame.shape[:2]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.current_video_path = str(Path(self.output_dir) / f"recording_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(self.current_video_path, fourcc, self.fps, (w, h))

            if not self.video_writer.isOpened():
                raise Exception("Failed to open video writer")

            self.should_record = True
            self.thread_active = True
            self.is_recording = True
            self.last_frame_time = time.time()

            with self.recording_queue.mutex:
                self.recording_queue.queue.clear()

            self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
            self.recording_thread.start()

    def _recording_worker(self):
        while self.thread_active:
            with self.thread_lock:
                self.last_heartbeat_time = time.time()
            try:
                frame = self.recording_queue.get(timeout=0.05)
                if frame is None:
                    self.recording_queue.task_done()
                    break
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(bgr_frame)
                self.recording_queue.task_done()
            except queue.Empty:
                time.sleep(0.005)

        if self.video_writer:
            self.video_writer.release()
        self.video_writer = None
        self.thread_active = False

    def close(self):
        if self.is_recording:
            self.stop_recording()
        if self.cap:
            self.cap.release()
        
        # Also close the display window if it exists
        if self.display_frames:
            cv2.destroyWindow(self.window_name)

    # Explicitly implement required abstract methods
    def receive_sensory_stimuli(self):
        """Read frame with rate-limiting for more consistent processing"""
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise Exception("Failed to read frame from camera")
            
            # Convert to RGB format
            try:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                raise Exception(f"Error converting frame: {e}")
            
            # If we're recording, add this frame to video queue
            if self.is_recording and self.thread_active:
                try:
                    self.record_frame()
                except Exception as e:
                    raise Exception(f"Error recording frame: {e}")
            
            # Display frame with recording indicator if enabled
            if self.display_frames and (self.frame_count % self.display_frequency == 0):
                # Create a copy of the frame for display (BGR format for OpenCV)
                display_frame = cv2.cvtColor(self.current_frame.copy(), cv2.COLOR_RGB2BGR)
                
                # Add timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(display_frame, timestamp, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(display_frame, timestamp, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Add frame counter
                cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Add recording indicator
                if self.is_recording:
                    # Red circle and "REC" text to indicate recording
                    cv2.circle(display_frame, (display_frame.shape[1] - 40, 30), 15, (0, 0, 255), -1)
                    cv2.putText(display_frame, "REC", (display_frame.shape[1] - 80, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add custom display texts
                for text_item in self.display_texts:
                    if text_item['outline']:
                        cv2.putText(display_frame, text_item['text'], text_item['position'], 
                                    cv2.FONT_HERSHEY_SIMPLEX, text_item['font_scale'], (0, 0, 0), 
                                    text_item['thickness'] + 1)
                    cv2.putText(display_frame, text_item['text'], text_item['position'], 
                                cv2.FONT_HERSHEY_SIMPLEX, text_item['font_scale'], text_item['color'], 
                                text_item['thickness'])
                
                # Show the frame
                cv2.imshow(self.window_name, display_frame)
                
                # Process any keyboard input (with a short timeout)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    raise KeyboardInterrupt("ESC key pressed")
            
            # Increment frame counter
            self.frame_count += 1
            
            return self.current_frame # TODO attach to a sensor
        except Exception as e:
            print_error(f"{e}")
            traceback.print_exc()
            return None

    def run_commands(self, commands):
        """Execute motor commands for recording control
        
        This method is required by the Environment abstract base class.
        
        Args:
            commands: Dictionary containing commands
            
        Returns:
            Current recording state
        """
        if commands is None:
            return self.is_recording

        if 'record' in commands:
            if commands['record'] == 0:  # Start recording
                self.start_recording()
            elif commands['record'] == 1:  # Stop recording
                self.stop_recording()
            # If None, maintain current state
        if 'display' in commands:
            self.update_display_text(self.text_id, text=commands['display'])
        else:
            self.update_display_text(self.text_id, text="")
    
        return self.is_recording
        
    def reset(self):
        """Reset the environment
        
        This method is required by some Environment implementations.
        
        Returns:
            Initial observation
        """
        # Stop any active recording
        if self.is_recording:
            self.stop_recording()
            
        # Clear any cached state
        self.current_frame = None
        
        # Get first observation
        return self.receive_sensory_stimuli()

    def add_display_text(self, text, position=(10, 90), font_scale=0.7, color=(255, 255, 255), thickness=1, outline=True):
        """Add text to be displayed on the video frame.
        
        Args:
            text (str): Text to display on the frame
            position (tuple): (x, y) coordinates for text placement
            font_scale (float): Font scale factor
            color (tuple): Text color in BGR format (default: white)
            thickness (int): Text thickness
            outline (bool): Whether to add black outline for better visibility
            
        Returns:
            int: Index of the added text item (can be used to update or remove it later)
        """
        text_item = {
            'text': text,
            'position': position,
            'font_scale': font_scale,
            'color': color,
            'thickness': thickness,
            'outline': outline
        }
        self.display_texts.append(text_item)
        return len(self.display_texts) - 1  # Return index for future reference
        
    def update_display_text(self, index, text=None, position=None, font_scale=None, color=None, thickness=None, outline=None):
        """Update previously added text properties.
        
        Args:
            index (int): Index of the text item to update
            text (str, optional): New text to display
            position (tuple, optional): New position
            font_scale (float, optional): New font scale
            color (tuple, optional): New text color
            thickness (int, optional): New thickness
            outline (bool, optional): Whether to add outline
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if index < 0 or index >= len(self.display_texts):
            return False
            
        if text is not None:
            self.display_texts[index]['text'] = text
        if position is not None:
            self.display_texts[index]['position'] = position
        if font_scale is not None:
            self.display_texts[index]['font_scale'] = font_scale
        if color is not None:
            self.display_texts[index]['color'] = color
        if thickness is not None:
            self.display_texts[index]['thickness'] = thickness
        if outline is not None:
            self.display_texts[index]['outline'] = outline
            
        return True
        
    def remove_display_text(self, index):
        """Remove text from display.
        
        Args:
            index (int): Index of the text item to remove
            
        Returns:
            bool: True if removal successful, False otherwise
        """
        if index < 0 or index >= len(self.display_texts):
            return False
            
        self.display_texts.pop(index)
        return True
        
    def clear_display_texts(self):
        """Remove all custom text overlays from display."""
        self.display_texts = []
#endregion
