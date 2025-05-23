import cv2
from pathlib import Path

# Create directory to store captured frames if it doesn't exist
save_dir = Path('film_agent/frames')/'pulin'
save_dir.mkdir(exist_ok=True)

# Initialize the camera (0 for default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open the camera.")

# Counter for frame filenames (continue numbering from existing files)
existing_files = save_dir.glob('frame_*.jpg')
frame_numbers = [int(f.stem.split('_')[1]) for f in existing_files if f.stem.split('_')[1].isdigit()]
frame_count = max(frame_numbers, default=0)

print("Press 's' to capture a frame, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame in a window
    cv2.imshow('Frame Capture', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Save frame when 's' is pressed
    if key == ord('s'):
        frame_count += 1
        filename = save_dir / f'frame_{frame_count}.jpg'
        cv2.imwrite(str(filename), frame)
        print(f"Captured frame saved as {filename}")
        
        # Add visual feedback (green text) that image was captured
        feedback_frame = frame.copy()
        cv2.putText(feedback_frame, 'Frame Captured!', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame Capture', feedback_frame)
        cv2.waitKey(500)  # Show feedback for 500ms
    
    # Quit when 'q' is pressed
    elif key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
print(f"Session ended. {frame_count} frames captured.")
