# camera_processor.py
# This node reads the drone camera and processes the image

# ROS 2 Python library — like importing numpy, but for robots
import rclpy
from rclpy.node import Node

# The message type our camera publishes
# Think of it as the "data format" of the WhatsApp message
from sensor_msgs.msg import Image

# cv_bridge = translator between ROS image format and OpenCV format
# ROS and OpenCV store images differently, this converts between them
from cv_bridge import CvBridge

# OpenCV — the actual computer vision library
import cv2

# numpy — for math on image arrays
import numpy as np


class CameraProcessor(Node):
    """
    This is a ROS 2 Node.
    A Node = one program that does one job.
    This node's job: read camera → process image → show result
    """

    def __init__(self):
        # Give this node a name (shows up in rqt_graph)
        super().__init__('camera_processor')

        # CREATE A SUBSCRIBER
        # Subscriber = someone who listens to a WhatsApp group
        # We are listening to: /stealth_quad/front_camera/image_raw
        # When a new image arrives, call self.image_callback automatically
        self.subscription = self.create_subscription(
            Image,                                          # Message type we expect
            '/stealth_quad/front_camera/image_raw',        # Topic to listen to
            self.image_callback,                           # Function to call on each message
            10                                             # Queue size (buffer 10 images)
        )

        # Create the ROS↔OpenCV translator
        self.bridge = CvBridge()

        # Counter so we can print every 30 frames instead of flooding terminal
        self.frame_count = 0

        self.get_logger().info('Camera Processor started! Waiting for images...')

    def image_callback(self, ros_image):
        """
        This function runs automatically every time a new camera frame arrives.
        ros_image = the raw image from Gazebo camera (in ROS format)
        """

        # STEP 1: Convert ROS image → OpenCV image (numpy array)
        # bgr8 = Blue Green Red format, 8 bits per channel (standard)
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')

        # STEP 2: Get image dimensions
        height, width, channels = cv_image.shape

        # STEP 3: Convert to grayscale (needed for many CV algorithms)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # STEP 4: Edge detection using Canny algorithm
        # This finds boundaries between objects — foundation of object detection
        # 50, 150 = minimum and maximum threshold values
        edges = cv2.Canny(gray, 50, 150)

        # STEP 5: Draw a targeting crosshair in the center of the image
        # This simulates a defense targeting reticle
        center_x = width // 2
        center_y = height // 2
        # Draw horizontal line (green, thickness 1)
        cv2.line(cv_image, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 0), 1)
        # Draw vertical line
        cv2.line(cv_image, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 0), 1)
        # Draw circle around crosshair
        cv2.circle(cv_image, (center_x, center_y), 20, (0, 255, 0), 1)

        # STEP 6: Add text overlay (like a HUD display)
        cv2.putText(cv_image,
                    f'STEALTH_QUAD CAM | {width}x{height}',
                    (10, 25),                          # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,          # Font
                    0.5,                               # Size
                    (0, 255, 0),                       # Color (green)
                    1)                                 # Thickness

        cv2.putText(cv_image,
                    f'FRAME: {self.frame_count}',
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1)

        # STEP 7: Show both windows
        cv2.imshow('Drone Camera Feed', cv_image)   # Color feed with HUD
        cv2.imshow('Edge Detection', edges)          # What the AI "sees"

        # STEP 8: Required OpenCV call — processes window events
        # waitKey(1) = wait 1 millisecond, keeps window responsive
        cv2.waitKey(1)

        self.frame_count += 1

        # Print a status message every 60 frames (about every 2 seconds)
        if self.frame_count % 60 == 0:
            self.get_logger().info(f'Processed {self.frame_count} frames. Image size: {width}x{height}')


def main(args=None):
    rclpy.init(args=args)           # Start ROS 2
    node = CameraProcessor()        # Create our node
    rclpy.spin(node)                # Keep running until Ctrl+C
    node.destroy_node()             # Clean up
    rclpy.shutdown()                # Stop ROS 2


if __name__ == '__main__':
    main()
