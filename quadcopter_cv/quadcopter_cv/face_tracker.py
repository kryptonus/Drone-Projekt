#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import cv2
import numpy as np
import time
import math


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.033
        P = self.Kp * error
        self.integral += error * dt
        self.integral = max(-50, min(50, self.integral))
        I = self.Ki * self.integral
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative
        self.prev_error = error
        self.last_time = current_time
        return P + I + D


class FaceTrackingDrone(Node):
    def __init__(self):
        super().__init__('face_tracking_drone')
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 1.5
        self.drone_yaw = 0.0
        self.pid_yaw = PIDController(Kp=0.003, Ki=0.0001, Kd=0.002)
        self.pid_height = PIDController(Kp=0.003, Ki=0.0001, Kd=0.002)
        self.pid_dist = PIDController(Kp=0.005, Ki=0.0001, Kd=0.002)
        self.target_face_width = 150
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.get_logger().info('Waiting for Gazebo service...')
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Gazebo not ready, waiting...')
        self.get_logger().info('Gazebo connected!')
        cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open webcam!')
            return
        self.get_logger().info('Webcam opened!')
        self.image_center_x = 320
        self.image_center_y = 240
        self.move_drone_to(0.0, 0.0, 1.5, 0.0)
        self.get_logger().info('Drone hovering at 1.5m — face tracking active!')
        self.timer = self.create_wall_timer(0.033, self.tracking_loop)
        self.face_lost_count = 0

    def move_drone_to(self, x, y, z, yaw):
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = 'stealth_quad'
        request.state.pose.position.x = float(x)
        request.state.pose.position.y = float(y)
        request.state.pose.position.z = float(z)
        request.state.pose.orientation.z = math.sin(yaw / 2.0)
        request.state.pose.orientation.w = math.cos(yaw / 2.0)
        self.client.call_async(request)

    def tracking_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        cv2.line(frame, (self.image_center_x-20, self.image_center_y),
                 (self.image_center_x+20, self.image_center_y), (0,255,0), 2)
        cv2.line(frame, (self.image_center_x, self.image_center_y-20),
                 (self.image_center_x, self.image_center_y+20), (0,255,0), 2)
        if len(faces) > 0:
            self.face_lost_count = 0
            largest_face = max(faces, key=lambda f: f[2]*f[3])
            x, y, w, h = largest_face
            face_cx = x + w // 2
            face_cy = y + h // 2
            error_x = face_cx - self.image_center_x
            error_y = face_cy - self.image_center_y
            error_z = w - self.target_face_width
            correction_yaw = self.pid_yaw.compute(error_x)
            correction_height = self.pid_height.compute(-error_y)
            correction_dist = self.pid_dist.compute(-error_z)
            self.drone_yaw += max(-0.05, min(0.05, correction_yaw))
            self.drone_z += max(-0.05, min(0.05, correction_height))
            self.drone_x += max(-0.05, min(0.05, correction_dist))
            self.drone_z = max(0.3, min(4.0, self.drone_z))
            self.drone_x = max(-5.0, min(5.0, self.drone_x))
            self.move_drone_to(self.drone_x, self.drone_y, self.drone_z, self.drone_yaw)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.line(frame, (face_cx,face_cy),
                     (self.image_center_x,self.image_center_y), (255,0,0), 1)
            cv2.putText(frame, 'TRACKING', (x,y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f'Error X: {error_x:.0f}px', (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(frame, f'Error Y: {error_y:.0f}px', (10,50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(frame, f'Drone Z: {self.drone_z:.2f}m', (10,70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            self.face_lost_count += 1
            status = 'SEARCHING...' if self.face_lost_count < 30 else 'FACE LOST'
            cv2.putText(frame, status, (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, f'Drone Z: {self.drone_z:.2f}m', (10,60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('Face Tracking HUD', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FaceTrackingDrone()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
