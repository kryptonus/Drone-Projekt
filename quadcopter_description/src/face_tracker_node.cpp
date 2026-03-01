#include "rclcpp/rclcpp.hpp"
#include "gazebo_msgs/srv/set_entity_state.hpp"
#include "gazebo_msgs/msg/entity_state.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <chrono>
#include <memory>
#include <cmath>
#include <string>

using namespace std::chrono_literals;

// ============================================================
// PID CONTROLLER CLASS
// Same math you know from control theory
// P = proportional, I = integral, D = derivative
// ============================================================
class PIDController
{
public:
    PIDController(double kp, double ki, double kd)
    : kp_(kp), ki_(ki), kd_(kd),
      prev_error_(0.0), integral_(0.0)
    {
        last_time_ = std::chrono::steady_clock::now();
    }

    double compute(double error)
    {
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_time_).count();
        if (dt <= 0.0) dt = 0.033;

        // P term — proportional to current error
        double P = kp_ * error;

        // I term — accumulated error over time
        integral_ += error * dt;
        // Clamp integral to prevent windup
        integral_ = std::max(-50.0, std::min(50.0, integral_));
        double I = ki_ * integral_;

        // D term — rate of change (acts like brakes)
        double derivative = (error - prev_error_) / dt;
        double D = kd_ * derivative;

        prev_error_ = error;
        last_time_ = now;

        return P + I + D;
    }

    void reset()
    {
        prev_error_ = 0.0;
        integral_ = 0.0;
    }

private:
    double kp_, ki_, kd_;
    double prev_error_;
    double integral_;
    std::chrono::steady_clock::time_point last_time_;
};


// ============================================================
// MAIN NODE CLASS
// Inherits from rclcpp::Node — same as inheriting any C++ class
// ============================================================
class FaceTrackerNode : public rclcpp::Node
{
public:
    FaceTrackerNode() : Node("face_tracker_node"),
        drone_x_(0.0), drone_y_(0.0),
        drone_z_(1.5), drone_yaw_(0.0),
        face_lost_count_(0),
        // Initialize PID controllers
        // Kp=0.003: move 0.003m per pixel of error
        pid_yaw_   (0.003, 0.0001, 0.002),
        pid_height_(0.003, 0.0001, 0.002),
        pid_dist_  (0.005, 0.0001, 0.002)
    {
        // Target face width in pixels
        // Controls how close drone stays to your face
        target_face_width_ = 150;

        // Image center — target position for face
        image_center_x_ = 320;  // 640/2
        image_center_y_ = 240;  // 480/2

        // Create Gazebo service client
        // This is what moves the drone in simulation
        gazebo_client_ = this->create_client<gazebo_msgs::srv::SetEntityState>(
            "/gazebo/set_entity_state"
        );

        // Wait for Gazebo to be ready
        RCLCPP_INFO(this->get_logger(), "Waiting for Gazebo service...");
        while (!gazebo_client_->wait_for_service(1s)) {
            RCLCPP_WARN(this->get_logger(), "Gazebo not ready, waiting...");
        }
        RCLCPP_INFO(this->get_logger(), "Gazebo connected!");

        // Load Haar Cascade face detector
        std::string cascade_path =
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
        
        if (!face_cascade_.load(cascade_path)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load Haar Cascade!");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Face detector loaded!");

        // Open webcam
        cap_.open(0);
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open webcam!");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Webcam opened!");

        // Move drone to hover position first
        moveDroneTo(0.0, 0.0, 1.5, 0.0);
        RCLCPP_INFO(this->get_logger(), "Drone hovering at 1.5m — face tracking active!");

        // Main tracking loop at 30Hz
        // create_wall_timer = while(true) loop with 33ms sleep
        timer_ = this->create_wall_timer(
            33ms,
            std::bind(&FaceTrackerNode::trackingLoop, this)
        );
    }

    ~FaceTrackerNode()
    {
        cap_.release();
        cv::destroyAllWindows();
    }

private:

    // --------------------------------------------------------
    // MOVE DRONE — sends position command to Gazebo
    // --------------------------------------------------------
    void moveDroneTo(double x, double y, double z, double yaw)
    {
        auto request = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
        request->state.name = "stealth_quad";
        request->state.pose.position.x = x;
        request->state.pose.position.y = y;
        request->state.pose.position.z = z;

        // Convert yaw to quaternion
        // For yaw rotation only: qz=sin(yaw/2), qw=cos(yaw/2)
        request->state.pose.orientation.x = 0.0;
        request->state.pose.orientation.y = 0.0;
        request->state.pose.orientation.z = std::sin(yaw / 2.0);
        request->state.pose.orientation.w = std::cos(yaw / 2.0);

        // Send async — don't block the tracking loop
        gazebo_client_->async_send_request(request);
    }

    // --------------------------------------------------------
    // MAIN TRACKING LOOP — runs 30 times per second
    // --------------------------------------------------------
    void trackingLoop()
    {
        // Read frame from webcam
        cv::Mat frame;
        bool ret = cap_.read(frame);
        if (!ret || frame.empty()) return;

        // Mirror frame — feels natural for face tracking
        cv::flip(frame, frame, 1);

        // Convert to grayscale for face detection
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Detect faces
        std::vector<cv::Rect> faces;
        face_cascade_.detectMultiScale(
            gray, faces,
            1.1,    // scaleFactor
            5,      // minNeighbors
            0,      // flags
            cv::Size(60, 60)  // minSize
        );

        // Draw crosshair at image center (target)
        cv::line(frame,
            cv::Point(image_center_x_ - 20, image_center_y_),
            cv::Point(image_center_x_ + 20, image_center_y_),
            cv::Scalar(0, 255, 0), 2);
        cv::line(frame,
            cv::Point(image_center_x_, image_center_y_ - 20),
            cv::Point(image_center_x_, image_center_y_ + 20),
            cv::Scalar(0, 255, 0), 2);

        if (!faces.empty()) {
            face_lost_count_ = 0;

            // Take the largest face (closest person)
            cv::Rect largest_face = *std::max_element(
                faces.begin(), faces.end(),
                [](const cv::Rect& a, const cv::Rect& b) {
                    return a.area() < b.area();
                }
            );

            int x = largest_face.x;
            int y = largest_face.y;
            int w = largest_face.width;
            int h = largest_face.height;

            // Face center
            int face_cx = x + w / 2;
            int face_cy = y + h / 2;

            // CALCULATE ERRORS
            double error_x = face_cx - image_center_x_;  // left/right
            double error_y = face_cy - image_center_y_;  // up/down
            double error_z = w - target_face_width_;      // distance

            // RUN PID CONTROLLERS
            double correction_yaw    = pid_yaw_.compute(error_x);
            double correction_height = pid_height_.compute(-error_y);
            double correction_dist   = pid_dist_.compute(-error_z);

            // Clamp corrections — prevent wild movements
            auto clamp = [](double val, double min, double max) {
                return std::max(min, std::min(max, val));
            };

            drone_yaw_ += clamp(correction_yaw,    -0.05, 0.05);
            drone_z_   += clamp(correction_height, -0.05, 0.05);
            drone_x_   += clamp(correction_dist,   -0.05, 0.05);

            // Safety limits
            drone_z_ = std::max(0.3, std::min(4.0, drone_z_));
            drone_x_ = std::max(-5.0, std::min(5.0, drone_x_));

            // Send position to Gazebo
            moveDroneTo(drone_x_, drone_y_, drone_z_, drone_yaw_);

            // Draw tracking box
            cv::rectangle(frame,
                cv::Point(x, y),
                cv::Point(x + w, y + h),
                cv::Scalar(0, 255, 0), 2);

            // Line from face to center
            cv::line(frame,
                cv::Point(face_cx, face_cy),
                cv::Point(image_center_x_, image_center_y_),
                cv::Scalar(255, 0, 0), 1);

            // HUD overlay
            cv::putText(frame, "TRACKING",
                cv::Point(x, y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);

            cv::putText(frame,
                "Error X: " + std::to_string((int)error_x) + "px",
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);

            cv::putText(frame,
                "Error Y: " + std::to_string((int)error_y) + "px",
                cv::Point(10, 55),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);

            cv::putText(frame,
                "Drone Z: " + std::to_string(drone_z_).substr(0, 4) + "m",
                cv::Point(10, 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);

            cv::putText(frame,
                "Face W: " + std::to_string(w) + "px",
                cv::Point(10, 105),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);

        } else {
            face_lost_count_++;
            std::string status = (face_lost_count_ < 30) ?
                "SEARCHING..." : "FACE LOST - HOVERING";

            cv::putText(frame, status,
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 255), 2);
        }

        // Show webcam window with HUD
        cv::imshow("Face Tracking HUD", frame);
        cv::waitKey(1);
    }

    // Member variables
    double drone_x_, drone_y_, drone_z_, drone_yaw_;
    int face_lost_count_;
    int target_face_width_;
    int image_center_x_, image_center_y_;

    // PID controllers
    PIDController pid_yaw_;
    PIDController pid_height_;
    PIDController pid_dist_;

    // ROS 2 objects
    rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr gazebo_client_;
    rclcpp::TimerBase::SharedPtr timer_;

    // OpenCV objects
    cv::VideoCapture cap_;
    cv::CascadeClassifier face_cascade_;
};


// ============================================================
// MAIN — entry point, same as any C++ program
// ============================================================
int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FaceTrackerNode>());
    rclcpp::shutdown();
    return 0;
}