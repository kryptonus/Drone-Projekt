 * face_tracker_node.cpp
 * 
 * Autonomous face-following drone controller for ROS 2 / Gazebo simulation.
 * Developed interest for Robotics research and project development, IEB (Informationstechnik/Elektronik), TH Mannheim.
 * 
 * Architecture:
 *   Drone camera topic → cv_bridge → OpenCV face detection →
 *   PID error correction → Gazebo SetEntityState → drone moves
 * 
 * Key design decisions:
 * - Used Haar Cascade over DNN detector: lower latency on CPU, sufficient
 *   for single-target frontal face tracking at <3m range
 * - PID gains loaded from YAML params: allows tuning without recompiling
 * - SetEntityState service used over cmd_vel: direct position control gives
 *   cleaner sim behavior without needing a full flight controller
 * 
 * Known limitations:
 * - Haar Cascade has false positives in high-contrast backgrounds
 * - No depth sensor: distance estimated from face bounding box width only
 * - Gazebo sim-to-real gap: gains will need retuning on physical hardware
 */

#include "rclcpp/rclcpp.hpp"
#include "gazebo_msgs/srv/set_entity_state.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "quadcopter_control/pid_controller.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <chrono>
#include <memory>
#include <cmath>
#include <string>

using namespace std::chrono_literals;
using std::placeholders::_1;

class FaceTrackerNode : public rclcpp::Node
{
public:
    FaceTrackerNode() : Node("face_tracker_node"),
        drone_x_(0.0), drone_y_(0.0),
        drone_z_(1.5), drone_yaw_(0.0),
        face_lost_count_(0)
    {
        // Load PID gains from YAML params — no recompile needed for tuninging them!
        double kp_yaw    = this->declare_parameter("kp_yaw",    0.003);
        double ki_yaw    = this->declare_parameter("ki_yaw",    0.0001);
        double kd_yaw    = this->declare_parameter("kd_yaw",    0.002);
        double kp_height = this->declare_parameter("kp_height", 0.003);
        double ki_height = this->declare_parameter("ki_height", 0.0001);
        double kd_height = this->declare_parameter("kd_height", 0.002);
        double kp_dist   = this->declare_parameter("kp_dist",   0.005);
        double ki_dist   = this->declare_parameter("ki_dist",   0.0001);
        double kd_dist   = this->declare_parameter("kd_dist",   0.002);

        target_face_width_ = this->declare_parameter("target_face_width", 150);
        cascade_path_      = this->declare_parameter("cascade_path",
            std::string("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"));
        min_height_  = this->declare_parameter("min_height",   0.3);
        max_height_  = this->declare_parameter("max_height",   4.0);
        max_xy_range_= this->declare_parameter("max_xy_range", 5.0);

        // PID controllers with loaded gains
        pid_yaw_    = std::make_unique<PIDController>(kp_yaw,    ki_yaw,    kd_yaw);
        pid_height_ = std::make_unique<PIDController>(kp_height, ki_height, kd_height);
        pid_dist_   = std::make_unique<PIDController>(kp_dist,   ki_dist,   kd_dist);

        // Image center — target for face to sit at
        // Hardcoded for 640x480; should be update if camera resolution changes
        image_center_x_ = 320;
        image_center_y_ = 240;

        // Gazebo positioning Service
        gazebo_client_ = this->create_client<gazebo_msgs::srv::SetEntityState>(
            "/gazebo/set_entity_state");

        RCLCPP_INFO(this->get_logger(), "Waiting for Gazebo SetEntityState service...");
        while (!gazebo_client_->wait_for_service(1s)) {
            RCLCPP_WARN(this->get_logger(), "Gazebo not ready yet...");
        }
        RCLCPP_INFO(this->get_logger(), "Gazebo service connected.");

        // Load Haar Cascade from path declared as ROS parameter
        if (!face_cascade_.load(cascade_path_)) {
            RCLCPP_ERROR(this->get_logger(),
                "Haar Cascade load failed at: %s", cascade_path_.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Haar Cascade loaded.");

        // Subscribe to drone's onboard camera via cv_bridge
        // This uses the actual Gazebo camera sensor output — not a webcam shortcut
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/stealth_quad/front_camera/image_raw",
            10,
            std::bind(&FaceTrackerNode::cameraCallback, this, _1)
        );
        RCLCPP_INFO(this->get_logger(), "Subscribed to drone camera topic.");

        // Move drone to initial hover before tracking begins
        moveDroneTo(0.0, 0.0, 1.5, 0.0);
        RCLCPP_INFO(this->get_logger(),
            "Drone set to hover at 1.5m. Face tracking active.");
    }

private:

    /**
     * Called every time a new frame arrives from the drone's camera.
     * Runs face detection, computes PID corrections, sends position command.
     */
    void cameraCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert ROS image message to OpenCV Mat via cv_bridge
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Detect faces — scaleFactor 1.1, minNeighbors 5 tuned for
        // indoor single-person tracking to reduce false positives
        std::vector<cv::Rect> faces;
        face_cascade_.detectMultiScale(
            gray, faces, 1.1, 5, 0, cv::Size(60, 60));

        // Draw targeting crosshair at image center
        cv::line(frame,
            cv::Point(image_center_x_-20, image_center_y_),
            cv::Point(image_center_x_+20, image_center_y_),
            cv::Scalar(0,255,0), 2);
        cv::line(frame,
            cv::Point(image_center_x_, image_center_y_-20),
            cv::Point(image_center_x_, image_center_y_+20),
            cv::Scalar(0,255,0), 2);

        if (!faces.empty()) {
            face_lost_count_ = 0;

            // Track largest face — most likely the primary subject
            cv::Rect target = *std::max_element(
                faces.begin(), faces.end(),
                [](const cv::Rect& a, const cv::Rect& b){
                    return a.area() < b.area();
                });

            int x = target.x, y = target.y;
            int w = target.width, h = target.height;
            int face_cx = x + w/2;
            int face_cy = y + h/2;

            // Pixel errors from image center
            double error_x = face_cx - image_center_x_;  // +ve = face right of center
            double error_y = face_cy - image_center_y_;  // +ve = face below center
            double error_z = w - target_face_width_;      // +ve = face too close

            // PID corrections
            double corr_strafe = pid_yaw_->compute(error_x);
            double corr_height = pid_height_->compute(-error_y); // invert: face up = drone up
            double corr_dist   = pid_dist_->compute(-error_z);   // invert: face big = back off

            auto clamp = [](double v, double lo, double hi){
                return std::max(lo, std::min(hi, v));
            };

            // Apply corrections with per_step limits to prevent sudden jumps
            drone_y_ -= clamp(corr_strafe, -0.05, 0.05); // left/right
            drone_z_ += clamp(corr_height, -0.05, 0.05); // altitude
            drone_x_ += clamp(corr_dist,   -0.05, 0.05); // forward/back

            // Safety floor at 0.3m to avoid ground turbulence in the simulator
            drone_z_ = std::max(min_height_, std::min(max_height_, drone_z_));
            drone_x_ = std::max(-max_xy_range_, std::min(max_xy_range_, drone_x_));
            drone_y_ = std::max(-max_xy_range_, std::min(max_xy_range_, drone_y_));

            moveDroneTo(drone_x_, drone_y_, drone_z_, drone_yaw_);

            // Draw bounding box and tracking line for the face
            cv::rectangle(frame, cv::Point(x,y), cv::Point(x+w,y+h),
                cv::Scalar(0,255,0), 2);
            cv::line(frame, cv::Point(face_cx, face_cy),
                cv::Point(image_center_x_, image_center_y_),
                cv::Scalar(255,0,0), 1);
            cv::putText(frame, "TRACKING",
                cv::Point(x, y-10), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0,255,0), 2);
            cv::putText(frame,
                "ErrX:" + std::to_string((int)error_x) +
                " Z:" + std::to_string(drone_z_).substr(0,4) + "m",
                cv::Point(10,25), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0,255,0), 1);

        } else {
            face_lost_count_++;
            std::string status = (face_lost_count_ < 30) ?
                "SEARCHING..." : "FACE LOST — HOLDING POSITION";
            cv::putText(frame, status, cv::Point(10,30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
        }

        cv::imshow("Stealth Quad — Drone Camera", frame);
        cv::waitKey(1);
    }

    void moveDroneTo(double x, double y, double z, double yaw)
    {
        auto req = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
        req->state.name = "stealth_quad";
        req->state.pose.position.x = x;
        req->state.pose.position.y = y;
        req->state.pose.position.z = z;
        req->state.pose.orientation.x = 0.0;
        req->state.pose.orientation.y = 0.0;
        req->state.pose.orientation.z = std::sin(yaw / 2.0);
        req->state.pose.orientation.w = std::cos(yaw / 2.0);
        gazebo_client_->async_send_request(req);
    }

    // Drone state
    double drone_x_, drone_y_, drone_z_, drone_yaw_;
    int face_lost_count_;
    int target_face_width_;
    int image_center_x_, image_center_y_;

    // Safety limits loaded from parameters.
    double min_height_, max_height_, max_xy_range_;

    // Path from parameters
    std::string cascade_path_;

    // PID controllers — unique_ptr so they can be constructed after parameters load without maual freeing
    std::unique_ptr<PIDController> pid_yaw_;
    std::unique_ptr<PIDController> pid_height_;
    std::unique_ptr<PIDController> pid_dist_;

    // ROS 2 interfaces
    rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr gazebo_client_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;

    // OpenCV 
    cv::CascadeClassifier face_cascade_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FaceTrackerNode>());
    rclcpp::shutdown();
    return 0;
}