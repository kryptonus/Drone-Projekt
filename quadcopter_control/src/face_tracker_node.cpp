/**
 * face_tracker_node.cpp
 * 
 * My face tracking drone controller — ROS 2 / Gazebo
 * Vaishnav - IEB - TH Mannheim 
 * 
 * How it works:
 * The drone's onboard camera sees my face.
 * I calculate how far my face is from the image center (the error).
 * Three PID controllers convert those errors into position corrections.
 * Gazebo moves the drone accordingly.
 * 
 * Why PID and not just proportional?
 * Pure P control causes oscillation — drone overshoots and wobbles.
 * D term acts like brakes — slows correction as face approaches center.
 * I term fixes stubborn small offsets that P alone never fully corrects.
 * 
 * Why Haar Cascade and not YOLO?
 * Haar runs at 30Hz on CPU with zero GPU. Good enough for single person
 * tracking under 3 meters. YOLO is planned for Stage 5 when I need
 * multi-target classification, not just face location. The final advanced version is the real drone.
 * 
 * Known issues I found during testing:
 * - Bright windows behind me cause false negatives (Haar limitation)
 * - equalizeHist() helps but does not fully fix it
 * - At distances over 2m face bounding box gets too small → tracking unstable
 * - Solution: reduce target_face_width param or move closer
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
        // Load PID gains from YAML — I can tune these without recompiling.
        // Spent a while tuning these. Too high Kp = drone oscillates like crazy.
        // Too high Kd = drone barely moves, feels sluggish.
        // Current values work well at about 1-1.5m distance from camera.
        double kp_yaw    = this->declare_parameter("kp_yaw",    0.003);
        double ki_yaw    = this->declare_parameter("ki_yaw",    0.0001);
        double kd_yaw    = this->declare_parameter("kd_yaw",    0.002);
        double kp_height = this->declare_parameter("kp_height", 0.003);
        double ki_height = this->declare_parameter("ki_height", 0.0001);
        double kd_height = this->declare_parameter("kd_height", 0.002);
        double kp_dist   = this->declare_parameter("kp_dist",   0.005);
        double ki_dist   = this->declare_parameter("ki_dist",   0.0001);
        double kd_dist   = this->declare_parameter("kd_dist",   0.002);

        // Target face width in pixels at ideal hover distance (~1.5m)
        // Measured empirically: at 1.5m my face is ~150px wide in 640x480
        target_face_width_ = this->declare_parameter("target_face_width", 150);

        // Cascade path as param so I can swap detectors without recompiling
        cascade_path_ = this->declare_parameter("cascade_path",
            std::string("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"));

        // Safety limits — drone_z floor at 0.3m because below that
        // Gazebo ground effect physics gets weird and drone bounces
        min_height_   = this->declare_parameter("min_height",   0.3);
        max_height_   = this->declare_parameter("max_height",   4.0);
        max_xy_range_ = this->declare_parameter("max_xy_range", 5.0);

        // Three separate PIDs — one per axis of movement
        // Could use one PID with a 3D error vector but keeping them
        // separate makes tuning much easier (change one axis at a time)
        pid_yaw_    = std::make_unique<PIDController>(kp_yaw,    ki_yaw,    kd_yaw);
        pid_height_ = std::make_unique<PIDController>(kp_height, ki_height, kd_height);
        pid_dist_   = std::make_unique<PIDController>(kp_dist,   ki_dist,   kd_dist);

        // Image center = where I want the face to be
        // Hardcoded for 640x480 — matches camera plugin in URDF
        image_center_x_ = 320;
        image_center_y_ = 240;

        // Connect to Gazebo's entity state service
        // This is what actually moves the drone in simulation
        // On real hardware this would be replaced by MAVLink or cmd_vel
        gazebo_client_ = this->create_client<gazebo_msgs::srv::SetEntityState>(
            "/gazebo/set_entity_state");

        RCLCPP_INFO(this->get_logger(), "Waiting for Gazebo SetEntityState service...");
        while (!gazebo_client_->wait_for_service(1s)) {
            RCLCPP_WARN(this->get_logger(), "Gazebo not ready yet...");
        }
        RCLCPP_INFO(this->get_logger(), "Gazebo connected.");

        // Load Haar Cascade
        // detectMultiScale uses sliding window at multiple image scales
        // finds face-like patterns based on Viola-Jones feature training
        if (!face_cascade_.load(cascade_path_)) {
            RCLCPP_ERROR(this->get_logger(),
                "Failed to load cascade from: %s", cascade_path_.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Haar Cascade loaded.");

        // Subscribe to drone's actual onboard camera
        // This is the real camera sensor output from Gazebo plugin
        // NOT a webcam — the drone sees what it would see in real life
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/stealth_quad/front_camera/image_raw", 10,
            std::bind(&FaceTrackerNode::cameraCallback, this, _1)
        );
        RCLCPP_INFO(this->get_logger(), "Subscribed to drone camera.");

        // Lift drone to hover before tracking starts, you can also do this the other way.
        moveDroneTo(0.0, 0.0, 1.5, 0.0);
        RCLCPP_INFO(this->get_logger(), "Hovering at 1.5m. Tracking active.");
    }

private:

    void cameraCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Step 1: Convert ROS image message to OpenCV Mat
        // cv_bridge handles the format conversion (encoding, timestamp etc)
        // "bgr8" = Blue Green Red, 8 bits per channel — standard OpenCV format
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;

        // Step 2: Convert to grayscale
        // Haar Cascade works on intensity gradients, not color since color information is irrelevant for edge-based face detection
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // equalizeHist spreads intensity values across full 0-255 range
        // This helps when lighting is uneven — tested in sim with dark corners
        cv::equalizeHist(gray, gray);

        // Step 3: Detect faces
        // scaleFactor=1.1 → scan image at 10% smaller each pass
        // minNeighbors=5 → a detection needs 5 overlapping detections to count
        //   (reduces false positives — tuned by trial and error)
        // minSize=60x60 → ignore tiny detections (background noise)
        std::vector<cv::Rect> faces;
        face_cascade_.detectMultiScale(
            gray, faces, 1.1, 5, 0, cv::Size(60, 60));

        // Draw crosshair at image center — this is my target position
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

            // Step 4: Pick the largest face
            // In a single person scenario, largest = closest = the target
            // Multi-person tracking would need ID assignment (future work)
            cv::Rect target = *std::max_element(
                faces.begin(), faces.end(),
                [](const cv::Rect& a, const cv::Rect& b){
                    return a.area() < b.area();
                });

            int x = target.x, y = target.y;
            int w = target.width, h = target.height;
            int face_cx = x + w/2;
            int face_cy = y + h/2;

            // Step 5: Calculate pixel errors from image center
            // error_x: positive = face is RIGHT of center → drone strafes right
            // error_y: positive = face is BELOW center → drone descends
            // error_z: positive = face wider than target → drone is too close
            double error_x = face_cx - image_center_x_;
            double error_y = face_cy - image_center_y_;
            double error_z = w - target_face_width_;

            // Step 6: Run PID controllers
            // Each error goes into its own PID and comes out as a position correction
            // Note the sign inversions:
            //   -error_y because image Y goes DOWN but drone Z goes UP
            //   -error_z because bigger face means drone should move BACK
            double corr_strafe = pid_yaw_->compute(error_x);
            double corr_height = pid_height_->compute(-error_y);
            double corr_dist   = pid_dist_->compute(-error_z);

            // Step 7: Clamp corrections
            // Max 0.05m movement per callback (30Hz = 1.5m/s maximum)
            // Without this limit, large initial errors cause the drone to teleport or move irregularly.
                auto clamp = [](double v, double lo, double hi){
                return std::max(lo, std::min(hi, v));
            };

            drone_y_ -= clamp(corr_strafe, -0.05, 0.05); // strafe
            drone_z_ += clamp(corr_height, -0.05, 0.05); // altitude
            drone_x_ += clamp(corr_dist,   -0.05, 0.05); // forward/back

            // Step 8: Hard safety limits
            // Floor at 0.3m — below this Gazebo ground physics gets unstable
            drone_z_ = std::max(min_height_, std::min(max_height_, drone_z_));
            drone_x_ = std::max(-max_xy_range_, std::min(max_xy_range_, drone_x_));
            drone_y_ = std::max(-max_xy_range_, std::min(max_xy_range_, drone_y_));

            // Step 9: Send updated position to Gazebo
            moveDroneTo(drone_x_, drone_y_, drone_z_, drone_yaw_);

            // Visualize tracking
            cv::rectangle(frame, cv::Point(x,y), cv::Point(x+w,y+h),
                cv::Scalar(0,255,0), 2);
            cv::line(frame, cv::Point(face_cx,face_cy),
                cv::Point(image_center_x_,image_center_y_),
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
            // Face lost — hold current position and don't drift
            face_lost_count_++;
            std::string status = (face_lost_count_ < 30) ?
                "SEARCHING..." : "FACE LOST - HOLDING";
            cv::putText(frame, status, cv::Point(10,30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
        }

        cv::imshow("Stealth Quad — Drone POV", frame);
        cv::waitKey(1);
    }

    void moveDroneTo(double x, double y, double z, double yaw)
    {
        // Build service request with target position
        // On real hardware: this maps to MAVLink SET_POSITION_TARGET_LOCAL_NED
        auto req = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
        req->state.name = "stealth_quad";
        req->state.pose.position.x = x;
        req->state.pose.position.y = y;
        req->state.pose.position.z = z;

        // Convert yaw angle to quaternion
        // For pure yaw: qx=0, qy=0, qz=sin(yaw/2), qw=cos(yaw/2)
        // Derived from quaternion rotation formula around Z axis
        req->state.pose.orientation.x = 0.0;
        req->state.pose.orientation.y = 0.0;
        req->state.pose.orientation.z = std::sin(yaw / 2.0);
        req->state.pose.orientation.w = std::cos(yaw / 2.0);

        // Async — don't block tracking loop waiting for Gazebo to respond
        gazebo_client_->async_send_request(req);
    }

    // Drone state — current position in Gazebo world frame
    double drone_x_, drone_y_, drone_z_, drone_yaw_;
    int face_lost_count_;
    int target_face_width_;
    int image_center_x_, image_center_y_;

    // Safety bounds from params
    double min_height_, max_height_, max_xy_range_;
    std::string cascade_path_;

    // unique_ptr because PIDController needs params before construction
    // Cannot use direct member — params only available after Node() runs
    std::unique_ptr<PIDController> pid_yaw_;
    std::unique_ptr<PIDController> pid_height_;
    std::unique_ptr<PIDController> pid_dist_;

    rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr gazebo_client_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    cv::CascadeClassifier face_cascade_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FaceTrackerNode>());
    rclcpp::shutdown();
    return 0;
}