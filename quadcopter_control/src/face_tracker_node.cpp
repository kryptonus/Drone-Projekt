#include "rclcpp/rclcpp.hpp"
#include "gazebo_msgs/srv/set_entity_state.hpp"
#include "quadcopter_control/pid_controller.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <chrono>
#include <memory>
#include <cmath>
#include <string>

using namespace std::chrono_literals;

class FaceTrackerNode : public rclcpp::Node
{
public:
    FaceTrackerNode() : Node("face_tracker_node"),
        drone_x_(0.0), drone_y_(0.0),
        drone_z_(2.5), drone_yaw_(0.0),
        face_lost_count_(0)
    {
        // Load PID gains from YAML params
        double kp_yaw    = this->declare_parameter("kp_yaw",    0.02);
        double ki_yaw    = this->declare_parameter("ki_yaw",    0.0);
        double kd_yaw    = this->declare_parameter("kd_yaw",    0.003);
        double kp_height = this->declare_parameter("kp_height", 0.003);
        double ki_height = this->declare_parameter("ki_height", 0.0001);
        double kd_height = this->declare_parameter("kd_height", 0.002);
        double kp_dist   = this->declare_parameter("kp_dist",   0.005);
        double ki_dist   = this->declare_parameter("ki_dist",   0.0);
        double kd_dist   = this->declare_parameter("kd_dist",   0.002);

        target_face_width_ = this->declare_parameter("target_face_width", 80);
        cascade_path_      = this->declare_parameter("cascade_path",
            std::string("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"));
        min_height_   = this->declare_parameter("min_height",   0.3);
        max_height_   = this->declare_parameter("max_height",   4.0);
        max_xy_range_ = this->declare_parameter("max_xy_range", 5.0);

        pid_yaw_    = std::make_unique<PIDController>(kp_yaw,    ki_yaw,    kd_yaw);
        pid_height_ = std::make_unique<PIDController>(kp_height, ki_height, kd_height);
        pid_dist_   = std::make_unique<PIDController>(kp_dist,   ki_dist,   kd_dist);

        image_center_x_ = 320;
        image_center_y_ = 240;

        gazebo_client_ = this->create_client<gazebo_msgs::srv::SetEntityState>(
            "/gazebo/set_entity_state");

        RCLCPP_INFO(this->get_logger(), "Waiting for Gazebo...");
        while (!gazebo_client_->wait_for_service(1s)) {
            RCLCPP_WARN(this->get_logger(), "Gazebo not ready...");
        }
        RCLCPP_INFO(this->get_logger(), "Gazebo connected.");

        if (!face_cascade_.load(cascade_path_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load Haar Cascade!");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Haar Cascade loaded.");

        // Open webcam — face detection input
        cap_.open(0);
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open webcam!");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Webcam opened.");

        moveDroneTo(0.0, 0.0, 1.5, 0.0);
        RCLCPP_INFO(this->get_logger(), "Hovering at 2.5m. Tracking active.");

        timer_ = this->create_wall_timer(
            33ms, std::bind(&FaceTrackerNode::trackingLoop, this));
    }

    ~FaceTrackerNode()
    {
        cap_.release();
        cv::destroyAllWindows();
    }

private:
    void trackingLoop()
    {
        cv::Mat frame;
        if (!cap_.read(frame) || frame.empty()) return;
        cv::flip(frame, frame, 1);

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> faces;
        face_cascade_.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(60, 60));

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
            double error_x = face_cx - image_center_x_;
            double error_y = face_cy - image_center_y_;
            double error_z = w - target_face_width_;

            double dead = 40.0;
            double corr_strafe = pid_yaw_->compute(std::abs(error_x) > dead ? error_x : 0.0);
            double corr_height = pid_height_->compute(std::abs(error_y) > dead ? -error_y : 0.0);
            double corr_dist   = pid_dist_->compute(-error_z);

            auto clamp = [](double v, double lo, double hi){
                return std::max(lo, std::min(hi, v));
            };

            drone_y_ += clamp(corr_strafe, -0.05, 0.05);
            RCLCPP_INFO(this->get_logger(), "error_x=%.1f corr=%.4f drone_y=%.3f", error_x, corr_strafe, drone_y_);
            drone_z_ += clamp(corr_height, -0.05, 0.05);
            drone_x_ += clamp(corr_dist,   -0.05, 0.05);

            drone_z_ = std::max(min_height_, std::min(max_height_, drone_z_));
            drone_x_ = std::max(-max_xy_range_, std::min(max_xy_range_, drone_x_));
            drone_y_ = std::max(-max_xy_range_, std::min(max_xy_range_, drone_y_));

            moveDroneTo(drone_x_, drone_y_, drone_z_, drone_yaw_);

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
            face_lost_count_++;
            std::string status = (face_lost_count_ < 30) ?
                "SEARCHING..." : "FACE LOST - HOLDING";
            cv::putText(frame, status, cv::Point(10,30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
        }

        cv::imshow("Face Tracking HUD", frame);
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

    double drone_x_, drone_y_, drone_z_, drone_yaw_;
    int face_lost_count_, target_face_width_;
    int image_center_x_, image_center_y_;
    double min_height_, max_height_, max_xy_range_;
    std::string cascade_path_;

    std::unique_ptr<PIDController> pid_yaw_;
    std::unique_ptr<PIDController> pid_height_;
    std::unique_ptr<PIDController> pid_dist_;

    rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr gazebo_client_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
    cv::CascadeClassifier face_cascade_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FaceTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
