#include "rclcpp/rclcpp.hpp"
#include "quadcopter_control/pid_controller.hpp"
#include "quadcopter_control/camera_processor.hpp"
#include "quadcopter_control/drone_controller.hpp"
#include <memory>

using namespace std::chrono_literals;

// FaceTrackerNode — thin orchestrator
// Owns: CameraProcessor, DroneController, 3x PIDController
// Does: connects them together, nothing else
class FaceTrackerNode : public rclcpp::Node
{
public:
    FaceTrackerNode() : Node("face_tracker_node"), face_lost_count_(0)
    {
        // ── Load params ──────────────────────────────────────────────
        double kp_yaw    = declare_parameter("kp_yaw",    0.02);
        double ki_yaw    = declare_parameter("ki_yaw",    0.0);
        double kd_yaw    = declare_parameter("kd_yaw",    0.003);
        double kp_height = declare_parameter("kp_height", 0.003);
        double ki_height = declare_parameter("ki_height", 0.0001);
        double kd_height = declare_parameter("kd_height", 0.002);
        double kp_dist   = declare_parameter("kp_dist",   0.005);
        double ki_dist   = declare_parameter("ki_dist",   0.0);
        double kd_dist   = declare_parameter("kd_dist",   0.002);

        target_face_width_ = declare_parameter("target_face_width", 80);
        auto cascade_path  = declare_parameter("cascade_path",
            std::string("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"));
        double min_h  = declare_parameter("min_height",   0.3);
        double max_h  = declare_parameter("max_height",   4.0);
        double max_xy = declare_parameter("max_xy_range", 5.0);

        // ── Build PID controllers (unique_ptr — node is sole owner) ──
        pid_yaw_    = std::make_unique<PIDController>(kp_yaw,    ki_yaw,    kd_yaw);
        pid_height_ = std::make_unique<PIDController>(kp_height, ki_height, kd_height);
        pid_dist_   = std::make_unique<PIDController>(kp_dist,   ki_dist,   kd_dist);

        // ── Build subsystems ─────────────────────────────────────────
        // shared_ptr: CameraProcessor could later be shared with a recorder node
        camera_ = std::make_shared<CameraProcessor>(cascade_path);

        // unique_ptr: only this node controls the drone
        drone_ = std::make_unique<DroneController>(this, min_h, max_h, max_xy);
        drone_->waitForGazebo();
        drone_->hover();

        RCLCPP_INFO(get_logger(), "Hovering. Tracking active.");

        timer_ = create_wall_timer(33ms,
            std::bind(&FaceTrackerNode::trackingLoop, this));
    }

private:
    void trackingLoop()
    {
        auto face = camera_->detect();

        if (face) {
            face_lost_count_ = 0;

            double error_x = face->face_cx  - camera_->centerX();
            double error_y = face->face_cy  - camera_->centerY();
            double error_z = face->face_width - target_face_width_;

            const double dead = 40.0;
            double corr_y = pid_yaw_->compute(std::abs(error_x) > dead ? error_x  : 0.0);
            double corr_z = pid_height_->compute(std::abs(error_y) > dead ? -error_y : 0.0);
            double corr_x = pid_dist_->compute(-error_z);

            drone_->applyCorrections(corr_x, corr_y, corr_z);

            RCLCPP_INFO(get_logger(), "ErrX=%.1f ErrY=%.1f ErrZ=%.1f DroneZ=%.3f",
                error_x, error_y, error_z, drone_->getZ());
        } else {
            face_lost_count_++;
        }

        camera_->drawHUD(face, drone_->getZ(), face_lost_count_);
    }

    // Subsystems
    std::shared_ptr<CameraProcessor>  camera_;   // shared_ptr — reusable by other nodes
    std::unique_ptr<DroneController>  drone_;    // unique_ptr — only we control it

    // PID controllers
    std::unique_ptr<PIDController> pid_yaw_;
    std::unique_ptr<PIDController> pid_height_;
    std::unique_ptr<PIDController> pid_dist_;

    rclcpp::TimerBase::SharedPtr timer_;
    int face_lost_count_;
    int target_face_width_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FaceTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
