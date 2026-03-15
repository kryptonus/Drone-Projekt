#pragma once

#include "rclcpp/rclcpp.hpp"
#include "gazebo_msgs/srv/set_entity_state.hpp"
#include <cmath>
#include <algorithm>

// Owns drone position state and Gazebo communication
class DroneController
{
public:
    DroneController(rclcpp::Node* node,
                    double min_height, double max_height, double max_xy_range)
        : node_(node),
          x_(0.0), y_(0.0), z_(1.5), yaw_(0.0),
          min_height_(min_height),
          max_height_(max_height),
          max_xy_range_(max_xy_range)
    {
        client_ = node_->create_client<gazebo_msgs::srv::SetEntityState>(
            "/gazebo/set_entity_state");
    }

    bool waitForGazebo()
    {
        using namespace std::chrono_literals;
        RCLCPP_INFO(node_->get_logger(), "Waiting for Gazebo...");
        while (!client_->wait_for_service(1s)) {
            RCLCPP_WARN(node_->get_logger(), "Gazebo not ready...");
        }
        RCLCPP_INFO(node_->get_logger(), "Gazebo connected.");
        return true;
    }

    void applyCorrections(double corr_x, double corr_y, double corr_z)
    {
        x_ += clamp(corr_x, -0.05, 0.05);
        y_ += clamp(corr_y, -0.05, 0.05);
        z_ += clamp(corr_z, -0.05, 0.05);

        // Enforce safety bounds
        z_ = std::clamp(z_, min_height_, max_height_);
        x_ = std::clamp(x_, -max_xy_range_, max_xy_range_);
        y_ = std::clamp(y_, -max_xy_range_, max_xy_range_);

        sendToGazebo();
    }

    void hover() { sendToGazebo(); }

    double getZ() const { return z_; }

private:
    void sendToGazebo()
    {
        auto req = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
        req->state.name = "stealth_quad";
        req->state.pose.position.x = x_;
        req->state.pose.position.y = y_;
        req->state.pose.position.z = z_;
        req->state.pose.orientation.x = 0.0;
        req->state.pose.orientation.y = 0.0;
        req->state.pose.orientation.z = std::sin(yaw_ / 2.0);
        req->state.pose.orientation.w = std::cos(yaw_ / 2.0);
        client_->async_send_request(req);
    }

    static double clamp(double v, double lo, double hi)
    {
        return std::max(lo, std::min(hi, v));
    }

    rclcpp::Node* node_;
    rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr client_;
    double x_, y_, z_, yaw_;
    double min_height_, max_height_, max_xy_range_;
};
