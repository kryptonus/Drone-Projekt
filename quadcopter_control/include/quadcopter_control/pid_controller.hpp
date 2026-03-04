#ifndef QUADCOPTER_CONTROL_PID_CONTROLLER_HPP
#define QUADCOPTER_CONTROL_PID_CONTROLLER_HPP

#include <chrono>
#include <algorithm>

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

        double P = kp_ * error;

        integral_ += error * dt;
        integral_ = std::max(-50.0, std::min(50.0, integral_));
        double I = ki_ * integral_;

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

#endif
