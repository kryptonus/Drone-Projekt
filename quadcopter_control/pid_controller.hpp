#ifndef QUADCOPTER_CONTROL_PID_CONTROLLER_HPP
#define QUADCOPTER_CONTROL_PID_CONTROLLER_HPP

#include <chrono>
#include <algorithm>

/**
 * Discrete PID Controller
 * 
 * Tuning notes (Vaishnav, TH Mannheim):
 * - Kp too high → drone oscillates around target face
 * - Kd too high → sluggish response, drone barely moves
 * - Ki too high → integral windup, drone drifts even when face is centered
 * 
 * Current gains tuned empirically in Gazebo at 1.5m hover height.
 * Real hardware will need retuning due to motor lag and air disturbance.
 */
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
        // dt in seconds — needed for proper integral/derivative scaling
        double dt = std::chrono::duration<double>(now - last_time_).count();
        if (dt <= 0.0) dt = 0.033; // fallback to 30Hz if clock glitches

        double P = kp_ * error;

        // Clamp integral to [-50, 50] to prevent windup on long tracking loss
        integral_ += error * dt;
        integral_ = std::max(-50.0, std::min(50.0, integral_));
        double I = ki_ * integral_;

        // Derivative on error — not on measurement, avoids derivative kick
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

#endif // QUADCOPTER_CONTROL_PID_CONTROLLER_HPP