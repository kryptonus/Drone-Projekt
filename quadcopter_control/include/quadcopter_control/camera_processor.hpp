#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <optional>

// Holds the result of one frame's face detection
struct FaceResult {
    int face_cx;
    int face_cy;
    int face_width;
    cv::Rect bounding_box;
};

// Owns the webcam and Haar Cascade — single responsibility
class CameraProcessor
{
public:
    CameraProcessor(const std::string& cascade_path, int frame_w = 640, int frame_h = 480)
        : frame_w_(frame_w), frame_h_(frame_h),
          center_x_(frame_w / 2), center_y_(frame_h / 2)
    {
        if (!cascade_.load(cascade_path))
            throw std::runtime_error("Failed to load Haar Cascade: " + cascade_path);

        cap_.open(0);
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  frame_w_);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, frame_h_);

        if (!cap_.isOpened())
            throw std::runtime_error("Cannot open webcam!");
    }

    ~CameraProcessor()
    {
        cap_.release();
        cv::destroyAllWindows();
    }

    // Returns largest detected face, or std::nullopt if none found
    std::optional<FaceResult> detect()
    {
        cv::Mat frame;
        if (!cap_.read(frame) || frame.empty()) return std::nullopt;

        cv::flip(frame, frame, 1);
        last_frame_ = frame.clone();

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> faces;
        cascade_.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(60, 60));

        if (faces.empty()) return std::nullopt;

        // Pick largest face
        cv::Rect target = *std::max_element(faces.begin(), faces.end(),
            [](const cv::Rect& a, const cv::Rect& b){ return a.area() < b.area(); });

        return FaceResult{
            target.x + target.width  / 2,
            target.y + target.height / 2,
            target.width,
            target
        };
    }

    void drawHUD(const std::optional<FaceResult>& face, 
                 double drone_z, int face_lost_count)
    {
        if (last_frame_.empty()) return;
        cv::Mat display = last_frame_.clone();

        // Crosshair
        cv::line(display, {center_x_-20, center_y_}, {center_x_+20, center_y_}, {0,255,0}, 2);
        cv::line(display, {center_x_, center_y_-20}, {center_x_, center_y_+20}, {0,255,0}, 2);

        if (face) {
            int x = face->bounding_box.x, y = face->bounding_box.y;
            int w = face->bounding_box.width, h = face->bounding_box.height;
            double error_x = face->face_cx - center_x_;

            cv::rectangle(display, {x,y}, {x+w,y+h}, {0,255,0}, 2);
            cv::line(display, {face->face_cx, face->face_cy}, {center_x_, center_y_}, {255,0,0}, 1);
            cv::putText(display, "TRACKING", {x, y-10}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
            cv::putText(display,
                "ErrX:" + std::to_string((int)error_x) +
                " Z:" + std::to_string(drone_z).substr(0,4) + "m",
                {10,25}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 1);
        } else {
            std::string status = (face_lost_count < 30) ? "SEARCHING..." : "FACE LOST - HOLDING";
            cv::putText(display, status, {10,30}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,255}, 2);
        }

        cv::imshow("Face Tracking HUD", display);
        cv::waitKey(1);
    }

    int centerX() const { return center_x_; }
    int centerY() const { return center_y_; }

private:
    cv::VideoCapture    cap_;
    cv::CascadeClassifier cascade_;
    cv::Mat             last_frame_;
    int frame_w_, frame_h_, center_x_, center_y_;
};
