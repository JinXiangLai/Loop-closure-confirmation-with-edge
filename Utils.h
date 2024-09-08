#ifndef UTILS
#define UTILS

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>

constexpr int kImageWidth = 192;
constexpr int kImageHeight = 108;
constexpr double kRad2Deg = 180/M_PI;
constexpr double kDeg2Rad = M_PI/180;
constexpr int kZnum = 4;
constexpr double kZ[kZnum] = {2.1, 2.2, 2.3, 2.4};


class Pose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Pose(const Eigen::Quaterniond& q_wb, const Eigen::Vector3d& t_wb);
    Pose Inverse() const;
    Eigen::Vector3d operator*(const Eigen::Vector3d &p) const;
    Pose operator*(const Pose& T) const;
    friend std::ostream& operator<<(std::ostream &cout, const Pose& T);
    int Size() const;
    void Update(const Eigen::Vector3d &delta_q, const Eigen::Vector3d &delta_t);

    Eigen::Quaterniond q_wb_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_wb_ = Eigen::Vector3d::Zero();
};

// 定义相机模型
class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 实现不考虑畸变参数
    Camera();
    Eigen::Vector2d Project2PixelPlane(const Eigen::Vector3d &Pc) const;
    Eigen::Vector3d InverseProject(const Eigen::Vector2i &uv, const double &z = 1.0) const;

    Eigen::Matrix3d K_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d K_inv_ = Eigen::Matrix3d::Identity();
private:
    double fx_, fy_, cx_, cy_;

};

class Landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Landmark(const Eigen::Vector2d &px, std::shared_ptr<Pose> Twc, const std::shared_ptr<Camera> cam, 
        const double z = 1.0);
    Eigen::Vector3d GetPcNorm() const;
    Eigen::Vector3d GetPc() const;
    Eigen::Vector3d GetPw() const;
    int Size() const; // 优化变量的维度
    void Update(const double delta_z, const bool useInvDepth);

    Eigen::Vector2d uv_; // 像素坐标
    double z_ = 1.0;
    double invZ_ = 1.0;
    // anchor pose
    std::shared_ptr<Pose> Twc_;
    std::shared_ptr<Camera> cam_;
};

// 距离变换是计算前景到背景的距离
cv::Mat GenerateEdgeImage(std::vector<Eigen::Vector2d> &blackPoint);

cv::Mat GetDistanceTransform(cv::Mat img);

std::vector<Eigen::Vector2d> GenerateBlackPoint();

std::vector<Eigen::Vector3d> GeneratePw(const std::vector<Eigen::Vector2d> &blackPoint, const std::shared_ptr<Camera> &cam);

std::vector<Eigen::Vector3d> TransformPoint2Pc(const Pose &T, std::vector<Eigen::Vector3d> &ps);

void CaculateDerivative(const cv::Mat &dist, cv::Mat &dx, cv::Mat &dy);

bool InRange(const cv::Mat &img, const Eigen::Vector2i &p);

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v);

double BilinearInterpolate(const cv::Mat &img, const Eigen::Vector2d &p);

void Assert(bool a, const std::string &s);

void ShowImage(const cv::Mat &img, const std::string &name, const bool show = true);

Eigen::Vector3d Quat2RPY(const Eigen::Quaterniond &_q);

std::ostream& operator<<(std::ostream &cout, const Pose& T);

void DrawMatch(const cv::Mat &img1, const cv::Mat &img2, const std::vector<Eigen::Vector2d> &kp1, 
    const std::vector<Eigen::Vector2d> &kp2, const std::string &name = "matches");

std::vector<Eigen::Vector2d> FindMatches(const Eigen::Vector2d &kp1, const cv::Mat &edgeImg, const Pose &T21, const Camera &cam);

Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam);

Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp1, const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam);

Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp1, const std::vector<Eigen::Vector2d> &kp2, const Pose &T21, const Camera &cam);

Pose ConvertRPYandPostion2Pose(const Eigen::Vector3d &rpy, const Eigen::Vector3d &t, const double deg2rad = kDeg2Rad);

void varifyTriangulate();
#endif