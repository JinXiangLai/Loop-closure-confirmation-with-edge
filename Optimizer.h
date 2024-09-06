#ifndef OPTIMIZER
#define OPTIMIZER

#include "Utils.h"

class Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Optimizer(const std::vector<cv::Mat> &dist, const std::vector<cv::Mat> &dx, const std::vector<cv::Mat> &dy, 
        const double lambda = 1.0, const int maxIte = 100, const bool useInvDepth = false);
    bool Optimize(std::vector<Landmark> &pc1, std::vector<Pose> &T12);
    Eigen::VectorXd CalculateResidual(const std::vector<Landmark> &pc1, const std::vector<Pose> &T12);
    Eigen::MatrixXd CalculateJacobian(const std::vector<Landmark> &pc1, const std::vector<Pose> &T12);
private:
    double lambda_ = 1.0;
    int maxIte_ = 100;
    std::vector<cv::Mat> dist_, dx_, dy_;
    bool useInvDepth_ = false;
};

#endif