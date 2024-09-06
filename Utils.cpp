#include "Utils.h"

using namespace cv;
using namespace std;

// ============ class Pose ======== //
// ================================ //
Pose::Pose(const Eigen::Quaterniond &q_wb, const Eigen::Vector3d &t_wb)
    : q_wb_(q_wb)
    , t_wb_(t_wb) {}

Pose Pose::Inverse() const {
    const Eigen::Quaterniond q_bw = q_wb_.inverse();
    const Eigen::Vector3d t_bw = -(q_bw * t_wb_);
    return Pose(q_bw, t_bw);
}

Eigen::Vector3d Pose::operator*(const Eigen::Vector3d &p) const {
    return q_wb_ * p + t_wb_;
}

int Pose::Size() const {return 6;}

void Pose::Update(const Eigen::Vector3d &delta_q, const Eigen::Vector3d &delta_t) {
    const Eigen::Matrix3d deltaR = Eigen::AngleAxisd(delta_q.norm(), delta_q.normalized()).toRotationMatrix();
    q_wb_ *= Eigen::Quaterniond(deltaR);
    t_wb_ += delta_t;
}


// ======== class Camera ======== //
// ================================ //
Camera::Camera()
    : fx_(kImageWidth * 0.25)
    , fy_(kImageHeight * 0.25)
    , cx_(kImageWidth * 0.5)
    , cy_(kImageHeight * 0.5) {
        K_ << fx_, 0, cx_,
              0,  fy_, cy_,
              0, 0, 1;
        K_inv_ = K_.inverse();
}

Eigen::Vector2d Camera::Project2PixelPlane(const Eigen::Vector3d &Pc) {
    const Eigen::Vector3d p_norm = Pc/Pc.z();
    Eigen::Vector2d res{fx_ * p_norm[0] + cx_, fy_ * p_norm[1] + cy_};
    return res;
}

Eigen::Vector3d Camera::InverseProject(const Eigen::Vector2i &uv, const double &z) const{
    Eigen::Vector3d p(uv[0], uv[1], 1.0);
    p = K_inv_ * p;
    // std::cout << "K.inv * p: " << p.transpose() << std::endl;
    return p * z;
}


// ======== class Landmark ======== //
// ================================ //
Landmark::Landmark(const Eigen::Vector2d &px, std::shared_ptr<Pose> Twc, const std::shared_ptr<Camera> cam, 
    const double z)
    : z_(z)
    , uv_(px)
    , Twc_(Twc)
    , cam_(cam) {}

Eigen::Vector3d Landmark::GetPcNorm() const {
    return cam_->InverseProject(uv_.cast<int>(), 1.0);
}

Eigen::Vector3d Landmark::GetPc() const {
    return cam_->InverseProject(uv_.cast<int>(), z_);
}

Eigen::Vector3d Landmark::GetPw() const {
    return (*Twc_) * GetPc();
}

int Landmark::Size() const {return 1;}

void Landmark::Update(const double delta_z, const bool useInvDepth) {
    if(useInvDepth) {
        invZ_ += delta_z;
        z_ = 1/invZ_;
    } else {
        z_ += delta_z; 
        invZ_ = 1/z_;
    }
}


// ======== utils functions ======== //
// ================================ //

// 距离变换是计算前景到背景的距离
Mat GenerateEdgeImage(vector<Eigen::Vector2d> &blackPoint) {
    Mat img(kImageHeight, kImageWidth, CV_8UC1, 255);
    for(Eigen::Vector2d &p : blackPoint) {
        // at(row, col)
        if(InRange(img, p.cast<int>()) ) {
            img.at<uchar>(int(p.y()), int(p.x())) = 0;
        }
    }
    cv::erode(img, img, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    return img;
}

Mat GetDistanceTransform(Mat img) {
    Mat res;
    // https://blog.csdn.net/kakiebu/article/details/82967085
    distanceTransform(img, res, DIST_L2, 0);
    Assert(res.type() == CV_32FC1, "Assert distance transform type error!");
    // TODO: 是否需要归一化呢
    // cv::normalize(res, res, 0, 1);
    return res;
}

vector<Eigen::Vector2d> GenerateBlackPoint() {
    // 生成一个矩形边框
    const int row1 = kImageHeight * 0.25, row2 = kImageHeight * 0.75,
              col1 = kImageWidth * 0.25, col2 = kImageWidth * 0.75;

    vector<Eigen::Vector2d> res;
    for(int i = 0; i < (col2 - col1); ++i) {
        res.push_back({col1 + i, row1});
        res.push_back({col1 + i, row2});
    }
    for(int i = 0; i < (row2 - row1); ++i) {
        res.push_back({col1, row1 + i});
        res.push_back({col2, row1 + i});
    }
    return res;
}

vector<Eigen::Vector3d> GeneratePw(const vector<Eigen::Vector2d> &blackPoint, const std::shared_ptr<Camera> &cam) {
    vector<Eigen::Vector3d> Pw;
    double z[10] = {2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0};
    int id = 0;
    for(const Eigen::Vector2d &p : blackPoint) {
        // cout << "id++%4: " << id++%4 << endl; --id;
        Pw.push_back({cam->InverseProject(p.cast<int>(), z[(id++)%10])});
    }
    return Pw;
}

vector<Eigen::Vector3d> TransformPoint2Pc(const Pose &T, vector<Eigen::Vector3d> &ps) {
    vector<Eigen::Vector3d> pc;
    for(const Eigen::Vector3d &p : ps) {
        const Eigen::Vector3d t = T * p;
        Assert(t[2] > 0, "[ERROR] depth can't be negative");
        pc.push_back(t);
    }
    return pc;
}

void CaculateDerivative(const Mat &dist, Mat &dx, Mat &dy) {
    const int row = dist.rows;
    const int col = dist.cols;
    dx = Mat(row, col, CV_32FC1, 0.);
    dy = dx.clone();
    for(int i = 0; i < row; ++i) {
        // 遍历一行
        for(int j = 1; j < col-1; ++j) {
            // dx.at<float>(i, j) = 0.5 * (dist.at<float>(i, j+1) - dist.at<float>(i, j-1));
            dx.at<float>(i, j) = (dist.at<float>(i, j+1) - dist.at<float>(i, j));
        }
    }
    for(int j = 0; j < col; ++j) {
        // 遍历一列
        for(int i = 1; i < row-1; ++i) {
            // dy.at<float>(i, j) = 0.5 * (dist.at<float>(i+1, j) - dist.at<float>(i-1, j));
            dy.at<float>(i, j) = (dist.at<float>(i+1, j) - dist.at<float>(i, j));
        }
    }
}

bool InRange(const cv::Mat &img, const Eigen::Vector2i &p) {
    // 边缘行、列忽略
    return p.x() >= 1 && p.x() < img.cols-1 && p.y() >= 1 && p.y() < img.rows-1;
}

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v) {
    Eigen::Matrix3d m;
    m.setZero();
    m << 0, -v[2], v[1],
         v[2], 0, -v[0],
         -v[1], v[0], 0;
    return m;
}

double BilinearInterpolate(const cv::Mat &img, const Eigen::Vector2d &p) {
    if(!InRange(img, p.cast<int>())) {
        return 0;
    }
    
    const int col = img.cols;
    const int row = img.rows;
    const int x = int(p.x());
    const int y = int(p.y());
    if(x == col-1 || x == 0 || y == row-1 || y == 0) {
        return img.at<float>(y, x);
    }

    /****** 双线性插值 ******
    * +---+---+
    * + v1+ v2+
    * +---+---+
    * + v3+ v4+
    * +---+---+
    ***********************/
    float v1 = img.at<float>(y, x);
    float v2 = img.at<float>(y, x+1);
    float v3 = img.at<float>(y+1, x);
    float v4 = img.at<float>(y+1, x+1);
    const double wx = p.x() - x;
    const double wy = p.y() - y;
    const double w1 = (1-wx) * (1-wy);
    const double w2 = wx * (1-wy);
    const double w3 = (1-wx) * wy;
    const double w4 = wx * wy;
    return w1*v1 + w2*v2 + w3*v3 + w4*v4;
}

void Assert(bool a, const std::string &s) {
    if(!a) {
        cerr << s << endl;
        exit(-1);
    }
}

void ShowImage(const cv::Mat &img, const std::string &name, const bool show) {
    if(!show) {
        return;
    }
    Mat m = img.clone();

    m.convertTo(m, CV_32F);
    cv::normalize(m, m, 255, 0, cv::NORM_MINMAX);
    m.convertTo(m, CV_8UC1);
    cv::imshow(name, m);
    cv::waitKey(0);
}