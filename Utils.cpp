#include "Utils.h"

#include <fstream>

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

Pose Pose::operator*(const Pose& T) const {
    return Pose(q_wb_ * T.q_wb_, q_wb_*T.t_wb_+t_wb_);
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
    : cx_(kImageWidth * 0.5)
    , cy_(kImageHeight * 0.5)
    , fx_(cx_ / kViewRangeX)
    , fy_(cy_ / kViewRangeY)
 {
        K_ << fx_, 0, cx_,
              0,  fy_, cy_,
              0, 0, 1;
        K_inv_ = K_.inverse();
}

Eigen::Vector2d Camera::Project2PixelPlane(const Eigen::Vector3d &Pc) const {
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
    , invZ_(1.0/z)
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
    const double backZ = z_;
    if(useInvDepth) {
        invZ_ += delta_z;
        z_ = 1.0/invZ_;
    } else {
        z_ += delta_z; 
        invZ_ = 1.0/z_;
    }
    if(z_ <= 0) {
        z_ = backZ;
        invZ_ = 1.0/z_;
    }
}

void Landmark::UpdateUncertainty() {
    uncertainty_ = depthRange_[1] - depthRange_[0];
    z_ = 0.5 * (depthRange_[0] + depthRange_[1]);
    // invZ_ = 1.0 / z_;
    invZ_ = 1.0;
}

// ======== utils functions ======== //
// ================================ //

// 距离变换是计算前景到背景的距离
Mat GenerateEdgeImage(vector<Eigen::Vector2d> &blackPoint) {
    Mat img(kImageHeight, kImageWidth, CV_8UC1, 255);
    constexpr int thickness = 1;
    for(int i = 1; i < blackPoint.size(); ++i) {
        const int x1 = blackPoint[i-1].x(), y1 = blackPoint[i-1].y(),
            x2 = blackPoint[i].x(), y2 = blackPoint[i].y();
        img.at<uchar>(y1, x1) = 0;
        img.at<uchar>(y2, x2) = 0;
        cv::line(img, {x1, y1}, {x2, y2}, 0, thickness);
        
        if(i == blackPoint.size() - 1) {
            const int x1 = blackPoint.front().x(), y1 = blackPoint.front().y();
            cv::line(img, {x1, y1}, {x2, y2}, 0, thickness);
        }
    }

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
    // 生成一个多边形顶点
    const int row1 = kImageHeight * 0.4, row2 = row1 + 100,
              col1 = kImageWidth * 0.4, col2 = col1 + 180;

    vector<Eigen::Vector2d> res;
    res.push_back({col1, row1});
    res.push_back({col2, row1});
    res.push_back({col2, row2});
    res.push_back({col1, row2});

    res.push_back({col1 + 10, row2 + 20});
    res.push_back({col2 + 20, row1 + 10});
    res.push_back({col2 + 8, row2 + 6});
    res.push_back({col1+10, row2});

    return res;
}

vector<Eigen::Vector3d> GeneratePw(const vector<Eigen::Vector2d> &blackPoint, const std::shared_ptr<Camera> &cam) {
    vector<Eigen::Vector3d> Pw;
    int id = 0;
    for(const Eigen::Vector2d &p : blackPoint) {
        // cout << "id++%4: " << id++%4 << endl; --id;
        Pw.push_back({cam->InverseProject(p.cast<int>(), kZ[(id++)%kZnum])});
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
             dx.at<float>(i, j) = 0.5 * (dist.at<float>(i, j+1) - dist.at<float>(i, j-1));
            //dx.at<float>(i, j) = (dist.at<float>(i, j+1) - dist.at<float>(i, j));
        }
    }
    for(int j = 0; j < col; ++j) {
        // 遍历一列
        for(int i = 1; i < row-1; ++i) {
             dy.at<float>(i, j) = 0.5 * (dist.at<float>(i+1, j) - dist.at<float>(i-1, j));
            //dy.at<float>(i, j) = (dist.at<float>(i+1, j) - dist.at<float>(i, j));
        }
    }
}

bool InRange(const cv::Mat &img, const Eigen::Vector2i &p) {
    // 边缘行、列忽略
    return p.x() >= 3 && p.x() < img.cols-3 && p.y() >= 3 && p.y() < img.rows-3;
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
    // cout << "w1+w2+w3+w4: " << (w1+w2+w3+w4) << endl; // equal to 1
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

Eigen::Vector3d Quat2RPY(const Eigen::Quaterniond &_q){
    const Eigen::Quaterniond q = _q.normalized();
    const double x = q.x(), y = q.y(), z = q.z(), w = q.w();
    
    // 防止除以零  
    double epsilon = 1e-6;  
      
    // roll (x-axis rotation)  
    double sinr_cosp = 2.0 * (w * x + y * z);  
    double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);  
    const double roll = std::atan2(sinr_cosp, cosr_cosp);  
  
    // pitch (y-axis rotation)  
    double sinp = 2.0 * (w * y - z * x); 
    double pitch = 0;
    if (std::abs(sinp) >= 1)  
        pitch = std::copysign(M_PI / 2, sinp); // 使用90度或-90度  
    else  
        pitch = std::asin(sinp);  
  
    // yaw (z-axis rotation)  
    double siny_cosp = 2.0 * (w * z + x * y);  
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);  
    const double yaw = std::atan2(siny_cosp, cosy_cosp);
    return {roll, pitch, yaw};
}

std::ostream& operator<<(std::ostream &cout, const Pose& T){
    cout << std::setprecision(2) << "RPY | t: " << Quat2RPY(T.q_wb_).transpose() * kRad2Deg 
         << " | " << T.t_wb_.transpose();
    return cout;
}

void DrawMatch(const Mat &img1, const Mat &img2, const vector<Eigen::Vector2d> &kp1, const vector<Eigen::Vector2d> &kp2,
                const string &name) {
    Assert(kp1.size()==kp2.size() || kp1.size() == 1, "match point size error!");
    constexpr int ratio = 1;
    Mat im(max(img1.rows, img2.rows), (img1.cols+img2.cols), CV_8UC3, cv::Scalar{0, 0, 0});
    Mat im1, im2;
    cvtColor(img1, im1, COLOR_GRAY2BGR);
    cvtColor(img2, im2, COLOR_GRAY2BGR);
    im1.copyTo(im.colRange(0, img1.cols));
    im2.copyTo(im.colRange(img1.cols, im.cols));
    cv::resize(im, im, cv::Size(ratio * im.cols, ratio * im.rows));

    const int sCol = img1.cols * ratio;
    cv::Scalar pColor = cv::Scalar(0, 255, 0);
    int radius = 1;
    cv::Scalar lColor = cv::Scalar(255, 255, 0);

    for(int i = 0; i < kp2.size(); ++i) {
        if(!InRange(img2, kp2[i].cast<int>())) {
            continue;
        }
        
        cv::Point c1(kp1[0].x()*ratio, kp1[0].y()*ratio);
        if(kp1.size() == kp2.size()) {
            c1 = cv::Point (kp1[i].x()*ratio, kp1[i].y()*ratio);
        }
        cv::circle(im, c1, radius, pColor, 2);

        cv::Point c2(sCol+kp2[i].x()*ratio, kp2[i].y()*ratio);
        cv::circle(im, c2, radius, pColor, 1);
        cv::line(im, c1, c2, lColor, 1);   
    }
    cv::namedWindow(name);
    cv::imshow(name, im);
    cv::imwrite(name+".png", im);
    cv::waitKey(0);
}

vector<Eigen::Vector2d> FindMatches(const Landmark &pc1, const Mat &edgeImg, const Pose &T21, const Camera &cam) {
    const Eigen::Vector2d &kp1 = pc1.uv_;
    
    /******** 使用极线约束寻找匹配关键点 ********
    * R21 * s1 * Pc1_norm + t21 = s2 * Pc2_norm
    * R21 * s1/s2 * Pc1_norm + 1/s2 * t21 = Pc2_norm
    * [t21]x * R21 * s1/s2 * Pc1_norm = [t21]x * Pc2_norm
    * Pc2_norm.T * [t21]x * R21 * s1/s2 * Pc1_norm = 0
    * Pc2_norm.T * [t21]x * R21 * Pc1_norm = 0 --------> 归一化平面上极线约束
    * [K.inv * px2].T * [t21]x * R21 * Pc1_norm = 0 ---> 像素平面上的极线约束 
    ********************************************/
    Eigen::Vector3d _c = skewSymmetric(T21.t_wb_) * T21.q_wb_.toRotationMatrix() * cam.InverseProject(kp1.cast<int>());
    // |k00 k01 k02|   |x|   |k00*x + k01*y + k02|
    // |k10 k11 k12| * |y| = |k10*x + k11*y + k12|
    // |k20 k21 k22|   |1|   |k20*x + k21*y + k22|
    //                                                                   |c0|
    // [k00*x + k01*y + k02, k10*x + k11*y + k12, k20*x + k21*y + k22] * |c1| = 
    //                                                                   |c2|
    // k00*c0*x + k01*c0*y + k02*c0 +
    // k10*c1*x + k11*c1*y + k12*c1 +
    // k20*c2*x + k21*c2*y + k22*c2 = (k00*c0 + k10*c1 + k20*c2)*x +
    //                                (k01*c0 + k11*c1 + k21*c2)*y +
    //                                (k02*c0 + k12*c1 + k22*c2)
    // 像素平面上极线约束的参数
    const Eigen::Matrix3d Ki = cam.K_inv_;
    const double k00 = Ki.row(0)[0], k01 = Ki.row(0)[1], k02 = Ki.row(0)[2],
                 k10 = Ki.row(1)[0], k11 = Ki.row(1)[1], k12 = Ki.row(1)[2],
                 k20 = Ki.row(2)[0], k21 = Ki.row(2)[1], k22 = Ki.row(2)[2];
    const double c0 = k00*_c[0] + k10*_c[1] + k20*_c[2],
                 c1 = k01*_c[0] + k11*_c[1] + k21*_c[2],
                 c2 = k02*_c[0] + k12*_c[1] + k22*_c[2];
    // c[0]*x + c[1]*y + c[2] = 0
    // y = -c[0]/c[1]*x - c[2]/c[1]
    
    auto Kp2Useful = [&edgeImg](const Eigen::Vector2i &p) -> bool {
        const int x=p[0], y=p[1];
        return InRange(edgeImg, p) && (edgeImg.at<uchar>(y, x) != 255);
        // return InRange(edgeImg, p) && (edgeImg.at<uchar>(y-1, x) != 255
        //     || edgeImg.at<uchar>(y, x) != 255 || edgeImg.at<uchar>(y+1, x) != 255
        //     || edgeImg.at<uchar>(y, x-1) != 255 || edgeImg.at<uchar>(y, x+1) != 255);
    };

    Eigen::Vector2i xRange, yRange;
    GetProjectRange(pc1, T21, cam, xRange, yRange);
    const int minCol = max(xRange[0], 0);
    const int maxCol = min(xRange[1], edgeImg.cols);
    const int minRow = max(yRange[0], 0);
    const int maxRow = min(yRange[1], edgeImg.rows);

    vector<Eigen::Vector2d> kp2;
    const double roundOff = 0.5; // 0.5 四舍五入参数

    auto IsZero = [](const double a) -> bool {return abs(a) < 1e-10;};
    auto PushBack = [&kp2] (const Eigen::Vector2i &px) -> void {
        // 添加一个patch内的像素点，否则由于边缘点的性质，容易出现收敛错的情况
        const int x = px.x(), y = px.y();
        kp2.push_back({x, y});
        // kp2.push_back({x-1, y});
        // kp2.push_back({x+1, y});
        // kp2.push_back({x, y-1});
        // kp2.push_back({x, y+1});
    };

    if(IsZero(c1) && IsZero(c0)) {
        return kp2; 
    } else if(IsZero(c1) && !IsZero(c0) ) {
        const int x = -c2/c0 + roundOff;
        cout << "epilor line col: " << x << endl;
        for(int y = minRow; y < maxRow; ++y) {
            const Eigen::Vector2i px{x, y};
            if(Kp2Useful(px) ) {
                PushBack(px);
            }
        }
    } else if(!IsZero(c1) && IsZero(c0) ) {
        const int y = -c2/c1 + roundOff;
        cout << "epilor line row: " << y << endl;
        for(int x = minCol; x < maxCol; ++x) {
            const Eigen::Vector2i px{x, y};
            if(Kp2Useful(px) ) {
                PushBack(px);
            }
        }
    } else {
        // y = ax + b
        const double a = -c0/c1, b = -c2/c1;
        for(int x = minCol; x < maxCol; ++x) {
            const int y = a*x + b + roundOff;
            const Eigen::Vector2i px{x, y};
            if(Kp2Useful(px) ) {
                PushBack(px);
            }
        }
    }
    return kp2;
}


Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam) {
    /******** 三角化地图点 ********
    * R21 * Pc1 + t21 = z2 * kp2_norm
    * [kp2_norm]x * R21 * Pc1 = -[kp2_norm]x * t21
    *****************************/
    // TODO: 检验为什么该种三角化方式不行！！！
    // 直观理解就是，与kp2出发射线有交点的地图点均可满足该约束，因为没有用到kp1信息，故而无法确定Pc1
    const Eigen::Vector3d kp2Norm = cam.InverseProject(kp2.cast<int>());
    const Eigen::Matrix3d skew = skewSymmetric(kp2Norm);
    const Eigen::Matrix3d A = skew * T21.q_wb_.toRotationMatrix();
    const Eigen::Vector3d b = -skew * T21.t_wb_;
    return A.colPivHouseholderQr().solve(b);
    //return A.inverse() * b;
}

Eigen::Vector3d Triangulate(const Eigen::Vector2d &kp1, const Eigen::Vector2d &kp2, const Pose &T21, const Camera &cam) {
    /******** 三角化地图点 Pc1 ********
    * R1w * Pw + t1w = z * kp1_norm
    * [kp1_norm]x * R1w * Pw = -[kp1_norm]x * t1w
    *
    * R21 * Pc1 + t21 = z * kp2_norm
    *****************************/
    const Eigen::Vector3d kp1Norm = cam.InverseProject(kp1.cast<int>());
    const Eigen::Vector3d kp2Norm = cam.InverseProject(kp2.cast<int>());
    const Eigen::Matrix3d skew1 = skewSymmetric(kp1Norm);
    const Eigen::Matrix3d skew2 = skewSymmetric(kp2Norm);
    Eigen::Matrix<double, 6, 3> A;
    A.block(0, 0, 3, 3) = skew1 * Eigen::Matrix3d::Identity();
    A.block(3, 0, 3, 3) = skew2 * T21.q_wb_.toRotationMatrix();
    Eigen::Matrix<double, 6, 1> b;
    b.middleRows(0, 3) = -skew1 * Eigen::Vector3d::Zero();
    b.middleRows(3, 3) = -skew2 * T21.t_wb_;
    
    // Eigen::JacobiSVD<Eigen::Matrix<double, 6, 3> > svd(A, Eigen::ComputeFullV);
    //cout << "conditional num: " << svd.singularValues()[2] / svd.singularValues()[0] << endl;
    return A.colPivHouseholderQr().solve(b);
}

Eigen::Vector3d Triangulate(const vector<Eigen::Vector2d> &kp2, const Pose &T21, const Camera &cam, Landmark &landmark) {
    // TODO:需要根据现实条件实现该函数，如使用光度残差作为阈值
    auto SolutionInrange = [&landmark](const double pcZ) -> bool {
                                // 进行深度滤波
                                return pcZ >= landmark.depthRange_[0] && 
                                    pcZ <= landmark.depthRange_[1];
                            };

    const Eigen::Vector2d kp1 = landmark.uv_;
    vector<double> depth;
    for(const Eigen::Vector2d &p : kp2) {
       const Eigen::Vector3d pc1 = Triangulate(kp1, p, T21, cam);
        if(SolutionInrange(pc1.z()) ) {
            depth.push_back(pc1.z());
        }
    }

    if(depth.size() > 1) {
        sort(depth.begin(), depth.end());
        landmark.depthRange_[0] = depth.front();
        landmark.depthRange_[1] = depth.back();
    } else if(depth.size() == 1 ) {
        // 极线上只有一个像素点，意味着深度收敛了?
        // 也有可能是未收敛，这时，最好的办法是要增加不确定度，
        // 或者对边缘图的边缘进行扩展，注意，扩展边缘图的像素不会是0
        landmark.depthRange_[0] = depth.front();
        landmark.depthRange_[1] = landmark.depthRange_[0] + 3.0;
    }
    landmark.UpdateUncertainty();
    
    static ofstream unf;
    static int num = 0;
    if(!num) {
        unf.open("depth_uncertainty.csv");
        unf.close();
    }
    unf.open("depth_uncertainty.csv", ios::app);
    unf << num++ << " " << landmark.uncertainty_ << endl;
    unf.close();
    return {0, 0, 0};
}

Pose ConvertRPYandPostion2Pose(const Eigen::Vector3d &rpy, const Eigen::Vector3d &t, const double deg2rad) {
    Eigen::Quaterniond q_c1c2 = Eigen::AngleAxisd(rpy[2] * deg2rad, Eigen::Vector3d::UnitY())
                              * Eigen::AngleAxisd(rpy[1] * deg2rad, Eigen::Vector3d::UnitX())
                              * Eigen::AngleAxisd(rpy[0] * deg2rad, Eigen::Vector3d::UnitZ());
    return Pose(q_c1c2, t);
}

void varifyTriangulate() {
    Pose Twc1(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    Pose Tc1c2 = ConvertRPYandPostion2Pose({0, 0, 20}, {0.5, 1., -0.5}, kDeg2Rad);
    shared_ptr<Camera> cam = make_shared<Camera>();
    
    const Eigen::Vector2d px1{15, 17};
    const double z1 = 5.0;
    const Eigen::Vector3d pc1 = cam->InverseProject(px1.cast<int>(), z1);
    cout << "pc1: " << pc1.transpose() << endl;
    cout << "px1: " << cam->Project2PixelPlane(pc1).transpose() << endl;

    const Eigen::Vector3d pc2 = Tc1c2.Inverse() * pc1;
    cout << "pc2: " << pc2.transpose() << endl;
    const Eigen::Vector2d px2 = cam->Project2PixelPlane(pc2);
    cout << "px2: " << px2.transpose() << endl; 

    // 验证像素偏差对三角化精度的影响
    vector<Eigen::Vector2d> px2_9;
    for(int x = -1; x < 2; ++x) {
        for(int y = -1; y < 2; ++y) {
            Eigen::Vector2d px{px2.x()+x, px2.y()+y};
            px2_9.push_back(px);
        }
    }

    for(int i = 0; i < 9; ++i) {
        Eigen::Vector3d pc1_est = Triangulate(px1, px2_9[i], Tc1c2.Inverse(), *cam);
        cout << "pc1_est: " << pc1_est.transpose() << endl;
    }
    // 结论：在[3x3]邻域范围内，对三角化精度的影响尚可接受

}

void GetProjectRange(const Landmark &lp, const Pose& T21, const Camera &cam, Eigen::Vector2i &xRange, 
    Eigen::Vector2i &yRange) {
    const Eigen::Vector3d pc1_1 = cam.InverseProject(lp.uv_.cast<int>(), lp.depthRange_[0]);
    const Eigen::Vector3d pc1_2 = cam.InverseProject(lp.uv_.cast<int>(), lp.depthRange_[1]);
    const Eigen::Vector3d pc2_1 = T21 * pc1_1;
    const Eigen::Vector3d pc2_2 = T21 * pc1_2;
    const Eigen::Vector2i uv1 = cam.Project2PixelPlane(pc2_1).cast<int>();
    const Eigen::Vector2i uv2 = cam.Project2PixelPlane(pc2_2).cast<int>();
    if(uv1[0] > uv2[0]) {
        xRange[0] = uv2[0];
        xRange[1] = uv1[0];
    } else {
        xRange[0] = uv1[0];
        xRange[1] = uv2[0];
    }

    if(uv1[1] > uv2[1]) {
        yRange[0] = uv2[1];
        yRange[1] = uv1[1];
    } else {
        yRange[0] = uv1[1];
        yRange[1] = uv2[1];
    }
}

void ShowPointCloud(const vector<Landmark> &ps, const Mat &img) {
    viz::Viz3d window("Point Cloud Viewer");
    cv::Affine3d viewPose;
    window.setViewerPose(viewPose);
    std::vector<Point3d> points;
    std::vector<Vec3b> colors;
    
    for(const Landmark &p : ps) {
        if(!p.Converge()) {
            continue;
        }
        const double depth = p.z_;//0.5 * (p.depthRange_[0] + p.depthRange_[1]);
        const Eigen::Vector3d &pc = p.cam_->InverseProject(p.uv_.cast<int>(), depth);
        points.push_back({pc.x(), pc.y(), pc.z()});
        cout << "[" << p.depthRange_[0] << " " << p.depthRange_[1] << "], " << depth << "  ";

        // {B G R}
        if(pc.z() > 8) {
            colors.push_back({255, 255, 0});
        } else if(pc.z() > 4) {
            colors.push_back({0, 255, 0});
        } else if(pc.z() > 2) {
            colors.push_back({0, 0, 255});
        } else {
            colors.push_back({255, 255, 255});
        }
    }
    cout << endl;
    cout << "show point size: " << points.size() << endl;

    if(points.empty() ) {
        cerr << "No Points' depth Converged!" << endl;
        return;
    } else {
        cout << points.size() << " points converged!" << endl;
    }

    // 显示一个长方体点云
    // for(float i = 0; i < 10; i+=0.1) {
    //     for(float j = 0; j < 10; j+=0.1) {
    //         points.push_back({i, j, double(rand()%100)});
    //         colors.push_back({255, 255, 255});
    //     }
    // }

    // 创建点云对象
    viz::WCloud cloud(points, colors);
 
    // 设置点云颜色和大小
    // cloud.setColor(cv::viz::Color::green());
    // cloud.setSize(5);
 
    // 显示点云
    window.showWidget("PointCloud", cloud);
 
    // 运行事件循环，使窗口响应用户输入
    window.spin();
}
