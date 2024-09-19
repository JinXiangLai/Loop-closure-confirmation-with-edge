#include <unistd.h>
#include <chrono>

#include "Utils.h"
#include "Optimizer.h"

using namespace std;
using namespace cv;

// 利用极线约束去寻找anchor帧与普通帧的匹配以确定匹配特征点
// 得到一个较为准确的深度初值，再与闭环帧执行BA优化
// 本程序验证了，当视差没有足够大的时候，即便只有一个像素偏差，也会对三角化结果产生极大的影响，
// 因此，基于边缘特征的SLAM无法直接用于单目相机的闭环检验

int main(int argc, char** argv){
    if(argc < 3) {
        cerr << "Usage: ./main  useInverseDepth  showImage" << endl;
        exit(-1);
    }
    const bool useInvZ = bool (stoi(argv[1]));
    const bool showImg = bool(stoi(argv[2]));
    shared_ptr<Camera> cam = make_shared<Camera>();
    
    //varifyTriangulate(); return 0;

    // 闭环帧
    vector<Eigen::Vector2d> vertexPoint1 = GenerateBlackPoint();
    Mat edgeImg1 = GenerateEdgeImage(vertexPoint1).clone(); // 不clone会导致core dump
    ShowImage(edgeImg1, "edgeImg1", showImg);

    // 产生Twc1下的真值3D坐标点
    shared_ptr<Pose> Twc1 = make_shared<Pose>(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    vector<Eigen::Vector3d> pc1_true = GeneratePw(vertexPoint1, cam);
    vector<Landmark> pc1;
    for(int j = 0; j < edgeImg1.cols; ++j) {
        for(int i = 0; i < edgeImg1.rows; ++i) {
            if(edgeImg1.at<uchar>(i, j) == 0) {
                pc1.push_back(Landmark({j, i}, Twc1, cam, 1.0));
            }
        }
    }
    
    auto GenerateData = [&pc1_true, &cam, &showImg, &edgeImg1, &vertexPoint1](const Pose &Tc1c2, const string &id, 
                            Mat &edgeImg, Mat &dist, Mat &dx, Mat &dy) {
        const Pose Tc2c1 = Tc1c2.Inverse();
        vector<Eigen::Vector3d> pc2 = TransformPoint2Pc(Tc2c1, pc1_true);
        vector<Eigen::Vector2d> vertexPoint2;
        for(Eigen::Vector3d &p : pc2) {
            vertexPoint2.push_back(cam->Project2PixelPlane(p));
        }

        edgeImg = GenerateEdgeImage(vertexPoint2);
        DrawMatch(edgeImg1, edgeImg, vertexPoint1, vertexPoint2, "edgeImage 1&"+id+" matches");

        ShowImage(edgeImg, "edgeImg" + id, showImg);

        dist = GetDistanceTransform(edgeImg);
        ShowImage(dist, "dist trans" + id, showImg);

        CaculateDerivative(dist, dx, dy);
        ShowImage(dx, "dx" + id, showImg);
        ShowImage(dy, "dy" + id, showImg);
    };

    auto UpdateDepth = [&edgeImg1] (vector<Landmark> &lps, const shared_ptr<Camera> cam, const Mat &edgeImg2, const Pose &Tc2c1) -> void {
        cout << "\n****** depth range ******\n";
        for(int i = 0; i < lps.size(); ++i) {
            Landmark &pc1 = lps[i];
            const std::vector<Eigen::Vector2d> kp2 = FindMatches(pc1, edgeImg2, Tc2c1, *cam);
            const Eigen::Vector3d pc = Triangulate(kp2, Tc2c1, *cam, pc1);
            
            cout << "[" << pc1.depthRange_[0] << ", " << pc1.depthRange_[1] << "]: "
                << pc1.depthRange_[1] - pc1.depthRange_[0] << endl;
            // DrawMatch(edgeImg1, edgeImg2, {pc1.uv_}, kp2, "Epipolar constraint matches");
        }
    };

    // 闭环帧的临近帧
    // 视差如果没有足够大的话，三角化出来的地图点会不准
    // Eigen::Vector3d t_c1c2{0.5, 0.1, -0.5}; 
    //Eigen::Vector3d t_c1c2{0.0, 0.0, -0.5}; 
    Eigen::Vector3d t_c1c2{0.0, 0.0, -3.5}; 
    const Pose Tc1c2 = ConvertRPYandPostion2Pose({0, 2, 3}, t_c1c2, kDeg2Rad);
    Mat edgeImg2, _dist2, _dx2, _dy2;
    GenerateData(Tc1c2, "2", edgeImg2, _dist2, _dx2, _dy2);
    // 需要多帧进行深度滤波
    UpdateDepth(pc1, cam, edgeImg2, Tc1c2.Inverse());
    ShowPointCloud(pc1, edgeImg2);

    Eigen::Vector3d t_c1c3{0.0, 0.0, 2.8};
    const Pose Tc1c3 = ConvertRPYandPostion2Pose({0, 2, 3}, t_c1c3, kDeg2Rad);
    Mat edgeImg3, dist3, dx3, dy3;
    GenerateData(Tc1c3, "3", edgeImg3, dist3, dx3, dy3);
    UpdateDepth(pc1, cam, edgeImg3, Tc1c3.Inverse());
    ShowPointCloud(pc1, edgeImg3);

    Eigen::Vector3d t_c1c4{0.0, 0.0, -1.2};
    const Pose Tc1c4 = ConvertRPYandPostion2Pose({0, 4, 5}, t_c1c4, kDeg2Rad);
    Mat edgeImg4, dist4, dx4, dy4;
    GenerateData(Tc1c4, "4", edgeImg4, dist4, dx4, dy4);
    UpdateDepth(pc1, cam, edgeImg4, Tc1c4.Inverse());
    ShowPointCloud(pc1, edgeImg4);

    Eigen::Vector3d t_c1c5{0.0, 0.0, 2.0};
    const Pose Tc1c5 = ConvertRPYandPostion2Pose({0, 3, 3}, t_c1c5, kDeg2Rad);
    Mat edgeImg5, dist5, dx5, dy5;
    GenerateData(Tc1c5, "5", edgeImg5, dist5, dx5, dy5);
    UpdateDepth(pc1, cam, edgeImg5, Tc1c5.Inverse());
    ShowPointCloud(pc1, edgeImg5);


    Eigen::Vector3d disturbP{0.0, 0.0, -0.0};
    Pose disturbT = ConvertRPYandPostion2Pose({0.0, 0.0, 0.0}, disturbP, kDeg2Rad);

    // 调用非线性优化进行BA
    vector<Mat> vDist{dist3, dist4};
    vector<Mat> vDx{dx3, dx4};
    vector<Mat> vDy{dy3, dy4};
    vector<Pose> T12{Tc1c3, Tc1c4};
    vector<Pose> T12_true{Tc1c3};
    Optimizer optimizer(vDist, vDx, vDy, 1, 100, useInvZ);

    optimizer.Optimize(pc1, T12);

    ShowPointCloud(pc1, edgeImg1);

    for(int i = 0; i < T12.size(); ++i) {
        const Pose &Tc1c2 = T12[i];
        vector<Eigen::Vector2d> projBlackPoint2;
        const Pose Tc2c1 = Tc1c2.Inverse();
        for(const Landmark &p : pc1) {
            Eigen::Vector3d pc2 = Tc2c1 * p.GetPw();
            projBlackPoint2.push_back(cam->Project2PixelPlane(pc2));
        }
        cout << endl;
        Mat projImg2 = GenerateEdgeImage(projBlackPoint2).clone();
        ShowImage(projImg2, "reprojImg" + to_string(i+2), showImg);
    }

    cout << "pc1 num: " << pc1.size() << endl;
    for(const Landmark &p : pc1) {
        //cout << setprecision(2) << p.z_ << " "; 
    }
    cout << endl;

    vector<vector<Eigen::Vector2d> > match(T12.size());
    vector<Pose> T21 = T12;
    for(Pose &T : T21) {
        T = T.Inverse();
    }
    for(int i = 0; i < T12.size(); ++i) {
        cout << "Pose diff " << i << ": " << T12_true[i].Inverse() * T12[i] << endl;
        for(const Landmark &p : pc1) {
            match[i].push_back(p.cam_->Project2PixelPlane(T21[i] * p.GetPc()));
        }
    }
    DrawMatch(edgeImg1, edgeImg3, vertexPoint1, match[0], "edge matches after optimization");
    return 0;
}
