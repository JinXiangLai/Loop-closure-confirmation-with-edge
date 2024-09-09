#include <unistd.h>


#include "Utils.h"
#include "Optimizer.h"

using namespace std;
using namespace cv;

// 利用极线约束去寻找anchor帧与普通帧的匹配以确定匹配特征点
// 得到一个较为准确的深度初值，再与闭环帧执行BA优化

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
    vector<Eigen::Vector2d> blackPoint1 = GenerateBlackPoint();
    Mat edgeImg1 = GenerateEdgeImage(blackPoint1).clone(); // 不clone会导致core dump
    ShowImage(edgeImg1, "edgeImg1", showImg);

    // 产生Twc1下的真值3D坐标点
    shared_ptr<Pose> Twc1 = make_shared<Pose>(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    vector<Eigen::Vector3d> pc1_true = GeneratePw(blackPoint1, cam);

    
    auto GenerateData = [&pc1_true, &cam, &showImg, &edgeImg1, &blackPoint1](const Pose &Tc1c2, const string &id, 
                            Mat &edgeImg, Mat &dist, Mat &dx, Mat &dy) {
        const Pose Tc2c1 = Tc1c2.Inverse();
        vector<Eigen::Vector3d> pc2 = TransformPoint2Pc(Tc2c1, pc1_true);
        vector<Eigen::Vector2d> blackPoint2;
        for(Eigen::Vector3d &p : pc2) {
            blackPoint2.push_back(cam->Project2PixelPlane(p));
        }

        edgeImg = GenerateEdgeImage(blackPoint2);
        DrawMatch(edgeImg1, edgeImg, blackPoint1, blackPoint2, "edgeImage 1&"+id+" matches");

        ShowImage(edgeImg, "edgeImg" + id, showImg);

        dist = GetDistanceTransform(edgeImg);
        ShowImage(dist, "dist trans" + id, showImg);

        CaculateDerivative(dist, dx, dy);
        ShowImage(dx, "dx" + id, showImg);
        ShowImage(dy, "dy" + id, showImg);
    };

    // 闭环帧的临近帧
    // Eigen::Vector3d t_c1c2{0.5, 0.1, -0.5}; // 视差如果没有足够大的话，三角化出来的地图点会不准
    Eigen::Vector3d t_c1c2{1.0, 1.0, -0.5}; 
    const Pose Tc1c2 = ConvertRPYandPostion2Pose({0, 5, 10}, t_c1c2, kDeg2Rad);
    Mat edgeImg2, _dist2, _dx2, _dy2;
    GenerateData(Tc1c2, "2", edgeImg2, _dist2, _dx2, _dy2);

    // 这里需要根据edgeImg1、edgeImg2、Tc1c2进行地图点的三角化
    vector<Landmark> pc1;
    const Pose Tc2c1 = Tc1c2.Inverse();
    for(int i = 0; i < blackPoint1.size(); ++i) {
        // const std::vector<Eigen::Vector2d> kp2 = FindMatches(blackPoint1[i], edgeImg2, Tc2c1, *cam);
        const std::vector<Eigen::Vector2d> kp2 = FindMatchesWithEpipolarConstraintOnImagePlane(blackPoint1[i], edgeImg2, Tc2c1, *cam);
        const Eigen::Vector3d pc = Triangulate(blackPoint1[i], kp2, Tc1c2.Inverse(), *cam);
        pc1.push_back(Landmark(blackPoint1[i], Twc1, cam, pc[2]));
        DrawMatch(edgeImg1, edgeImg2, {blackPoint1[i]}, kp2, "Epipolar constraint matches");

        //pc1.push_back(Landmark(blackPoint1[i], Twc1, cam, kZ[i%kZnum]));
    }

    Eigen::Vector3d t_c1c3{1.0, 0.1, -0.8};
    const Pose Tc1c3 = ConvertRPYandPostion2Pose({0, 0, 20}, t_c1c3, kDeg2Rad);
    Mat edgeImg3, dist3, dx3, dy3;
    GenerateData(Tc1c3, "3", edgeImg3, dist3, dx3, dy3);

    Eigen::Vector3d disturbP{0.2, 0.1, -0.50};
    Pose disturbT = ConvertRPYandPostion2Pose({0.0, 0.0, 3.0}, disturbP, kDeg2Rad);

    // 调用非线性优化进行BA
    vector<Mat> vDist{dist3};
    vector<Mat> vDx{dx3};
    vector<Mat> vDy{dy3};
    vector<Pose> T12{Tc1c3 * disturbT};
    vector<Pose> T12_true{Tc1c3};
    Optimizer optimizer(vDist, vDx, vDy, 1, 100, useInvZ);

    optimizer.Optimize(pc1, T12);

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
        cout << setprecision(2) << p.z_ << " ";
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
    DrawMatch(edgeImg1, edgeImg3, blackPoint1, match[0], "edge matches after optimization");
    return 0;
}
