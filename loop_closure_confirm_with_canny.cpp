#include <unistd.h>


#include "Utils.h"
#include "Optimizer.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    if(argc < 3) {
        cerr << "Usage: ./main  useInverseDepth  showImage" << endl;
        exit(-1);
    }
    const bool useInvZ = bool (stoi(argv[1]));
    const bool showImg = bool(stoi(argv[2]));
    shared_ptr<Camera> cam = make_shared<Camera>();

    // 闭环帧
    vector<Eigen::Vector2d> blackPoint1 = GenerateBlackPoint();
    Mat edgeImg1 = GenerateEdgeImage(blackPoint1).clone(); // 不clone会导致core dump
    ShowImage(edgeImg1, "edgeImg1", showImg);

    // 产生Twc1下的真值3D坐标点
    shared_ptr<Pose> Twc1 = make_shared<Pose>(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    vector<Eigen::Vector3d> pc1_true = GeneratePw(blackPoint1, cam);

    vector<Landmark> pc1;
    for(int i = 0; i < blackPoint1.size(); ++i) {
        //pc1.push_back(Landmark(blackPoint1[i], Twc1, cam, 1.0));
        pc1.push_back(Landmark(blackPoint1[i], Twc1, cam, kZ[i%kZnum]));
    }
    
    auto GenerateData = [&pc1_true, &cam, &showImg](const Pose &Tc1c2, const string &id, 
                            Mat &edgeImg, Mat &dist, Mat &dx, Mat &dy) {
        const Pose Tc2c1 = Tc1c2.Inverse();
        vector<Eigen::Vector3d> pc2 = TransformPoint2Pc(Tc2c1, pc1_true);
        vector<Eigen::Vector2d> blackPoint2;
        for(Eigen::Vector3d &p : pc2) {
            blackPoint2.push_back(cam->Project2PixelPlane(p));
        }

        edgeImg = GenerateEdgeImage(blackPoint2);
        ShowImage(edgeImg, "edgeImg" + id, showImg);

        dist = GetDistanceTransform(edgeImg);
        ShowImage(dist, "dist trans" + id, showImg);

        CaculateDerivative(dist, dx, dy);
        ShowImage(dx, "dx" + id, showImg);
        ShowImage(dy, "dy" + id, showImg);
    };

    // 闭环帧的临近帧
    Eigen::Quaterniond q_c1c2 = Eigen::AngleAxisd(10 * kDeg2Rad, Eigen::Vector3d::UnitY())
                             * Eigen::AngleAxisd(0. * kRad2Deg, Eigen::Vector3d::UnitX())
                             * Eigen::AngleAxisd(0. * kRad2Deg, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d t_c1c2{0.5, 0.1, -0.5};
    Pose Tc1c2(q_c1c2, t_c1c2);
    Mat edgeImg2, dist2, dx2, dy2;
    GenerateData(Tc1c2, "2", edgeImg2, dist2, dx2, dy2);

    Eigen::Quaterniond q_c1c3 = Eigen::AngleAxisd(20 * kDeg2Rad, Eigen::Vector3d::UnitY())
                             * Eigen::AngleAxisd(0. * kRad2Deg, Eigen::Vector3d::UnitX())
                             * Eigen::AngleAxisd(0. * kRad2Deg, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d t_c1c3{1.0, 0.1, -0.8};
    Pose Tc1c3(q_c1c3, t_c1c3);
    Mat edgeImg3, dist3, dx3, dy3;
    GenerateData(Tc1c3, "3;", edgeImg3, dist3, dx3, dy3);

    Eigen::Quaterniond disturbQ = Eigen::AngleAxisd(10 * kDeg2Rad, Eigen::Vector3d::UnitY())
                             * Eigen::AngleAxisd(0. * kRad2Deg, Eigen::Vector3d::UnitX())
                             * Eigen::AngleAxisd(0. * kRad2Deg, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d disturbP{0.5, 0.1, -0.5};
    Pose disturbT(disturbQ, disturbP);

    // 调用非线性优化进行BA
    vector<Mat> vDist{dist2, dist3};
    vector<Mat> vDx{dx2, dx3};
    vector<Mat> vDy{dy2, dy3};
    vector<Pose> T12{disturbT * Tc1c2, Tc1c3 * disturbT};
    vector<Pose> T12_true{Tc1c2, Tc1c3};
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
    DrawMatch(edgeImg1, edgeImg2, blackPoint1, match[0]);
    DrawMatch(edgeImg1, edgeImg3, blackPoint1, match[1]);

    return 0;
}
