#include <stdlib.h>

#include "Optimizer.h"

using namespace std;
using namespace cv;

constexpr double kPriorDepthWeight = 1000;

Optimizer::Optimizer(const vector<Mat> &dist, const vector<Mat> &dx, const vector<Mat> &dy, const double lambda, 
    const int maxIte, const bool useInvDepth, const bool onlyPoseUpdate) 
    : lambda_(lambda)
    , dist_(dist)
    , dx_(dx)
    , dy_(dy)
    , maxIte_(maxIte)
    , useInvDepth_(useInvDepth)
    , onlyPoseUpdate_(onlyPoseUpdate) {}

Eigen::VectorXd Optimizer::CalculateResidual(const vector<Landmark> &pc1, const vector<Pose> &T12){
    
    constexpr int resDim = 1;
    Eigen::VectorXd res(pc1.size() * T12.size() * resDim + pc1.size());
    res.setZero();

    const shared_ptr<Camera> cam = pc1[0].cam_;
    int noInrangeNum = 0;
    for(int i = 0; i < T12.size(); ++i) {
        const Pose T21 = T12[i].Inverse();
        const Mat dist = dist_[i];

        for(int j = 0; j < pc1.size(); ++j) {
            const Eigen::Vector3d pc = T21 * pc1[j].GetPc();
            const Eigen::Vector2d px = cam->Project2PixelPlane(pc);
            if(InRange(dist, px.cast<int>())) {
                // res[i*pc1.size()*resDim + j] = dist_.at<float>(px.y(), px.x());
                const double r = BilinearInterpolate(dist, px);
                // TODO: 增加异常值鲁棒核函数
                if(r < 5) {
                    res[i*pc1.size()*resDim + j] = r;
                }
            } else {
                ++noInrangeNum;
                continue;
            }
        }
    }

    // 添加残差，避免深度值z为负
    if(!onlyPoseUpdate_) {
        const int startRow = pc1.size() * T12.size() * resDim;
        for(int i = 0; i < pc1.size(); ++i) {
            if(useInvDepth_) {
                res[startRow+i] = exp(-kPriorDepthWeight / pc1[i].invZ_);
                continue;
            }
            res[startRow+i] = exp(-kPriorDepthWeight * pc1[i].z_);

        }
    }

    cout << "residual noInrangeNum: " << noInrangeNum << endl;
    return res;
}

Eigen::MatrixXd Optimizer::CalculateJacobian(const vector<Landmark>&pc1, const vector<Pose> &T12) {
    
    constexpr int resDim = 1;
    Eigen::MatrixXd J(pc1.size()*T12.size()*resDim + pc1.size(), T12.size()*T12[0].Size() + pc1.size()*pc1[0].Size());
    if(onlyPoseUpdate_) {
        J.resize(J.rows(), T12.size() * T12[0].Size());
    }
    J.setZero();

    /******** 投影过程 ********
    * K.inv * (u1, v1, 1) -> Pc1_norm * z1 -> T12.inv * Pc1 -> Pc2 / z2 -> K * Pc2_norm -> (u2, v2, 1) -> res(u2, v2)
    * res w.r.t (u2, v2) [1x2]
    * (u2, v2) w.r.t Pc2_norm [2x3]
    * Pc2_norm w.r.t Pc2 [3x3]
    * Pc2 w.r.t T12 [3x6] ------> optimization variable
    * Pc2 w.r.t Pc1 [3x3]
    * Pc1 w.r.t z1 [3x1] -------> optimization variable
    ************************/
    for(int i = 0; i < T12.size(); ++i) {
        const int poseStartCol = T12[0].Size() * i;
        int pointStartCol = T12.size() * T12[0].Size();

        const Pose T21 = T12[i].Inverse();
        const Mat dxMat = dx_[i];
        const Mat dyMat = dy_[i];

        for(int j = 0; j < pc1.size(); ++j) {
            const Landmark &p = pc1[j];
            const Eigen::Vector3d Pc1 = p.GetPc();
            const Eigen::Vector3d Pc2 = T21 * Pc1;
            const Eigen::Vector2d px2 = p.cam_->Project2PixelPlane(Pc2);
            if(!InRange(dxMat, px2.cast<int>())) {
                continue;
            }
            const double dx = BilinearInterpolate(dxMat, px2);
            const double dy = BilinearInterpolate(dyMat, px2);
            
            // res w.r.t px2 [1x2]
            const Eigen::Matrix<double, 1, 2> J_res_px2(dx, dy);

            // px2 w.r.t Pc2 [2x3]
            Eigen::Matrix<double, 2, 3> J_px2_Pc2Norm = p.cam_->K_.block(0, 0, 2, 3);
            Eigen::Matrix<double, 3, 3> J_Pc2Norm_Pc2;
            J_Pc2Norm_Pc2 << 1/Pc2[2], 0, -Pc2[0]/pow(Pc2[2], 2),
                            0, 1/Pc2[2], -Pc2[1]/pow(Pc2[2], 2),
                            0, 0, 0;
            Eigen::Matrix<double, 2, 3> J_px2_Pc2 = J_px2_Pc2Norm * J_Pc2Norm_Pc2;

            // Pc2 w.r.t T12 [3x6]
            Eigen::Matrix<double, 3, 6> J_Pc2_T12 = Eigen::Matrix<double, 3, 6>::Zero();
            const Eigen::Vector3d dt = Pc1 - T12[i].t_wb_;
            // * Pc2 w.r.t R12
            J_Pc2_T12.block(0, 0, 3, 3) = skewSymmetric(T21.q_wb_ * dt);
            // * Pc2 w.r.t t12
            J_Pc2_T12.block(0, 3, 3, 3) = -T21.q_wb_.toRotationMatrix();

            // Pc2 w.r.t Pc1 [3x3]
            Eigen::Matrix<double, 3, 3> J_Pc2_Pc1 = T21.q_wb_.toRotationMatrix();

            // Pc1 w.r.t z1 [3x1]
            const Eigen::Vector3d pc1Norm(p.GetPcNorm());      
            Eigen::Matrix<double, 3, 1> J_Pc1_z1{pc1Norm.x(), pc1Norm.y(), 1};
            // 使用逆深度表示
            if(useInvDepth_) {
                const double d = pow(p.invZ_, 2);
                J_Pc1_z1 << -pc1Norm.x()/d, -pc1Norm.y()/d, -1/d;
            }

            // 给整体雅可比矩阵赋值
            // TODO： 不需要Fixed pose，只要相对Pose准确
            J.block(i * pc1.size() + j, poseStartCol, 1, 6) = J_res_px2 * J_px2_Pc2 * J_Pc2_T12;
            if(!onlyPoseUpdate_) {
                J.block(i * pc1.size() + j, pointStartCol + j, 1, 1) = J_res_px2 * J_px2_Pc2 * J_Pc2_Pc1 * J_Pc1_z1;
            }
        }
    }

    // 添加深度值z非负雅可比
    if(!onlyPoseUpdate_) {
        const int startRow = pc1.size() * T12.size() * resDim;
        int pointStartCol = T12.size() * T12[0].Size();
        for(int i = 0; i < pc1.size(); ++i) {
            if(useInvDepth_) {
                J.block(startRow+i, pointStartCol+i, 1, 1) << kPriorDepthWeight / pow(pc1[i].invZ_, 2) * exp(-kPriorDepthWeight / pc1[i].invZ_);
            }
            J.block(startRow+i, pointStartCol+i, 1, 1) << -kPriorDepthWeight * exp(-kPriorDepthWeight * pc1[i].z_);
        }
    }

    return J;
}

Eigen::VectorXd Optimizer::SchurCompleteSolve(const Eigen::MatrixXd &H, const Eigen::VectorXd &b) {
    /***
    *     T   p...
    * T   A   B
    * p   C   D
    * ...
    ***/
    // 对B进行边缘化，左乘形成上三角矩阵
    // | I          0 |   | A  B |   | A  B |
    // | -C*A.inv   I | * | C  D | = | 0  ΔA| ==> ΔA = -C*A.inv*B + D
    // 对C进行边缘化，右乘形成下三角矩阵
    // | A  B |   | I  -A.inv*B |   | A  0 |
    // | C  D | * | 0       I   | = | C  ΔA| = H' ==> 用来求pose
    // 可以得到:
    // | I        0 |   | A  B |   | I  -A.inv*B |   | A  0 |
    // |-C*A.inv  I | * | C  D | * | 0      I    | = | 0  ΔA| = H'
    
    // 自己可以构建舒尔补，左乘形成下三角矩阵，先求解pose增量，再求解point增量
    // | I  -B*D.inv |   | A  B |   | A-B*D.inv*C  0 |
    // | 0     I     | * | C  D | = |    C         D |
    // H -= lambdaMatrix;
    const int poseSize = 6;
    const int pointSize = H.cols() - poseSize;
    const int pointDim = 1;
    const Eigen::MatrixXd &A = H.block(0, 0, poseSize, poseSize);
    const Eigen::MatrixXd &B = H.block(0, poseSize, poseSize, pointSize);
    const Eigen::MatrixXd &C = H.block(poseSize, 0, pointSize, poseSize);
    const Eigen::MatrixXd &D = H.block(poseSize, poseSize, pointSize, pointSize);
    // cout << "B - C.T:\n" << B-C.transpose() <<std::endl;
    Eigen::MatrixXd Dinv(D.rows(), D.cols());
    for(int i = 0; i < pointSize; i+=pointDim) {
        Dinv.block(i, i, pointDim, pointDim) = D.block(i, i, pointDim, pointDim).inverse();
    }
    const Eigen::MatrixXd E = -B * Dinv;
    Eigen::MatrixXd leftMatrix(H.rows(), H.cols());
    leftMatrix.block(0, 0, poseSize, poseSize).setIdentity();
    leftMatrix.block(0, poseSize, poseSize, pointSize) = E;
    leftMatrix.block(poseSize, 0, pointSize, poseSize).setZero();
    leftMatrix.block(poseSize, poseSize, pointSize, pointSize).setIdentity();

    // 求pose增量
    Eigen::MatrixXd newA = A + E * C;
    Eigen::VectorXd new_b = leftMatrix * b;
    Eigen::VectorXd deltaPose = newA.inverse() * (new_b).head(poseSize);
    cout << setprecision(3) << "deltaPose: " << deltaPose.transpose() << std::endl;



    // 求point增量
    // H * Δx = b ==> C*deltaX_pose + D*deltaX_point = b
    // D*deltaX_point = b - C*deltaX_pose
    // deltaX_point = D.inv * (b - C*deltaX_pose)
    Eigen::VectorXd deltaPoint = Dinv * (new_b.middleRows(poseSize, pointSize) - C * deltaPose);
    std::cout << setprecision(3) << "deltaPoint: "<< deltaPoint.transpose() << std::endl;

    Eigen::VectorXd deltaX(deltaPose.rows() + deltaPoint.rows());
    deltaX.middleRows(0, poseSize) = deltaPose;
    deltaX.middleRows(poseSize, pointSize) = deltaPoint;
    return deltaX;
}

bool Optimizer::Optimize(vector<Landmark> &pc1, vector<Pose> &T12) {
    double lastCost = -1;
    double firstCost = -1;
    for(int i = 0; i < maxIte_; ++i) {
        Eigen::VectorXd b = CalculateResidual(pc1, T12);
        Eigen::MatrixXd J = CalculateJacobian(pc1, T12);
        Eigen::MatrixXd H = J.transpose() * J; // J' * W * J, 设权重W均为1
        Eigen::MatrixXd I(H.rows(), H.cols());
        I.setIdentity();
        H += lambda_ * I;
        Eigen::MatrixXd g = -J.transpose() * b; // -J' * W * b
        Eigen::VectorXd delta_x;
        if(!onlyPoseUpdate_) {
            delta_x = SchurCompleteSolve(H, g);
        } else {
            delta_x = H.colPivHouseholderQr().solve(g);
        }
        cout << setprecision(3) << "delta_x: " << delta_x.transpose() << endl; 
        
        // 保留状态备份
        if(lastCost < 0) {
            lastCost = b.norm();
            firstCost = lastCost;
        }
        const vector<Landmark> pcBackup = pc1;
        const vector<Pose> poseBackup = T12;
        
        // 状态更新
        int updateId = 0;
        for(int i = 0; i < T12.size(); ++i) {
            const int startRow = i * T12[i].Size();
            T12[i].Update(delta_x.middleRows(startRow, 3), delta_x.middleRows(startRow + 3, 3));
        }
        if(!onlyPoseUpdate_) {
                updateId += T12.size() * T12[0].Size();
            for(int i = 0; i < pc1.size(); ++i) {
                pc1[i].Update(delta_x.middleRows(updateId, 1)[0], useInvDepth_);
                ++updateId;
            }
        }

        // 判断当前更新是否有效
        const double cost = CalculateResidual(pc1, T12).norm();
        cout << fixed << "iterate " << i << " times, cost: " << lastCost << " &lambda: " << lambda_ << endl << endl;

        if(lastCost <= cost) {
            lambda_ *= 1.5;
            pc1 = pcBackup;
            T12 = poseBackup;
        } else {
            lambda_ *= 0.5;
            lastCost = cost;
        }
        if(cost < 1e-9) {
            cout << "Congratulations! LM converge!!!" << endl;
            cout << "First cost | final cost: " << firstCost << " | " << lastCost << endl;
            return true;
        }
    }
    cout << "Reach max iteration time." << endl;
    cout << "First cost | final cost: " << firstCost << " | " << lastCost << endl;

    return false;
}
