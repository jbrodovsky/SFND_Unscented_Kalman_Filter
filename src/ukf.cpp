#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;
  use_laser_ = true;      // if this is false, laser measurements will be ignored (except during init)
  use_radar_ = true;      // if this is false, radar measurements will be ignored (except during init)
  x_ = VectorXd(5);       // initial state vector
  P_ = MatrixXd(5, 5);    // initial covariance matrix
  std_a_ = 0.6;           // Process noise standard deviation longitudinal acceleration in m/s^2
  std_yawdd_ = 0.5;       // Process noise standard deviation yaw acceleration in rad/s^2
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */
  std_laspx_ = 0.15;  // Laser measurement noise standard deviation position1 in m
  std_laspy_ = 0.15;  // Laser measurement noise standard deviation position2 in m
  std_radr_ = 0.3;    // Radar measurement noise standard deviation radius in m
  std_radphi_ = 0.03; // Radar measurement noise standard deviation angle in rad
  std_radrd_ = 0.3;   // Radar measurement noise standard deviation radius change in m/s
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  n_x_ = 5;           // State dimension
  n_aug_ = 7;         // Augment state dimension
  lambda_ = 3 - n_x_; // Sigma point spreading parameter
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);  // Sigma points list (augmented)
  weights_ = VectorXd(2*n_aug_ + 1);
  weights_.fill(0.5/(lambda_ + n_aug_));
  weights_(0) = lambda_/(lambda_ + n_aug_);
  R_Laser = MatrixXd(2,2);
  R_Laser << std_laspx_*std_laspx_, 0,
              0 , std_laspy_*std_laspy_;
  R_Radar = MatrixXd(3,3);
  R_Radar << std_radr_*std_radr_,     0, 0,
             0, std_radphi_*std_radphi_, 0,
             0, 0,     std_radr_*std_radr_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_){
    Initialize(meas_package);
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
  else {
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;
    Prediction(dt);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER && 
        use_laser_) { 
          UpdateLidar(meas_package); 
    }
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && 
        use_radar_) { 
          UpdateRadar(meas_package); 
    }
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  MatrixXd Xsig_aug = GenerateSigmaPoints();
  PropagateSigmaPoints(Xsig_aug, delta_t);
  
  x_ = Xsig_pred_ * weights_;
  P_.fill(0.0);
  for (int i=0; i<2*n_aug_+1; ++i){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while(x_diff(3) > M_PI) { x_diff(3) -= 2.0*M_PI; }
    while(x_diff(3) <-M_PI) { x_diff(3) += 2.0*M_PI; }
    P_ += weights_(i) * x_diff*x_diff.transpose();
  }
  
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Create measurement sigma points
  int n_z = 2;
  MatrixXd Zsig(n_z, 2*n_aug_+1);
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i=0; i<2*n_aug_ + 1; ++i){
    Zsig(0, i) = Xsig_pred_(0,i);
    Zsig(1, i) = Xsig_pred_(1,i);
    z_pred += weights_(i) * Zsig.col(i);
  }
  // Calculate inovation covariance matrix
  MatrixXd S(n_z, n_z);
  S.fill(0.0);
  for (int i=0; i<2*n_aug_+1; ++i){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i)*z_diff*z_diff.transpose();
  }
  S += R_Laser;
  // Calculate cross corelation matrix
  MatrixXd T = MatrixXd::Zero(n_x_, n_z);
  for (int i=0; i<2*n_aug_+1; ++i){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    T += weights_(i)*x_diff*z_diff.transpose();
  }
  // Calculate Kalman gain matrix and update the mean and covariance
  MatrixXd K = T*S.inverse();
  VectorXd z_ = meas_package.raw_measurements_;
  x_ += K*(z_ - z_pred);
  P_ -= K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;
  MatrixXd Zsig(n_z, 2*n_aug_+1);
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i=0; i<2*n_aug_ + 1; ++i){
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double vel = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    Zsig(0, i) = sqrt(px*px + py*py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px*vel*cos(yaw) + py*vel*sin(yaw)) / sqrt(px*px + py*py);
   
    z_pred += weights_(i) * Zsig.col(i);
  }
  // Calculate inovation covariance matrix
  MatrixXd S(n_z, n_z);
  S.fill(0.0);
  for (int i=0; i<2*n_aug_+1; ++i){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) >  M_PI) { z_diff(1) -= 2.*M_PI; }
    while (z_diff(1) <- M_PI) { z_diff(1) += 2.*M_PI; }
    S += weights_(i)*z_diff*z_diff.transpose();
  }
  S += R_Radar;
  // Calculate cross corelation matrix
  MatrixXd T = MatrixXd::Zero(n_x_, n_z);
  for (int i=0; i<2*n_aug_+1; ++i){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) >  M_PI) { z_diff(1) -= 2.*M_PI; }
    while (z_diff(1) <- M_PI) { z_diff(1) += 2.*M_PI; }

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(1) >  M_PI) { x_diff(1) -= 2.*M_PI; }
    while (x_diff(1) <- M_PI) { x_diff(1) += 2.*M_PI; }

    T += weights_(i)*x_diff*z_diff.transpose();
  }
  // Calculate Kalman gain matrix and update the mean and covariance
  MatrixXd K = T*S.inverse();
  VectorXd z_ = meas_package.raw_measurements_;
  x_ = x_ + K*(z_ - z_pred); 
  P_ = P_ - K*S*K.transpose();
}

void UKF::Initialize(MeasurementPackage meas_package) {
  switch (meas_package.sensor_type_)
  {
  case MeasurementPackage::LASER:
  {
    double px = meas_package.raw_measurements_(0);
    double py = meas_package.raw_measurements_(1);
    x_ << px, py, 0, 0, 0;
    P_ = MatrixXd::Identity(n_x_, n_x_);
  }
  break;
  case MeasurementPackage::RADAR:
  {
    double rho = meas_package.raw_measurements_(0);
    double phi = meas_package.raw_measurements_(1);
    double rhodot = meas_package.raw_measurements_(2);
    double vx = rhodot*cos(phi);
    double vy = rhodot*sin(phi);
    x_ << rho*cos(phi),
          rho*sin(phi),
          sqrt(vx*vx + vy*vy),
          0,
          0;
    P_ = MatrixXd::Identity(n_x_, n_x_);
  }
  break;
  default:
    break;
  }
}

MatrixXd UKF::GenerateSigmaPoints(){
  VectorXd x_aug = VectorXd::Zero(n_aug_);          // Augmented state vector
  x_aug.head(n_x_) = x_;
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);  // Augmented covariance
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;
  MatrixXd L = P_aug.llt().matrixL();               // Square root matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
  
  Xsig_aug.col(0) = x_aug;
  for (int i=0; i<n_aug_; ++i){
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
  return Xsig_aug;
}

void UKF::PropagateSigmaPoints(Eigen::MatrixXd XSig_aug, double dt){
  for (int i=0; i<2*n_aug_+1; ++i){
    double px = XSig_aug(0, i);
    double py = XSig_aug(1, i);
    double v = XSig_aug(2, i);
    double yaw = XSig_aug(3, i);
    double yawd = XSig_aug(4, i);
    double nu_a = XSig_aug(5, i);
    double nu_yawdd = XSig_aug(6, i);

    double px_p, py_p;
    if (fabs(yawd) > 0.001){
      px_p = px + v/yawd * (sin(yaw + yawd*dt) - sin(yaw));
      py_p = py + v/yawd * (cos(yaw) - cos(yaw + yawd*dt));
    }
    else {
      px_p = px + v*dt*cos(yaw);
      py_p = py + v*dt*sin(yaw);
    }
    Xsig_pred_(0, i) = px_p + 0.5*nu_a*dt*dt*cos(yaw);
    Xsig_pred_(1, i) = py_p + 0.5*nu_a*dt*dt*sin(yaw);
    Xsig_pred_(2, i) = v + nu_a*dt;
    Xsig_pred_(3, i) = yaw + 0.5*nu_yawdd*dt*dt;
    Xsig_pred_(4, i) = yawd + nu_yawdd*dt;
  }
}

