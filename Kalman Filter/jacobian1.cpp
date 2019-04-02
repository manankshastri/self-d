#include <iostream>
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

MatrixXd CalculateJacobian(const VectorXd& x_state);

int main() {
	/**
	* Compute the Jacobian Matrix
	*/

	// predicted state example
	// px = 1, py = 2, vx = 0.2, vy = 0.4
	VectorXd x_predicted(4);
	x_predicted << 1, 2, 0.2, 0.4;

	MatrixXd Hj = CalculateJacobian(x_predicted);

	cout << "Hj:" << endl << Hj << endl;

	return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	// recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	// TODO: YOUR CODE HERE 
	float d1 = px*px + py*py;
	float d2 = sqrt(d1);
	float d3 = d1*d2;

	float r31 = py*(vx*py - vy*px);
	float r32 = px*(vy*px - vx*py);

	// check division by zero
	if (fabs(d1) < 0.00001){
		cout<<"Divide by zero"<<endl;
		return Hj;
	}
	// compute the Jacobian matrix
	Hj << px/d2, py/d2, 0, 0,
		-py/d1, px/d1, 0, 0,
		r31/d3, r32/d3, px/d2, py/d2;
	return Hj;
}