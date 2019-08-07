#include <cmath>
#include <iostream>


int main(){

  double x_p, y_p, x_c, y_c, theta;
  double x_map, y_map;

  x_p = 4;
  y_p = 5;
  x_c = 0;
  y_c = -4;
  theta = -M_PI/2;

  x_map = x_p + (cos(theta) * x_c) - (sin(theta) * y_c);
  y_map = y_p + (sin(theta) * x_c) + (cos(theta) * y_c);

  std::cout<<int(round(x_map))<<", "<<int(round(y_map))<<std::endl;

  return 0;


}
