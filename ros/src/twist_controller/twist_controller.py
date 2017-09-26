
import math
import time
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, yaw_ctrl, pid_lin_vel, filt_ang_vel):
        self.yaw_ctrl = yaw_ctrl
        self.pid_lin_vel = pid_lin_vel
        self.filt_ang_vel = filt_ang_vel
        self.t0 = time.time()
        self.dt = 0.0

    def control(self, current_velocity, linear_velocity, angular_velocity):
        # Return throttle, brake, steer

        #Apply the filter to the angular velocity
        angular_velocity = self.filt_ang_vel.filt(angular_velocity)

        #Compute the steering angle
        steer = self.yaw_ctrl.get_steering(
                linear_velocity,
                angular_velocity,
                current_velocity)
        steer = math.degrees(steer)

        #Compute the throttle command
        cmd = 0
        if self.dt:
            vel_error = (linear_velocity - current_velocity)
            cmd = self.pid_lin_vel.step(vel_error, self.dt)

        self.dt = time.time() - self.t0
        self.t0 += self.dt

        #Apply the brakes if necessary
        brake = 0.0
        if cmd > 0:
            throttle = cmd
        else:
            throttle = 0.0
            # limit abs(cmd) to 0 and the negative lower limit of the pid controller
            # limiting brake to 1.0 would result in controller windup, since
            # pid_lin_vel.min can be > 1.0.
            brake = max(0, min(math.fabs(cmd), -self.pid_lin_vel.min))

        return throttle, brake, steer

    def reset(self):
    	self.pid_lin_vel.reset()
