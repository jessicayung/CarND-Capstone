
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

    def control(self, current_velocity,linear_velocity,angular_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
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

        #TODO - Apply the brakes if necessary
        brake = 0.0
        if cmd > 0:
            throttle = cmd
            # brake = 0.0
        else:
            throttle = 0.0
            # brake = math.fabs(throttle)
            # if brake > 1.0:
            #     brake = 1.0
            # #Add deadband for the brakes
            # elif brake < 0.0:
            #     brake = 0.0

        return throttle, brake, steer

    def reset(self):
    	self.pid_lin_vel.reset()
