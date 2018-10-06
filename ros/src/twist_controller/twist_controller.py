from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object): #def __init__(self, *args, **kwargs):
	def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
			  accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
		# TODO: Implement
		self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

		# Parameters for the throttle control PID controller #
		kp = 0.3
		ki = 0.1
		kd = 0.0
		mn = 0.0 # Minimum throttle value #
		mx = 0.2 # Maximum throttle value #
		self.throttle_controller = PID(kp, ki, kd, mn, mx)

		# Low pass filter to reduce the effect of the noise of the incoming velocity #
		tau = 0.5 # 1/(2pi*tau)_ = cutoff frequency #
		ts = 0.02 # [s] Sample time #
		self.vel_lpf = LowPassFilter(tau, ts)
		
		self.vehicle_mass = vehicle_mass
		self.fuel_capacity = fuel_capacity
		self.brake_deadband = brake_deadband
		self.decel_limit = decel_limit
		self.accel_limit = accel_limit
		self.wheel_radius = wheel_radius
		
		self.last_time = rospy.get_time()

	#def control(self, *args, **kwargs):
	def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
		# TODO: Change the arg, kwarg list to suit your needs
		# Return throttle, brake, steer
		# Check if the dbw is enabled in order not to erroneously add to the I part of the PID controller #
		if not dbw_enabled:
			self.throttle_controller.reset()
			return 0.0, 0.0, 0.0
		
		current_vel = self.vel_lpf.filt(current_vel)

		steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
		
		# Calculate the difference between where the vehicle is requested to be vs where to vehicle currently is #
		vel_error = linear_vel - current_vel
		self.last_vel = current_vel
		
		# Get the sample time for each step of the PID controller #
		current_time = rospy.get_time()
		sample_time = current_time - self.last_time
		self.last_time = current_time
		
		throttle = self.throttle_controller.step(vel_error, sample_time)
		brake = 0.0
		
		# If the requested velocity is 0 and the vehicle has come to a standstill, apply the brakes #
		if linear_vel == 0.0 and current_vel < 0.1:
			throttle = 0.0
			brake = 400.0 # [N*m]  - To hold the car in place when waiting for the traffic light to turn from red to green. Acceleration ~ 1 [m/s^2] #
		
		# If the requested velocity is smaller than the current velocity, apply the brakes with a factor depending on the required deceleration and maximum deceleration #
		elif throttle < 0.1 and vel_error < 0.0:
			throttle = 0.0
			decel = max(vel_error, self.devel_limit)
			brake = abs(decel) * self.vehicle_mass * self.wheel_radius # [N*m] Torque #

		return throttle, brake, steering
