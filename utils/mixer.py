import torch


class QuadXMixer:
    def __init__(self, roll_p, roll_i, roll_d, pitch_p, pitch_i, pitch_d, yaw_p, yaw_i, yaw_d, device = 'cuda'):
        self.roll_p = roll_p
        self.roll_i = roll_i
        self.roll_d = roll_d
        self.pitch_p = pitch_p
        self.pitch_i = pitch_i
        self.pitch_d = pitch_d
        self.yaw_p = yaw_p
        self.yaw_i = yaw_i
        self.yaw_d = yaw_d

        self.roll_vel_err_sum = 0
        self.pitch_vel_err_sum = 0
        self.yaw_vel_err_sum = 0

        self.last_roll_vel_err = 0
        self.last_pitch_vel_err = 0
        self.last_yaw_vel_err = 0

        self.mixer_matrix = torch.tensor([
            [1.0,  -1.0,  -1.0,  -1.0], 
            [1.0,  -1.0,   1.0,   1.0],
            [1.0,   1.0,   1.0,  -1.0], 
            [1.0,   1.0,  -1.0,   1.0], 
        ], device = device)

    def pid_attitude_command_for_mix(self, roll_vel_r, pitch_vel_r, yaw_vel_r, roll_vel_d, pitch_vel_d, yaw_vel_d, dt):
        '''
            input: 
                *_vel_r: real angular velocity
                *_vel_d: desired angular velocity
                dt: time interval
            output: motor_speed: attitude command for mix
        '''
        roll_vel_err = roll_vel_d - roll_vel_r
        pitch_vel_err = pitch_vel_d - pitch_vel_r
        yaw_vel_err = yaw_vel_d - yaw_vel_r

        self.roll_vel_err_sum += roll_vel_err
        self.pitch_vel_err_sum += pitch_vel_err
        self.yaw_vel_err_sum += yaw_vel_err

        self.last_roll_vel_err = roll_vel_err
        self.last_pitch_vel_err = pitch_vel_err
        self.last_yaw_vel_err = yaw_vel_err

        roll_pid = self.roll_p * roll_vel_err + self.roll_i * self.roll_vel_err_sum * dt + self.roll_d * (roll_vel_err - self.last_roll_vel_err) / dt
        pitch_pid = self.pitch_p * pitch_vel_err + self.pitch_i * self.pitch_vel_err_sum * dt + self.pitch_d * (pitch_vel_err - self.last_pitch_vel_err/ dt)
        yaw_pid = self.yaw_p * yaw_vel_err + self.yaw_i * self.yaw_vel_err_sum * dt + self.yaw_d * (yaw_vel_err - self.last_yaw_vel_err/ dt)

        return roll_pid, pitch_pid, yaw_pid

    def mix(self, throttle: float, roll_pid: float, pitch_pid: float, yaw_pid: float):
        '''
            input: 
                throttle: throttle value
                *_pid: 3 axis pid command value
            output:
                motor_speed: motor speed for 4 motors
        '''
        controls = torch.stack([throttle, roll_pid, pitch_pid, yaw_pid])
        motor_speed = torch.matmul(self.mixer_matrix, controls)
        motor_speed = motor_speed.t()

        return torch.clamp(motor_speed, 0, 1)
