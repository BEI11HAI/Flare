import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, inv_quat, transform_quat_by_quat, quat_to_R
import utils.mixer as mixer


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class HoverEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(0.0, 7.0, 3.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1)),
                                              background_color=(0.9, 0.9, 0.9)),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=False,
                enable_joint_limit=False,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="urdf/plane/checker_blue.png"
                ),
            ),
        )

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.07,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(0.3, 0.3, 0.9),
                    ),
                ),
            )
        else:
            self.target = None

        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(1920, 1080),
                pos=(0.0, 10.0, 3.0),
                lookat=(0.0, 0.0, 1.5),
                fov=30,
                GUI=True,
            )

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.payload_init_pos = self.base_init_pos - torch.tensor([0, 0, self.env_cfg["cable_length"]], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.default_dofs_pos = torch.zeros((self.num_envs, self.env_cfg['num_of_dofs']), device=self.device, dtype=gs.tc_float)

        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(file="assets/suspended_system/real_sus_exp.urdf"),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.2, 0.2),
                ),
            ),
        )

        # build scene
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.next_commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # quadrotor related states
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        # payload related states
        self.payload_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_payload_pos = torch.zeros_like(self.payload_pos)
        self.payload_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.cable_length = self.env_cfg["cable_length"]
        self.cable_angle = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)  # [pitch, roll] angles
        self.last_cable_angle = torch.zeros_like(self.cable_angle)
        self.cable_angle_vel = torch.zeros_like(self.cable_angle)
        self.cable_body_angle = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.max_safe_cable_angle = math.radians(self.env_cfg.get("max_safe_cable_angle_deg", 80))
        self.safety_violation_count = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.max_safety_violations = self.env_cfg.get("max_safety_violations", 10)

        self.extras = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx] = self.next_commands[envs_idx]

        self.next_commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.next_commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.next_commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        return at_target

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # transform action range to respective ranges
        self.throttle_cmd = torch.sqrt(exec_actions[:, 0] * 0.5 + 0.5)
        self.roll_rate_cmd = exec_actions[:, 1] * self.env_cfg["max_roll_rate"]
        self.pitch_rate_cmd = exec_actions[:, 2] * self.env_cfg["max_pitch_rate"]
        self.yaw_rate_cmd = exec_actions[:, 3] * self.env_cfg["max_yaw_rate"]

        # mixer computes motor RPM
        ang_vel = self.drone.get_ang()
        quad_mixer = mixer.QuadXMixer(
            roll_p=0.1, roll_i=0.05, roll_d=0.0001,
            pitch_p=0.1, pitch_i=0.05, pitch_d=0.0001,
            yaw_p=0.1, yaw_i=0.05, yaw_d=0.0001
        )
        roll_pid, pitch_pid, yaw_pid = quad_mixer.pid_attitude_command_for_mix(
            roll_vel_r=ang_vel[:, 0], pitch_vel_r=ang_vel[:, 1], yaw_vel_r=ang_vel[:, 2],
            roll_vel_d=self.roll_rate_cmd, pitch_vel_d=self.pitch_rate_cmd, yaw_vel_d=self.yaw_rate_cmd,
            dt=0.01
        )
        motor_outputs = quad_mixer.mix(
            throttle=self.throttle_cmd,
            roll_pid=roll_pid,
            pitch_pid=pitch_pid,
            yaw_pid=yaw_pid
        )
        self.drone.set_propellels_rpm(motor_outputs * torch.sqrt(torch.tensor(self.env_cfg["TWR_max"])) * 14468.429183500699 * 4.71)

        # update target pos
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True, envs_idx=list(range(self.num_envs)))
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.next_rel_pos = self.next_commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        self.base_lin_vel[:] = self.drone.get_vel()
        self.base_ang_vel[:] = self.drone.get_ang()
        self.last_payload_pos[:] = self.payload_pos
        self.payload_pos[:] = self.drone.get_link(name="end_sphere").get_pos()
        self.payload_lin_vel[:] = self.drone.get_link(name="end_sphere").get_vel()
        self.rel_pos_payload = self.commands - self.payload_pos
        self.last_rel_pos_payload = self.commands - self.last_payload_pos

        # update safety-related states
        self._update_safety_states()

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            | (self.safety_violation_count > self.max_safety_violations)
            | (self.cable_body_angle > self.env_cfg.get("termination_cable_angle_threshold", math.radians(90)))
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        rotmat = quat_to_R(self.base_quat)
        rotmat_flat = rotmat.reshape(self.num_envs, 9)
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos[:, 0:2] * self.obs_scales["rel_pos_xy"], -1, 1),
                torch.clip(self.rel_pos[:, 2:3] * self.obs_scales["rel_pos_z"], -1, 1),
                torch.clip(self.next_rel_pos[:, 0:2] * self.obs_scales["rel_pos_xy"], -1, 1),
                torch.clip(self.next_rel_pos[:, 2:3] * self.obs_scales["rel_pos_z"], -1, 1),
                torch.clip(self.base_lin_vel[:, 0:2] * self.obs_scales["lin_vel_xy"], -1, 1),
                torch.clip(self.base_lin_vel[:, 2:3] * self.obs_scales["lin_vel_z"], -1, 1),
                rotmat_flat,
                self.last_actions,
                torch.clip(self.cable_angle * self.obs_scales.get("cable_angle", 1.0), -1, 1),
                torch.clip(self.cable_angle_vel * self.obs_scales.get("cable_angle_vel", 0.1), -1, 1),
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset cable and payload
        self.payload_pos[envs_idx] = self.payload_init_pos
        self.last_payload_pos[envs_idx] = self.payload_init_pos
        self.rel_pos_payload = self.commands - self.payload_pos
        self.last_rel_pos_payload = self.commands - self.last_payload_pos
        self.payload_lin_vel[envs_idx] = 0

        cable_joint_dofs = []
        for i in range(1, int((len(self.default_dofs_pos[0])+1)/2)):
            sphere_joint_name = f"segment_{i}_sphere_to_cylinder"
            cylinder_joint_name = f"segment_{i-1}_cylinder_to_segment_{i}_sphere" if i > 1 else "center_of_mass_link_to_segment_1_sphere"

            cylinder_joint = self.drone.get_joint(name=cylinder_joint_name)
            cable_joint_dofs.append(cylinder_joint.dofs_idx_local)
            sphere_joint = self.drone.get_joint(name=sphere_joint_name)
            cable_joint_dofs.append(sphere_joint.dofs_idx_local)

        end_joint = self.drone.get_joint(name="joint_end_sphere")
        cable_joint_dofs.append(end_joint.dofs_idx_local)

        cable_joint_dofs_flat = []
        for dof_idx in cable_joint_dofs:
            if isinstance(dof_idx, (list, tuple)):
                cable_joint_dofs_flat.extend(dof_idx)
            else:
                cable_joint_dofs_flat.append(dof_idx)

        cable_dofs_positions = torch.zeros((len(envs_idx), len(cable_joint_dofs_flat)), device=self.device, dtype=gs.tc_float)

        self.drone.set_dofs_position(
            position=cable_dofs_positions,
            dofs_idx_local=cable_joint_dofs_flat,
            zero_velocity=True,
            envs_idx=envs_idx)

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # reset safety states
        self.cable_angle[envs_idx] = 0.0
        self.last_cable_angle[envs_idx] = 0.0
        self.cable_angle_vel[envs_idx] = 0.0
        self.cable_body_angle[envs_idx] = 0.0
        self.safety_violation_count[envs_idx] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions ----------------
    def _reward_target(self):
        target_rew = torch.square(self.last_rel_pos) - torch.square(self.rel_pos)
        target_rew = torch.sum(target_rew, dim=1)
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    def _reward_cable_angle_safety(self):
        """Cable swing angle safety reward: only penalize at extreme angles."""
        cable_body_angle = self.cable_body_angle
        angle_safety_reward = torch.zeros_like(cable_body_angle)
        unsafe_mask = cable_body_angle > self.max_safe_cable_angle
        angle_safety_reward[unsafe_mask] = -self.reward_cfg.get("unsafe_angle_penalty", 0.001)
        return angle_safety_reward

    def _update_safety_states(self):
        """Update safety-related state variables."""
        self.last_payload_pos[:] = self.payload_pos[:]
        self.last_cable_angle[:] = self.cable_angle[:]

        cable_vector = self.payload_pos - self.base_pos

        self.cable_angle[:, 0] = torch.atan2(cable_vector[:, 0], -cable_vector[:, 2])
        self.cable_angle[:, 1] = torch.atan2(cable_vector[:, 1], -cable_vector[:, 2])

        self.cable_angle_vel[:] = (self.cable_angle - self.last_cable_angle) / self.dt

        cable_vector_normalized = cable_vector / torch.norm(cable_vector, dim=1, keepdim=True)
        rotmat = quat_to_R(self.base_quat)
        body_z_axis = -rotmat[:, :, 2]

        dot_product = torch.sum(cable_vector_normalized * body_z_axis, dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        self.cable_body_angle = torch.acos(dot_product)

        extreme_angle_violation = self.cable_body_angle > self.max_safe_cable_angle
        self.safety_violation_count += extreme_angle_violation.int()
