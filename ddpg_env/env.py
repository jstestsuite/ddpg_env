"""TurtleBot3 goal-reaching environment over ROS 2 / Gazebo Harmonic.

Resets teleport the robot and the goal marker with Gazebo's set_pose service (bridged to
ROS by stage.launch.py), which takes ~0.1 s instead of the ~2 s delete/respawn cycle.
Teleporting doesn't reset the diff-drive odometry, so the robot's world pose is tracked
relative to the odometry reading taken at each reset.

One step = publish a velocity command, then wait for the next lidar scan (5 Hz sim time).
"""

import math
import os
import random
import threading

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import LaserScan

MAX_GOAL_DISTANCE = 5.0     # for normalizing; the stage arenas are about 4.5 m across
MIN_GOAL_SPAWN_DISTANCE = 0.35
SCAN_PERIOD = 0.2           # burger lidar is 5 Hz
# Same goal sets as turtlebot3_dqn's dqn_gazebo
STAGE4_GOALS = [
    (1.0, 0.0), (2.0, -1.5), (0.0, -2.0), (2.0, 1.5), (0.5, 2.0), (-1.5, 2.1),
    (-2.0, 0.5), (-2.0, -0.5), (-1.5, -2.0), (-0.5, -1.0), (2.0, -0.5), (-1.0, -1.0),
]
# turtlebot3_dqn_world/obstacles: 0.15 m radius cylinders
STAGE2_PILLARS = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
STAGE2_PILLAR_CLEARANCE = 0.5   # pillar radius + robot + collision margin + goal radius


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def clean_ranges(ranges, range_min, range_max):
    r = np.asarray(ranges, dtype=np.float32)
    r = np.where(np.isnan(r) | np.isposinf(r), range_max, r)
    # -inf or below range_min means something is closer than the lidar can see
    r = np.where(np.isneginf(r) | (r < range_min), 0.0, r)
    return np.clip(r, 0.0, range_max)


def sample_goal(stage):
    if stage == 4:
        return random.choice(STAGE4_GOALS)
    while True:
        x, y = random.randrange(-21, 21) / 10, random.randrange(-21, 21) / 10
        # stage 2's static pillars: a goal on or beside one can't be reached
        if stage != 2 or all(math.hypot(x - px, y - py) >= STAGE2_PILLAR_CLEARANCE
                             for px, py in STAGE2_PILLARS):
            return x, y


class TurtleBotEnv(Node):
    def __init__(self, param_defaults):
        """`param_defaults` is declared as this node's ROS parameters; the resolved
        values (defaults overridden by --params-file / -p) end up in self.cfg."""
        super().__init__('ddpg_env')
        cfg = {k: self.declare_parameter(k, v).value for k, v in param_defaults.items()}
        self.cfg = cfg
        self.mode = cfg['mode']
        if self.mode not in ('steer', 'steer_speed'):
            raise ValueError(f"mode must be 'steer' or 'steer_speed', got {self.mode!r}")
        self.action_dim = 1 if self.mode == 'steer' else 2
        self.state_dim = cfg['n_scan_bins'] + 4

        self._cond = threading.Condition()
        self._scan = None
        self._scan_max = 3.5
        self._scan_stamp = 0.0
        self._scan_seq = 0
        self._odom = None

        self.goal_x = 0.0
        self.goal_y = 0.0
        self._odom_origin = (0.0, 0.0, 0.0)
        self.prev_distance = None
        self.past_action = np.zeros(2, dtype=np.float32)
        self.last_step_stamp = None
        self.late_steps = 0

        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.create_subscription(LaserScan, 'scan', self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self._on_odom, 10)
        self.set_pose_client = self.create_client(
            SetEntityPose, f"/world/{cfg['world_name']}/set_pose")

    # ---------- ROS plumbing ----------

    def _on_scan(self, msg):
        ranges = clean_ranges(msg.ranges, msg.range_min, msg.range_max)
        with self._cond:
            self._scan = ranges
            self._scan_max = msg.range_max
            self._scan_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._scan_seq += 1
            self._cond.notify_all()

    def _on_odom(self, msg):
        p = msg.pose.pose
        with self._cond:
            self._odom = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
            self._cond.notify_all()

    def _check_ok(self):
        if not self.context.ok():
            raise KeyboardInterrupt

    def _wait_for(self, predicate, what, timeout=5.0):
        with self._cond:
            while not self._cond.wait_for(predicate, timeout=timeout):
                self._check_ok()
                self.get_logger().warn(f'still waiting for {what} (is the sim running?)')

    def _set_poses(self, poses):
        """Teleport models: poses = {name: (x, y)}. Calls run concurrently."""
        while not self.set_pose_client.wait_for_service(timeout_sec=2.0):
            self._check_ok()
            self.get_logger().warn(f"service '{self.set_pose_client.srv_name}' not available; "
                                   'start the sim with `ros2 launch ddpg_env stage.launch.py`')
        pending = []
        for name, (x, y) in poses.items():
            req = SetEntityPose.Request()
            req.entity.name = name
            req.entity.type = Entity.MODEL
            req.pose.position.x = float(x)
            req.pose.position.y = float(y)
            req.pose.orientation.w = 1.0
            done = threading.Event()
            future = self.set_pose_client.call_async(req)
            future.add_done_callback(lambda _, d=done: d.set())
            pending.append((name, future, done))
        for name, future, done in pending:
            while not done.wait(timeout=1.0):
                self._check_ok()
            if not future.result().success:
                self.get_logger().error(f'set_pose failed for {name!r}')

    def publish_cmd(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish_cmd(0.0, 0.0)

    # ---------- observation / reward ----------

    def robot_pose(self):
        """World pose. The robot starts each episode at the origin facing +x, so the
        world pose is the odometry pose expressed relative to the reading at reset."""
        x, y, yaw = self._odom
        ox, oy, oyaw = self._odom_origin
        dx, dy = x - ox, y - oy
        c, s = math.cos(-oyaw), math.sin(-oyaw)
        return c * dx - s * dy, s * dx + c * dy, yaw - oyaw

    def _goal_geometry(self):
        x, y, yaw = self.robot_pose()
        distance = math.hypot(self.goal_x - x, self.goal_y - y)
        heading = math.atan2(self.goal_y - y, self.goal_x - x) - yaw
        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        return distance, heading

    def _observe(self):
        with self._cond:
            scan = self._scan
            scan_max = self._scan_max
        distance, heading = self._goal_geometry()
        # min-pool into bins; bin 0 is straight ahead
        bins = np.array([b.min() for b in np.array_split(scan, self.cfg['n_scan_bins'])])
        state = np.concatenate([
            2.0 * bins / scan_max - 1.0,
            self.past_action,
            [heading / math.pi, 2.0 * min(distance, MAX_GOAL_DISTANCE) / MAX_GOAL_DISTANCE - 1.0],
        ]).astype(np.float32)
        return state, distance, float(scan.min())

    def _action_to_cmd(self, action):
        c = self.cfg
        if self.mode == 'steer':
            linear = c['fixed_linear']
            angular = float(action[0]) * c['max_angular']
        else:
            linear = (float(action[0]) + 1.0) / 2.0 * c['max_linear']
            angular = float(action[1]) * c['max_angular']
        return linear, angular

    # ---------- episode control ----------

    def _new_goal_near(self, x, y):
        while True:
            gx, gy = sample_goal(self.cfg['stage'])
            if math.hypot(gx - x, gy - y) >= MIN_GOAL_SPAWN_DISTANCE:
                return gx, gy

    def _wait_scans(self, n):
        seq = self._scan_seq
        self._wait_for(lambda: self._scan_seq >= seq + n, 'scan')

    def check_single_sim(self):
        """Two sims on the same ROS domain interleave their scans and odometry and both obey
        cmd_vel, which silently wrecks training. Refuse to continue if that's happening."""
        n = self.count_publishers('scan')
        if n > 1:
            raise RuntimeError(
                f'{n} publishers on /scan: more than one simulation is running on '
                f'ROS_DOMAIN_ID={os.environ.get("ROS_DOMAIN_ID", "0")}. '
                "Stop the others (`pgrep -af 'gz sim'`) or give each its own "
                'ROS_DOMAIN_ID and GZ_PARTITION.')

    def reset(self):
        """Teleport the robot to the origin with a new goal; return the first state."""
        self.stop()
        self._wait_for(lambda: self._scan is not None and self._odom is not None, 'scan and odom')
        self.check_single_sim()
        self.goal_x, self.goal_y = self._new_goal_near(0.0, 0.0)
        for attempt in range(3):
            self._set_poses({self.cfg['robot_name']: (0.0, 0.0),
                             'goal_box': (self.goal_x, self.goal_y)})
            # two scans: the first may have been rendered before the teleport
            self._wait_scans(2)
            with self._cond:
                clear = float(self._scan.min()) >= self.cfg['collision_distance']
            if clear:
                break
            self.get_logger().warn(f'reset attempt {attempt + 1}: robot still in collision '
                                   'after teleport, retrying')
        else:
            raise RuntimeError(
                'reset failed: the robot is still in collision after 3 teleports to the origin. '
                'It may have tipped over, or another simulation is answering set_pose. '
                'Restart the sim.')
        with self._cond:
            self._odom_origin = self._odom
        self.past_action[:] = 0.0
        self.last_step_stamp = None
        self.late_steps = 0
        state, distance, _ = self._observe()
        self.prev_distance = distance
        return state

    def next_goal(self):
        """After reaching a goal: move it somewhere new and keep going from the current pose."""
        x, y, _ = self.robot_pose()
        self.goal_x, self.goal_y = self._new_goal_near(x, y)
        self._set_poses({'goal_box': (self.goal_x, self.goal_y)})
        self.last_step_stamp = None     # the pause isn't a late step
        state, distance, _ = self._observe()
        # Reset progress tracking to the *new* goal, otherwise the next step's
        # progress reward is a huge negative (old bug #1)
        self.prev_distance = distance
        return state

    def step(self, action, during_wait=None):
        """Apply `action` until the next scan. `during_wait` runs while the sim advances
        (used for the gradient update so it costs no wall time).

        Returns (next_state, reward, terminal, event) where event is None, 'goal' or 'collision'.
        `terminal` means no bootstrapping from next_state.
        """
        c = self.cfg
        linear, angular = self._action_to_cmd(action)
        seq = self._scan_seq
        self.publish_cmd(linear, angular)
        if during_wait is not None:
            during_wait()
        self._wait_for(lambda: self._scan_seq > seq, 'scan')

        stamp = self._scan_stamp
        if self.last_step_stamp is not None and stamp - self.last_step_stamp > 1.5 * SCAN_PERIOD:
            self.late_steps += 1
        self.last_step_stamp = stamp

        self.past_action[:] = (2.0 * linear / c['max_linear'] - 1.0, angular / c['max_angular'])
        state, distance, min_range = self._observe()

        reward = c['progress_scale'] * (self.prev_distance - distance) - c['time_penalty']
        self.prev_distance = distance
        event = None
        if min_range < c['collision_distance']:
            event = 'collision'
            reward = c['collision_reward']
            self.stop()
        elif distance < c['goal_distance']:
            event = 'goal'
            reward = c['goal_reward']
        return state, reward, event is not None, event
