#!/usr/bin/env python3

# ROS2 Migration - Major Changes:
# 1. Import rclpy instead of rospy
# 2. Node is now a class inheriting from rclpy.node.Node
# 3. Publishers/Subscribers use create_publisher/create_subscription
# 4. Timers use create_timer with callback groups for thread safety
# 5. Time handling uses self.get_clock().now()
# 6. QoS profiles replace queue_size
# 7. Parameters use declare_parameter and get_parameter
# 8. Logging uses self.get_logger()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import sys

from std_msgs.msg import Float32, ColorRGBA, Int32
from geometry_msgs.msg import PoseStamped, Twist, Vector3, Point
from ford_msgs.msg import PedTrajVec, NNActions, PlannerMode, Clusters
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import numpy.matlib
import pickle
from matplotlib import cm
import matplotlib.pyplot as plt
import copy
import os
import time
import random
import math

# ROS2 Change: rospkg is replaced with ament_index_python
from ament_index_python.packages import get_package_share_directory

import network
import agent
import util

PED_RADIUS = 0.3

# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff

class NN_jackal(Node):
    def __init__(self, veh_name, veh_data, nn, actions):
        # ROS2 Change: Initialize node with super().__init__
        super().__init__('nn_jackal')

        self.prev_other_agents_state = []

        # vehicle info
        self.veh_name = veh_name
        self.veh_data = veh_data

        # neural network
        self.nn = nn
        self.actions = actions
        self.operation_mode = PlannerMode()
        self.operation_mode.mode = self.operation_mode.NN

        # for subscribers
        self.pose = PoseStamped()
        self.vel = Vector3()
        self.psi = 0.0
        self.ped_traj_vec = []
        self.other_agents_state = []
        self.feasible_actions = NNActions()

        # for publishers
        self.global_goal = PoseStamped()
        self.goal = PoseStamped()
        self.goal.pose.position.x = veh_data['goal'][0]
        self.goal.pose.position.y = veh_data['goal'][1]
        self.desired_position = PoseStamped()
        self.desired_action = np.zeros((2,))

        # handle obstacles close to vehicle's front
        self.stop_moving_flag = False
        self.d_min = 0.0
        self.new_subgoal_received = False
        self.new_global_goal_received = False

        # visualization
        self.path_marker = Marker()

        # Clusters
        self.prev_clusters = Clusters()
        self.current_clusters = Clusters()

        # ROS2 Change: QoS profiles replace queue_size
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ROS2 Change: Callback groups for concurrent execution
        # Use ReentrantCallbackGroup for callbacks that can run in parallel
        self.callback_group = ReentrantCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()

        # ROS2 Change: Publishers use create_publisher
        self.pub_others = self.create_publisher(Vector3, '~/other_vels', qos_profile)
        self.pub_twist = self.create_publisher(Twist, '~/nn_cmd_vel', qos_profile)
        self.pub_pose_marker = self.create_publisher(Marker, '~/pose_marker', qos_profile)
        self.pub_agent_marker = self.create_publisher(Marker, '~/agent_marker', qos_profile)
        self.pub_agent_markers = self.create_publisher(MarkerArray, '~/agent_markers', qos_profile)
        self.pub_path_marker = self.create_publisher(Marker, '~/path_marker', qos_profile)
        self.pub_goal_path_marker = self.create_publisher(Marker, '~/goal_path_marker', qos_profile)

        # subscribers and publishers counters
        self.num_poses = 0
        self.num_actions_computed = 0.0

        # ROS2 Change: Subscribers use create_subscription with callback_group
        self.sub_pose = self.create_subscription(
            PoseStamped, '~/pose', self.cbPose, qos_profile,
            callback_group=self.callback_group)
        self.sub_vel = self.create_subscription(
            Vector3, '~/velocity', self.cbVel, qos_profile,
            callback_group=self.callback_group)
        self.sub_nn_actions = self.create_subscription(
            NNActions, '~/safe_actions', self.cbNNActions, qos_profile,
            callback_group=self.callback_group)
        self.sub_mode = self.create_subscription(
            PlannerMode, '~/mode', self.cbPlannerMode, qos_profile,
            callback_group=self.callback_group)
        self.sub_global_goal = self.create_subscription(
            PoseStamped, '~/goal', self.cbGlobalGoal, qos_profile,
            callback_group=self.callback_group)

        self.use_clusters = True
        if self.use_clusters:
            self.sub_clusters = self.create_subscription(
                Clusters, '~/clusters', self.cbClusters, qos_profile,
                callback_group=self.callback_group)
        else:
            self.sub_peds = self.create_subscription(
                PedTrajVec, '~/peds', self.cbPeds, qos_profile,
                callback_group=self.callback_group)

        # ROS2 Change: Timers use create_timer with callback_group
        # 0.01 seconds = 100Hz for control
        self.control_timer = self.create_timer(
            0.01, self.cbControl, callback_group=self.timer_group)
        # 0.1 seconds = 10Hz for NN computation
        self.nn_timer = self.create_timer(
            0.1, self.cbComputeActionGA3C, callback_group=self.timer_group)

        self.get_logger().info(f'NN Jackal node initialized for {veh_name}')

    def cbGlobalGoal(self, msg):
        self.new_global_goal_received = True
        self.global_goal = msg
        self.operation_mode.mode = self.operation_mode.SPIN_IN_PLACE

        self.goal.pose.position.x = msg.pose.position.x
        self.goal.pose.position.y = msg.pose.position.y
        self.goal.header = msg.header
        self.new_subgoal_received = True

    def cbNNActions(self, msg):
        self.feasible_actions = msg

    def cbPlannerMode(self, msg):
        self.operation_mode = msg
        self.operation_mode.mode = self.operation_mode.NN

    def cbPose(self, msg):
        self.num_poses += 1
        q = msg.pose.orientation
        self.psi = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.pose = msg
        self.visualize_pose(msg.pose.position, msg.pose.orientation)

    def cbVel(self, msg):
        self.vel = msg

    def cbPeds(self, msg):
        # ROS2 Change: Time handling
        t_start = self.get_clock().now()

        self.ped_traj_vec = [ped_traj for ped_traj in msg.ped_traj_vec if len(ped_traj.traj) > 0]
        num_peds = len(self.ped_traj_vec)

        # compute relative position with respect to the Jackal
        rel_dist = np.zeros((num_peds, ))
        rel_angle = np.zeros((num_peds, ))

        for i, ped_traj in enumerate(self.ped_traj_vec):
            rel_x = ped_traj.traj[-1].pose.x - self.pose.pose.position.x
            rel_y = ped_traj.traj[-1].pose.y - self.pose.pose.position.y
            rel_dist[i] = np.linalg.norm(np.array([rel_x, rel_y]))
            rel_angle[i] = find_angle_diff(np.arctan2(rel_y, rel_x), self.psi)

        # ignore people in the back of Jackal (60 deg cone)
        valid_inds = np.where(abs(rel_angle) < 5.0 / 6.0 * np.pi)[0]

        # get the n closest agents
        self.other_agents_state = []
        if len(valid_inds) == 0:
            return
        else:
            if len(valid_inds) == 1:
                valid_inds = valid_inds[0]
                ped_traj_vec = [self.ped_traj_vec[valid_inds]]
                rel_dist = np.array([rel_dist[valid_inds]])
            elif len(valid_inds) > 1:
                ped_traj_vec = [self.ped_traj_vec[tt] for tt in valid_inds]
                rel_dist = rel_dist[valid_inds]

            # sort other agents by rel_dist
            if len(rel_dist) > self.nn.num_agents - 1:
                num_neighbors = self.nn.num_agents - 1
                neighbor_inds = np.argpartition(rel_dist, num_neighbors)[:num_neighbors]
            else:
                neighbor_inds = np.arange(len(rel_dist))

            for tt in neighbor_inds:
                ped_traj = ped_traj_vec[tt]
                x = ped_traj.traj[-1].pose.x
                y = ped_traj.traj[-1].pose.y
                v_x = ped_traj.traj[-1].velocity.x
                v_y = ped_traj.traj[-1].velocity.y
                radius = PED_RADIUS
                turning_dir = 0.0

                heading_angle = np.arctan2(v_y, v_x)
                pref_speed = np.linalg.norm(np.array([v_x, v_y]))
                goal_x = x + 5.0
                goal_y = y + 5.0

                # filter speed
                alpha = 0.2
                for prev_other_agent_state in self.prev_other_agents_state:
                    pos_diff = np.linalg.norm(prev_other_agent_state[0:2] - np.array([x, y]))
                    heading_diff_abs = abs(find_angle_diff(prev_other_agent_state[4], heading_angle))
                    if pos_diff < 0.5 and heading_diff_abs < np.pi / 4.0:
                        v_x = alpha * v_x + (1 - alpha) * prev_other_agent_state[2]
                        v_y = alpha * v_y + (1 - alpha) * prev_other_agent_state[3]
                        break

                if pref_speed < 0.2:
                    pref_speed = 0
                    v_x = 0
                    v_y = 0

                other_agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed,
                                             goal_x, goal_y, radius, turning_dir])
                self.other_agents_state.append(other_agent_state)

            self.prev_other_agents_state = copy.deepcopy(self.other_agents_state)

        t_end = self.get_clock().now()
        # ROS2 Change: Duration calculation
        duration = (t_end - t_start).nanoseconds / 1e9
        # self.get_logger().debug(f"cbPeds took: {duration} sec")

    def cbClusters(self, msg):
        other_agents = []

        xs = []
        ys = []
        radii = []
        labels = []
        num_clusters = len(msg.labels)

        for i in range(num_clusters):
            index = msg.labels[i]
            x = msg.mean_points[i].x
            y = msg.mean_points[i].y
            v_x = msg.velocities[i].x
            v_y = msg.velocities[i].y

            lower_r = np.linalg.norm(np.array([
                msg.mean_points[i].x - msg.min_points[i].x,
                msg.mean_points[i].y - msg.min_points[i].y]))
            upper_r = np.linalg.norm(np.array([
                msg.mean_points[i].x - msg.max_points[i].x,
                msg.mean_points[i].y - msg.max_points[i].y]))
            inflation_factor = 1.5
            radius = max(PED_RADIUS, inflation_factor * max(upper_r, lower_r))

            xs.append(x)
            ys.append(y)
            radii.append(radius)
            labels.append(index)

            heading_angle = np.arctan2(v_y, v_x)
            pref_speed = np.linalg.norm(np.array([v_x, v_y]))
            goal_x = x + 5.0
            goal_y = y + 5.0

            if pref_speed < 0.2:
                pref_speed = 0
                v_x = 0
                v_y = 0

            other_agents.append(agent.Agent(x, y, goal_x, goal_y, radius,
                                           pref_speed, heading_angle, index))

        self.visualize_other_agents(xs, ys, radii, labels)
        self.other_agents_state = other_agents

    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)

    def update_action(self, action):
        self.desired_action = action
        self.desired_position.pose.position.x = self.pose.pose.position.x + 1 * action[0] * np.cos(action[1])
        self.desired_position.pose.position.y = self.pose.pose.position.y + 1 * action[0] * np.sin(action[1])

        twist = Twist()
        twist.linear.x = action[0]
        yaw_error = action[1] - self.psi
        if yaw_error > np.pi:
            yaw_error -= 2*np.pi
        if yaw_error < -np.pi:
            yaw_error += 2*np.pi
        twist.angular.z = 2*yaw_error

    def find_vmax(self, d_min, heading_diff):
        d_min = max(0.0, d_min)
        x = 0.3
        margin = 0.3
        y = max(d_min, 0.0)

        if x > y:
            x = 0
        w_max = 1

        v_max = w_max * np.sqrt(x**2 + y**2)
        v_max = np.clip(v_max, 0.0, self.veh_data['pref_speed'])

        if abs(heading_diff) < np.pi / 18:
            return self.veh_data['pref_speed']
        return v_max

    def cbControl(self):
        # ROS2 Change: Time(0) becomes Time with sec=0, nanosec=0
        zero_time = rclpy.time.Time(seconds=0, nanoseconds=0)

        if self.goal.header.stamp.sec == 0 and self.goal.header.stamp.nanosec == 0:
            if self.stop_moving_flag and not self.new_global_goal_received:
                self.stop_moving()
                return
        elif self.operation_mode.mode == self.operation_mode.NN:
            desired_yaw = self.desired_action[1]
            yaw_error = desired_yaw - self.psi
            if abs(yaw_error) > np.pi:
                yaw_error -= np.sign(yaw_error) * 2 * np.pi

            gain = 2
            vw = gain * yaw_error

            use_d_min = True
            if use_d_min:
                vx = min(self.desired_action[0], self.find_vmax(self.d_min, yaw_error))
            else:
                vx = self.desired_action[0]

            twist = Twist()
            twist.angular.z = vw
            twist.linear.x = vx
            self.pub_twist.publish(twist)
            self.visualize_action(use_d_min)
            return

        elif self.operation_mode.mode == self.operation_mode.SPIN_IN_PLACE:
            self.get_logger().info('Spinning in place.')
            self.stop_moving_flag = False
            angle_to_goal = np.arctan2(
                self.global_goal.pose.position.y - self.pose.pose.position.y,
                self.global_goal.pose.position.x - self.pose.pose.position.x)
            global_yaw_error = self.psi - angle_to_goal

            if abs(global_yaw_error) > 0.5:
                vx = 0.0
                vw = 1.0
                twist = Twist()
                twist.angular.z = vw
                twist.linear.x = vx
                self.pub_twist.publish(twist)
            else:
                self.get_logger().info('Done spinning in place')
                self.operation_mode.mode = self.operation_mode.NN
                self.new_global_goal_received = False
            return
        else:
            self.stop_moving()
            return

    def cbComputeActionGA3C(self):
        if self.operation_mode.mode != self.operation_mode.NN:
            self.get_logger().info(f'Not in NN mode: {self.operation_mode.mode}')
            return

        # construct agent_state
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        v_x = self.vel.x
        v_y = self.vel.y
        radius = self.veh_data['radius']
        turning_dir = 0.0
        heading_angle = self.psi
        pref_speed = self.veh_data['pref_speed']
        goal_x = self.goal.pose.position.x
        goal_y = self.goal.pose.position.y

        # in case current speed is larger than desired speed
        v = np.linalg.norm(np.array([v_x, v_y]))
        if v > pref_speed:
            v_x = v_x * pref_speed / v
            v_y = v_y * pref_speed / v

        host_agent = agent.Agent(x, y, goal_x, goal_y, radius, pref_speed, heading_angle, 0)
        host_agent.vel_global_frame = np.array([v_x, v_y])

        other_agents_state = copy.deepcopy(self.other_agents_state)
        obs = host_agent.observe(other_agents_state)[1:]
        obs = np.expand_dims(obs, axis=0)

        predictions = self.nn.predict_p(obs)[0]
        raw_action = copy.deepcopy(self.actions[np.argmax(predictions)])
        action = np.array([pref_speed * raw_action[0], util.wrap(raw_action[1] + self.psi)])

        # if close to goal
        kp_v = 0.5
        kp_r = 1

        if host_agent.dist_to_goal < 2.0:
            pref_speed = max(min(kp_v * (host_agent.dist_to_goal - 0.1), pref_speed), 0.0)
            action[0] = min(raw_action[0], pref_speed)
            turn_amount = max(min(kp_r * (host_agent.dist_to_goal - 0.1), 1.0), 0.0) * raw_action[1]
            action[1] = util.wrap(turn_amount + self.psi)

        if host_agent.dist_to_goal < 0.3:
            self.stop_moving_flag = True
        else:
            self.stop_moving_flag = False

        self.update_action(action)



    def update_subgoal(self,subgoal):
        self.goal.pose.position.x = subgoal[0]
        self.goal.pose.position.y = subgoal[1]

    def visualize_subgoal(self,subgoal, subgoal_options=None):
        markers = MarkerArray()

        # Display GREEN DOT at NN subgoal
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'map'
        marker.ns = 'subgoal'
        marker.id = 0
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position.x = subgoal[0]
        marker.pose.position.y = subgoal[1]
        marker.scale = Vector3(x=0.4,y=0.4,z=0.2)
        marker.color = ColorRGBA(g=1.0,a=1.0)
        marker.lifetime = rospy.Duration(2.0)
        self.pub_goal_path_marker.publish(marker)

        if subgoal_options is not None:
            for i in xrange(len(subgoal_options)):
                marker = Marker()
                marker.header.stamp = rospy.Time.now()
                marker.header.frame_id = 'map'
                marker.ns = 'subgoal'
                marker.id = i+1
                marker.type = marker.CUBE
                marker.action = marker.ADD
                marker.pose.position.x = subgoal_options[i][0]
                marker.pose.position.y = subgoal_options[i][1]
                marker.scale = Vector3(x=0.2,y=0.2,z=0.2)
                marker.color = ColorRGBA(b=1.0,r=1.0,a=1.0)
                marker.lifetime = rospy.Duration(1.0)
                self.pub_goal_path_marker.publish(marker)



    def visualize_pose(self, pos, orientation):
        # Yellow Box for Vehicle
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'agent'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = pos
        marker.pose.orientation = orientation
        marker.scale = Vector3(x=0.7, y=0.42, z=1.0)
        marker.color = ColorRGBA(r=1.0, g=1.0, a=1.0)
        # ROS2 Change: Duration is now from rclpy.duration
        marker.lifetime = rclpy.duration.Duration(seconds=1).to_msg()
        self.pub_pose_marker.publish(marker)

        # Red track for trajectory over time
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'agent'
        marker.id = self.num_poses
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = pos
        marker.pose.orientation = orientation
        marker.scale = Vector3(x=0.2, y=0.2, z=0.2)
        marker.color = ColorRGBA(r=1.0, a=1.0)
        marker.lifetime = rclpy.duration.Duration(seconds=10).to_msg()
        self.pub_pose_marker.publish(marker)

    def visualize_other_agents(self, xs, ys, radii, labels):
        markers = MarkerArray()
        for i in range(len(xs)):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.ns = 'other_agent'
            marker.id = labels[i]
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = xs[i]
            marker.pose.position.y = ys[i]
            marker.scale = Vector3(x=2*radii[i], y=2*radii[i], z=1.0)
            marker.color = ColorRGBA(r=1.0, g=0.4, a=1.0)
            marker.lifetime = rclpy.duration.Duration(seconds=0, nanoseconds=100000000).to_msg()
            markers.markers.append(marker)

        self.pub_agent_markers.publish(markers)

    def visualize_action(self, use_d_min):
        # Display BLUE ARROW from current position to NN desired position
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'path_arrow'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points.append(self.pose.pose.position)
        marker.points.append(self.desired_position.pose.position)
        marker.scale = Vector3(x=0.1, y=0.2, z=0.2)
        marker.color = ColorRGBA(b=1.0, a=1.0)
        marker.lifetime = rclpy.duration.Duration(seconds=0, nanoseconds=500000000).to_msg()
        self.pub_goal_path_marker.publish(marker)

        # Display BLUE DOT at NN desired position
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'path_trail'
        marker.id = self.num_poses
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = copy.deepcopy(self.desired_position.pose.position)
        marker.scale = Vector3(x=0.2, y=0.2, z=0.2)
        marker.color = ColorRGBA(b=1.0, a=0.1)
        marker.lifetime = rclpy.duration.Duration(seconds=0, nanoseconds=500000000).to_msg()

        if self.desired_action[0] == 0.0:
            marker.pose.position.x += 2.0 * np.cos(self.desired_action[1])
            marker.pose.position.y += 2.0 * np.sin(self.desired_action[1])

        self.pub_goal_path_marker.publish(marker)

    def on_shutdown(self):
        self.get_logger().info(f"Shutting down {self.veh_name}")
        self.stop_moving()
        self.get_logger().info(f"Stopped {self.veh_name}'s velocity")


def main(args=None):
    print('Neural Network Jackal ROS2 Node')

    # ROS2 Change: Initialize rclpy instead of rospy
    rclpy.init(args=args)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    plt.rcParams.update({'font.size': 18})

    # ROS2 Change: Use ament_index instead of rospkg
    package_share_dir = get_package_share_directory('cadrl_ros')

    a = network.Actions()
    actions = a.actions
    num_actions = a.num_actions
    nn = network.NetworkVP_rnn(network.Config.DEVICE, 'network', num_actions)
    nn.simple_load(os.path.join(package_share_dir, 'checkpoints', 'network_01900000'))

    veh_name = 'JA01'

    # ROS2 Change: Create temporary node to get parameter
    temp_node = rclpy.create_node('temp_param_node')
    temp_node.declare_parameter('jackal_speed', 0.5)
    pref_speed = temp_node.get_parameter('jackal_speed').value
    temp_node.destroy_node()

    veh_data = {
        'goal': np.zeros((2,)),
        'radius': 0.5,
        'pref_speed': pref_speed,
        'kw': 10.0,
        'kp': 1.0,
        'name': 'JA01'
    }

    print(f"********\n*******\n*********\nJackal speed: {pref_speed}\n**********\n******")

    nn_jackal = NN_jackal(veh_name, veh_data, nn, actions)

    # ROS2 Change: Use MultiThreadedExecutor for concurrent callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(nn_jackal)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        nn_jackal.on_shutdown()
        nn_jackal.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()