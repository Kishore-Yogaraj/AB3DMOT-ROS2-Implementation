#!/usr/bin/python3

#Import of standard libararies
import time
import numpy as np

#Yellow underline appearing beacause ros2 is not installed on the machine itself and ros2 extension is not installed
# Message types that have been imported both custom and pre installed to be used in the node
from vision_msgs.msg import Detection3DArray
from tracking_msgs.msg import TrackedDetection3D, TrackedDetection3DArray

#Packages created to be used within the node each with different functionalities
from .core.ab3dmot import AB3DMOT  
from .core.utils import ros_utils
from .core.utils import geometry_utils
from .core.utils.config import cfg, cfg_from_yaml_file, log_config_to_file

import rclpy
from rclpy.node import Node

# tf2 library is used for transformations of coordinate between frames. A good way to think about this is by thinking about a camera
# that detects an object. We want the robot to move towards this object however the coordinates of this detections is in reference to 
# the camera. We need to transform the coordinates to be in reference to the base of the robot so that the robot can actually
# move towards the object. Thsese translations are set before hand based on measurements that were taken and linear algebra.

# Buffer class is responsible for stroing transformations between different coordiante frames. It acts as a storgae for all the
# transfomrations that are being publihser, allowing other parts of the systen to query the transformations as needed. Maintainsa  hsitory of trasnformations
# Between frames. Provides methods to search the transformation between any two frames at a specific time or the latest available transfromation

# The Buffer class stores different transforms. This allows for coordinates of one frame to be transformed to coordiantes of other frames
# when needed. For example if the camera detects an object at a certain point and you need the coordinates in reference to the base frame then you
# can use the buffer class to use the transforms that were stored for translating from camera frame to base frame. 
from tf2_ros.buffer import Buffer

# The TransformListener class listens to changes in transforms. In listens from the topics /tf for dynamic transformation and /tf_static for 
# static transformation and updates it. For example if a robot arm starts at one point and it moves to another point, the transform between the arm
# and the base will be different now. So the TransformListener ensures that the buffer gets this updated transformation.

# The odometry sensor continuously provides updates on the robot's movement. The node the process these updates to calculate the new position and 
# orientation of the base frame relative to the world frame. Whenever new odometry data is received, the broadcasts node updates the transfomration accordingly.

"""
The process is as follows:

1. There are `tf2_ros.TransformBroadcaster` and `tf2_ros.StaticTransformBroadcaster` classes which are typically used in different nodes.
   - `tf2_ros.TransformBroadcaster` is used for dynamic transformations that change over time.
   - `tf2_ros.StaticTransformBroadcaster` is used for static transformations that do not change over time.

2. These nodes are subscribed to specific odometry or relevant data.
   - The nodes process the incoming data to apply the necessary updates to the transformation between two different frames.

3. The new transformations are then published to the `/tf` and `/tf_static` topics respectively.
   - Dynamic transformations are published to the `/tf` topic.
   - Static transformations are published to the `/tf_static` topic.

4. A node with the `tf2_ros.TransformListener` class will listen to these topics and obtain the updated transformations.

5. The `TransformListener` class then sends the new transformations to the `tf2_ros.Buffer` class.
   - The `Buffer` class stores the transformations, maintaining a history of transformations to be queried and used when needed.

This process ensures that all nodes in the ROS2 system have access to accurate and up-to-date transformations, enabling reliable coordinate transformations and interactions within the robot's environment.
"""
from tf2_ros.transform_listener import TransformListener

import copy

class TrackerNode(Node):
    def __init__(self):

        super().__init__("tracker_node")
        self.get_logger().info("Creating object tracker node...")


        # Parameters need to be declared before they can be used in ros2
        # This basically means theat "camera_detections_topic" can be used instead of the hard coded name of the topic
        self.declare_parameter("camera_detections_topic", "/augmented_camera_detections")
        self.declare_parameter("lidar_detections_topic", "/lidar_detections")
        self.declare_parameter("tracked_detections_topic", "/tracked_detections")
        self.declare_parameter("velocity_filter_constant", 5)
        self.declare_parameter("velocity_prediction_horizon", 0.5)
        self.declare_parameter("prediction_resolution", 0.5)
        self.declare_parameter("prediction_length", 4.0)
        self.declare_parameter("max_lidar_box_dim", 2)
        self.declare_parameter("min_lidar_box_dim", 0.5)
        self.declare_parameter("lidar_distance_threshold", 1)
        self.declare_parameter("default_distance_threshold", 3)
        self.declare_parameter("max_age", 5)
        self.declare_parameter("min_hits", 3)

        # Odometery data is derived from multiple sensors such as wheel encoders, IMUs and sometimes visual sensors. These sensors provide
        # which we then process to estimate the robot's position an dorientation over time. The process involves integrating this sensors
        # data to track the robot's movement and roation whcih gives odometry estimate. Odometery data is essentially the combination
        # of data from multiple sensors. Odometery data is essentially the base coordinate system we want all detected objects and 
        # parts of the robot to be in
        self.declare_parameter("reference_frame", "odom")

        # Declated without default values
        self.declare_parameter("~traffic_sign_class_names")
        self.declare_parameter("~obstacle_class_names")
        self.declare_parameter("frequency", 10)
 
        self.declare_parameter("config_path", "/home/bolty/ament_ws/src/tracking/config/mahalanobis.yaml")


        self.camera_detections_topic = self.get_parameter("camera_detections_topic").value
        self.lidar_detections_topic = self.get_parameter("lidar_detections_topic").value
        self.tracked_detections_topic = self.get_parameter("tracked_detections_topic").value

        self.velocity_prediction_horizon = self.get_parameter("velocity_prediction_horizon").value
        self.prediction_resolution = self.get_parameter("prediction_resolution").value
        self.prediction_length = self.get_parameter("prediction_length").value
        self.max_lidar_box_dim = self.get_parameter("max_lidar_box_dim").value
        self.min_lidar_box_dim = self.get_parameter("min_lidar_box_dim").value
        self.lidar_distance_threshold = self.get_parameter("lidar_distance_threshold").value
        self.default_distance_threshold = self.get_parameter("default_distance_threshold").value
        self.max_age = self.get_parameter("max_age").value
        self.min_hits = self.get_parameter("min_hits").value
        self.reference_frame = self.get_parameter("reference_frame").value

        # Allows for parameter names in the params.yaml files to be accessed like a dictonary and used wherever neeed in the ros2 node
        cfg.TRAFFIC_SIGN_CLASSES = self.get_parameter("~traffic_sign_class_names").value
        cfg.OBSTACLE_CLASSES = self.get_parameter("~obstacle_class_names").value
        self.config_path = self.get_parameter("config_path").value
        self.frequency = self.get_parameter("frequency").value
        self.dt = 1.0 / self.frequency

        cfg_from_yaml_file(self.config_path, cfg)


        # Initialize Tracker
        # Initializes an instace of the tracker with parameters max age which is the maximum number of frames to go by before the track gets deleted
        # Min hits which is th eminim number of detectiosn required before a track is published
        # And a tracking name which is set to N/A right now
        # This means that self.mot_tracker can use any method within the AB3DMOT class. It will also use 5 for the max age and use 3 for the min
        # age because that is what was initialized in the node
        self.mot_tracker = AB3DMOT(max_age=self.max_age, 	# max age in seconds of a track before deletion
                                   min_hits=self.min_hits,	# min number of detections before publishing
                                   tracking_name="N/A") 	# default tracking age

        # Velocities for each track
        # Initializes a dictionary that will map each tracked object's unique identifier or "track)id" to its velocity components
        # As new detections are processed, the positions of the tracked objects are updated. The velocity for each object is recalculated based on the change in position over the time intervale between frames
        self.velocites = {}

        # low pass filter constant for velocity
        # constant used for low pass filtering the velocity estiamtes to smooth out noise and get a more accurate velocity measurement
        self.velocity_filter_constant = self.get_parameter("velocity_filter_constant")

        # tf2 listener
        # Creating instances of the buffer and listener to store and send information about updating transforms
        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer, self)

        # Subscribers / Publishers

        # Obstacle subscriber for detections. Will execute the obstacle call back everytime it receives a message
        self.obstacles_sub = self.create_subscription(
                 Detection3DArray, self.camera_detections_topic, self.obstacle_callback, 10)
        
        # Obstacle subscriber for lidar detections. Will execute the obstacle call back everytime it receives a message
        self.lidar_obstacles_sub = self.create_subscription(
                Detection3DArray, self.lidar_detections_topic, self.lidar_obstacle_callback, 10)
        
        # Publisher that publishes the tracked array of from the detections
        self.tracked_obstacles_publisher = self.create_publisher(
                 TrackedDetection3DArray, self.tracked_detections_topic, 10)
        
        #Will publish messages from the publisher every 0.1 seconds
        self.tracked_obstacles_timer = self.create_timer(0.1, self.publish_tracks,)

        # tracked_signs_topic = '/tracked_signs'
        # self.tracked_signs_publisher = self.create_publisher(
        # 		tracked_signs_topic, TrafficSignListMsg, queue_size=10)


    # This function is used to re initialize the tracker and creates a new instance of the AB3DMOT class with the same initialization parameters.
    # This allows for the tracker to be reset whenenver needed
        
    def reset(self):
        self.mot_tracker = AB3DMOT(max_age=self.max_age, 	# max age in seconds of a track before deletion
                                   min_hits=self.min_hits,	# min number of detections before publishing
                                   tracking_name="N/A") 	# default tracking age

    # Function is used to converst a list of 3d obstacles into 2d bounding boxes, typically for visualization or further processing in 2d space.
    # Obstacles is a list of 3d obstacels, each described as 3d cuboid (simply put a cuboid is just a rectangular prism drawn around 3d obstacles such as cars, trucks, etc. 
    # the obstacles parameter will be a list of objects/instances with each one representing a 3d obstacle. An example will look like this:
    """
    obstacles = [
    Obstacle(10, 20, 1, 4, 2, 1.5, 0.5),  # Obstacle 1
    Obstacle(15, 25, 1, 3.5, 2.5, 1.2, 1.0),  # Obstacle 2
    Obstacle(30, 10, 1, 4.2, 1.8, 1.6, 0.2)   # Obstacle 3
    ]
    """
    # to_camera_transform is the transformation matrix (mathematical operation to perform transforms from different frames) that converts coordiantes from the
    # 3d world to the camera frame. instrinsic_name is the name of the parameter that stores the camera's intrinsic matrix. An intrinsic matrix describes how your
    # camera captures images using the focal lentgh and principal point. Let's say the robot's camera sees an apple at certain pixel coordinates in the image. To 
    # interact with the apple (like picking it up), the robot needs to know the apple's position in the 3d world. Using the intrinsic matrix, the robot can transform
    # the 2d image coordinates into 3d coordinates. The function retrieves the intrinsic matrix from a parameter server using the parameter name 'intrinsic_name'. In ROS
    # a parameter server is a shared, multivaraible dictorinary that nodes use to store and retruieve paramtere at runtim. Parameters are typically set in configuration files or dynamically at run time.
    # Here is an example of what setting this parameter in a YAML configuration file might look like:
    """
    camera:
        right:
            intrinsic: "800,0,320,0,800,240,0,0,1"  # Example intrinsic matrix values
    """

    """
    So this function takes the detection from some sensors in which you'll get a list of obstacles. Each obstacle detected will then be an made object of the obstacle 
    class which simply structures the object so that its easy to manipulate values such height, width, etc. Then the function takes these values and turns them and arranges 
    them in way where a 3d bounding box could be constructed. Then it uses these 3d bounding box values to create a 3d bounding box which is just the top left coordinate and 
    height and width. Then it assigns the values within this new 2d bounding box array to a value that can be easily used called obstacle_2d. Then it appends all of the 
    obstacles to a list called obstacles_2d which hold arrays of the obstacles that represent a 2d bounding box. 
    """
    def project_objects_2d(self, obstacles, to_camera_transform, intrinsic_name="/camera/right/intrinsic"):
        """
        Inputs:
        - obstacles: List[Obstacle], describing 3D cuboids
        Outputs:
        - obstacles_2d: List[Obstacle], describing 2D bounding boxes
        """
        # Check for empty obstalces list. If the obstacles list is empty, the function returns an empty list. This is a quick exit to avoid further processing 
        # When there are no obstacles
        if not obstacles:
            return []

        # Retrieve intrinsic matrix
        try:
            # The function attempts to retrieve the intrinsic matrix and assign it to variable intrinsic matrix
            intrinsic_matrix = self.get_parameter(intrinsic_name)
        except:
            # Otherwise it logs an error and returns an empty array
            self.get_logger().error("Can't find intrinsic matrix: {}".format(intrinsic_name))
            return []
        intrinsic_matrix = np.array(intrinsic_matrix.split(','), dtype=np.float64).reshape(3, -1)

        # Project obstacles to 2d

        #Empty list to store the 2d representation of the 3d obstacles
        obstacles_2d = []

        #Begins to loop over each obstacles that was detected in the obstacle list
        for obstacle in obstacles:

            #Calls the function obstacle_to_bbox from ros_utilts to convert the obstalce into a 3d bounding box. The bounding box is a representation of the obstacle's dimensions and position.
            # The obstacls class just assigns each value in teh array to a variable so that its weasier to work with and manipulate. The obstacle_to_bbox function then takes this structured object and 
            # converts it into a format (array) that can used to draw a 3d bounding box
            bbox = ros_utils.obstacle_to_bbox(obstacle)
            try:
                # This function projects a 3d bounding box into 2d space (turn a rectangular prism to a rectangle)
                bbox_2d = geometry_utils.project_bbox_2d(bbox, to_camera_transform, intrinsic_matrix)
            except geometry_utils.ClippingError:
                continue
            # Creates a copy of the obstacle so that it doesn't make any changes to the original obstacle instance
            obstacle_2d = copy.deepcopy(obstacle)

            # Updating all dimensions so that its mapped accordingly in the new list which is obstacls_2d
            obstacle_2d.pose.pose.position.x = bbox_2d[0]
            obstacle_2d.pose.pose.position.y = bbox_2d[1]
            obstacle_2d.pose.pose.position.z = 0
            obstacle_2d.width_along_x_axis = bbox_2d[2]
            obstacle_2d.height_along_y_axis = bbox_2d[3]
            obstacle_2d.depth_along_z_axis = 0
            # Appends to the list
            obstacles_2d.append(obstacle_2d)
            #Returns list after iterating through every obstacle
        return obstacles_2d
    


    # Purpose of this function is to update the tracking state with new detections. Detections is a list of detections, each defined by a list 
    # of doubles representing the position, orientation, and dimensions. Informations is a list of lists containing class labels and confidence scores for each detections
    # frame_id is the frame in which the detections are given (odom frame, base frame, etc.). Timestam is the timea at which the detections are given. update_only means only exisitng tracks are updated
    # and new tracks are not created. distance_threshold is the threshold for associating detections with exisitng tracks based on distance
    def track(self, detections, informations, frame_id, timestamp, update_only=False, distance_threshold=None):
        """
        Update all trackers.
        Input:
            detections: [[x,y,z,rz,w,l,h]...] list of list of doubles in the Odom frame
            informations: [[String, Double]...] list of list of class labels with confidence
            frame_id: String, frame name
            timstamp: rclpy.time detections timestamp
            update_only: Bool, only update tracks don't create new ones
            distance_threshold: euclidian distance threshold for association
        """

        # If distace_threshhole is not provided then use the default value stored in the self.default_distance_threshold. In this case we want to use 3 because that is the default value
        if distance_threshold == None:
            distance_threshold = self.default_distance_threshold

#		frame_id = "base_link"
        # convert detections and informations to numpy arrays for easier maniuplation and processing. Detections are  [x, y, z, rz, w, l, h] which are the 3d detections isolated rz is yaw (rotation around z)
        # info is all other information from the detections
        detections = np.array(detections)
        infos = np.array(informations, dtype=object)

        if (frame_id != self.reference_frame and frame_id != ""):
            try:
                # Looks up the transformation from frame_id source frame to odom frame at the latest available time. Extracts the actual transforms (translation and rotations from the results TrasnformStamped message)
                # Suppose you have detections in the "caerma_frame" and you want to transform them to the "odom" frame. The lookup_transfomr method will give you the transfmration needed to convert coordinates from camera_frame to odom 
                # frame. This try will always execute becasue the frame id is most likely in the camera frame and we want all detections to be in the reference frame
                bbox_to_reference_transform = self.tf2_buffer.lookup_transform(self.reference_frame, frame_id, rclpy.time.Time(), rclpy.time.Duration(0.1)).transform
            except:
                self.get_logger().error("Failed to look up transform from {} to {}".format(self.reference_frame, frame_id))
                return
            
            # Function that takes a list of 3d boudning boxes ('detections') and tra transformation bbox_to_reference_transfomr and applies the transformations to each bounding box
            # This line actually applies the transformation to the 3d detection coordinates so you have new coordinates in the odom frame
            detections = geometry_utils.transform_boxes(detections, bbox_to_reference_transform)

        # Calls the update method of the AB3DMOT class to update the state of the tracker with new detections.
        # Numpy array of the new detections is detections. Info is a numpy array of additional information for each detections such as calss lables and confidence scores. timestamp to sec is the timestamp of the detections in seconds
        # update only is a boolean flag indicating whether to only update exisitng tracks without creatingnew ones. #Distance threshold is a value for associatng detections with existing tracks
        # The update function is what actually updates the trackers. So the value returned is an array with either new detections for tracked values or new 3d detections. 
        # Kalman filter predicts the next state of the tracker and the euclidean distance is used to measure the distance between the new detection and track. 
        # If there was 3 tracks and 3 predictions from the kalman filter and 3 new detections. The hungarian algorithm would attempt to match each detection with each prediciton that minimizes the euclidean distance (costs) 
        # between the predicted positions of tracks andnew detections. The Hungarian algorithm is crucial in the object tracking process because initially, we don't know which detection corresponds to which track
        self.mot_tracker.update(detections, infos, timestamp.to_sec(), update_only, distance_threshold)

    def obstacle_callback(self, detections_msg):
        start_time = time.time()

        # Extract the timestamp and the frame ID from the message header
        # timestamp indicates when the detction data was generated
        # frame id indicates the coordinate frame in which the detections are provided
        timestamp = detections_msg.header.stamp
        frame_id = detections_msg.header.frame_id

        detections = [] 		# [x, y, z, rot_y, l, w, h]
        informations = []		# [class label, confidence score]

        for detection in detections_msg.detections:
            detections.append(ros_utils.obstacle_to_bbox(detection.bbox))
            informations.append([detection.results[0].hypothesis.class_id, detection.results[0].hypothesis.score])

        self.track(detections, informations, frame_id, timestamp)

        # self.get_logger().info("Obstacle Update Time: {}".format(time.time() - start_time))

    def traffic_sign_callback(self, sign_list_msg):

        start_time = time.time()

        timestamp = sign_list_msg.header.stamp
        frame_id = sign_list_msg.header.frame_id

        detections = [] 		# [x, y, z, rot_y, l, w, h]
        informations = []		# [class label, confidence score]

        for sign in sign_list_msg.traffic_signs:
            detections.append(ros_utils.traffic_sign_to_bbox(sign))
            informations.append([sign.traffic_sign_type, sign.confidence])

        self.track(detections, informations, frame_id, timestamp)

        # self.get_logger().info("Traffic Sign Update Time: {}".format(time.time() - start_time))

    def lidar_obstacle_callback(self, detections_msg):

        start_time = time.time()

        timestamp = detections_msg.header.stamp
        frame_id = detections_msg.header.frame_id

        detections = [] 		# [x, y, z, rot_y, l, w, h]
        informations = []		# [class label, confidence score]

        for box in detections_msg.obstacles:
            bbox = ros_utils.obstacle_to_bbox(box)
            lwh = np.array(bbox[-3:])
            if (lwh < self.max_lidar_box_dim).all() and (lwh > self.min_lidar_box_dim).any():  # filter out boxes too big or too small
                detections.append(bbox)
                informations.append(['UNKNOWN', 0.5])
        self.track(detections, informations, frame_id, timestamp, update_only=True, distance_threshold=self.lidar_distance_threshold)

        # self.get_logger().info("Obstacle Update Time: {}".format(time.time() - start_time))

    def publish_tracks(self):
        """
        Publishes the tracks as obj_tracked. Also publishes the node status message
        dt: time since last update (in seconds)
        """
        tracked_obstacle_list = TrackedDetection3DArray()
        # tracked_obstacle_list.header.frame_id = self.reference_frame
        # tracked_obstacle_list.header.stamp = rclpy.time.Time()
        # tracked_obstacle_list.tracked_obstacles = []

        # traffic_sign_list = TrafficSignListMsg()
        # traffic_sign_list.header.frame_id = self.reference_frame
        # traffic_sign_list.header.stamp = rclpy.time.Time()
        # traffic_sign_list.traffic_signs = []

        # for kf_track in self.mot_tracker.trackers:
        #     tracking_name = kf_track.tracking_name
        #     if tracking_name in self.traffic_sign_classes:
        #         traffic_sign_message = self._create_traffic_sign_message(kf_track, tracking_name)
        #         traffic_sign_list.traffic_signs.append(traffic_sign_message)
        #     else:
        #         x = kf_track.get_state()
        #         tracked_obstacle_message = self._create_tracked_obstacle_message(kf_track, tracking_name, self.dt)
        #         tracked_obstacle_list.tracked_obstacles.append(tracked_obstacle_message)
        # self.tracked_obstacles_publisher.publish(tracked_obstacle_list)
        # self.tracked_signs_publisher.publish(traffic_sign_list)

#		self.get_logger().info("Pub Time: {}".format(time.time() - start_time))

    def _create_traffic_sign_message(self, kf_track, tracking_name):
        # [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot, rot_y_dot]
        bbox = kf_track.get_state()

        traffic_sign_message = ros_utils.bbox_to_traffic_sign(bbox, kf_track.id, tracking_name)
        traffic_sign_message.header.frame_id = self.reference_frame

        return traffic_sign_message

    def _create_tracked_obstacle_message(self, kf_track, tracking_name, dt):
        """
        Helper that creates the TrackedObstacle message from KalmanBoxTracker (python class)
        Args:
            kf_track: KalmanBoxTracker (Python class) object that represents the track
            tracking_name: String
            dt: time since last publish (in seconds)

        Returns: TrackedObstacle (anm_msgs)
        """

        tracked_obstacle_message = TrackedDetection3D()
        # [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot, rot_y_dot]
        bbox = kf_track.get_state()
        tracked_obstacle_message.obstacle = ros_utils.bbox_to_obstacle(bbox, kf_track.id, tracking_name)
        tracked_obstacle_message.obstacle.header.frame_id = self.reference_frame
        tracked_obstacle_message.header.frame_id = self.reference_frame
        #tracked_obstacle_message.score = kf_track.track_score

        if len(kf_track.history) == 0:
            self.get_logger().error("No History for id {}".format(kf_track.id))
            return tracked_obstacle_message

        if kf_track.id not in self.velocites:
            self.velocites[kf_track.id] = np.array([0.0, 0.0])

        # Estimate velocity using KF track history
        latest_timestamp = kf_track.history[-1][0]
        for timestamp, history in reversed(kf_track.history):
            dt = abs(latest_timestamp - timestamp)
            if dt >= self.velocity_prediction_horizon:
                # Ignore z velocity.
                new_velocity = np.array([(bbox[0] - history[0]) / dt, (bbox[1] - history[1]) / dt])
                # Use low pass filter
                alpha = dt / self.velocity_filter_constant
                self.velocites[kf_track.id] += alpha * (new_velocity - self.velocites[kf_track.id])
                break

        velocity = self.velocites[kf_track.id]

        # Set velocity
        tracked_obstacle_message.obstacle.twist.twist.linear.x = velocity[0]
        tracked_obstacle_message.obstacle.twist.twist.linear.y = velocity[1]

        # Create observation history messages
        for timestamp, history in kf_track.history:
            tracked_obstacle_message.observation_history.append(self.create_tracked_object_state_msg(rclpy.time.from_sec(timestamp), history))

        last_known_obs = tracked_obstacle_message.observation_history[-1]

        # Add the current observation
        tracked_obstacle_message.predicted_states.append(last_known_obs)

        current_time = last_known_obs.header.stamp
        max_pred_time = current_time + rclpy.time.Duration.from_sec(self.prediction_length)
        delta_time = 0

        # Initialize - first prediction is actual observation
        pred_time = current_time

        while (pred_time < max_pred_time):
            # self.get_logger().info('dt:{0}'.format(delta_time))
            delta_time +=  self.prediction_resolution

            pred_time += rclpy.time.Duration.from_sec(self.prediction_resolution)

            pred_obs = TrackedDetection3DArray()
            pred_obs.pose.position.x = last_known_obs.pose.position.x + (delta_time)*velocity[0]
            pred_obs.pose.position.y = last_known_obs.pose.position.y + (delta_time)*velocity[1]

            # zero velocity on Up direction
            pred_obs.pose.position.z = last_known_obs.pose.position.z

            pred_obs.pose.orientation = last_known_obs.pose.orientation
            pred_obs.velocity = last_known_obs.velocity
            pred_obs.header.stamp = pred_time
            pred_obs.header.frame_id = self.reference_frame

            tracked_obstacle_message.predicted_states.append(pred_obs)

        return tracked_obstacle_message

    def create_tracked_object_state_msg(self, timestamp, bbox):
        """Helper that creates creates the TrackedObstacleState message from a KF tracked state
        Args:
            timestamp: ROS timestamp
            bbox:  [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot, rot_y_dot]
        Returns: TrackedObstacleState (anm_msgs)
        """
        tracked_object_state = TrackedDetection3DArray()
        tracked_object_state.header.frame_id = self.reference_frame
        tracked_object_state.header.stamp = timestamp
        # [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot, rot_y_dot]
        tracked_object_state.pose.position.x = bbox[0]
        tracked_object_state.pose.position.y = bbox[1]
        tracked_object_state.pose.position.z = bbox[2]
        q = tr.quaternionerror(0, 0, bbox[3])
        tracked_object_state.pose.orientation.x = q[0]
        tracked_object_state.pose.orientation.y = q[1]
        tracked_object_state.pose.orientation.z = q[2]
        tracked_object_state.pose.orientation.w = q[3]

        tracked_object_state.velocity.linear.x = bbox[7]
        tracked_object_state.velocity.linear.y = bbox[8]
        tracked_object_state.velocity.linear.z = bbox[9]
        if (len(bbox) == 11):
            tracked_object_state.velocity.angular.z = bbox[10]

        return tracked_object_state

def main(args=None):
    rclpy.init(args=args)
    tracker_node = TrackerNode()
    rclpy.spin(tracker_node)
    tracker_node.destroy()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
