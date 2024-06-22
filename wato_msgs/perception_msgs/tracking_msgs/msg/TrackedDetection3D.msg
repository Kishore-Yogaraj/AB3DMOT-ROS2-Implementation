std_msgs/Header header

# ID of the detection being tracked
# Integer value 
int64 tracked_id

# The bounding box and current position of the detection in question
vision_msgs/Detection3D detection

# Velocity of the detection
# Speed in the linear direction for x y and z
# Speed in angular direction for x y and z
geometry_msgs/Twist velocity

# History of Observations (duration is configurable) The last element in observation_history corresponds to the current state
#Array of values from Detection3D message type
vision_msgs/Detection3D[] observation_history

# Linear future predictions based on velocity and position
#Array of values from Detection3D message type
vision_msgs/Detection3D[] predicted_states
