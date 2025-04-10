#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/string.hpp> 

class MoveitManipulatorController : public rclcpp::Node
{
public:
    MoveitManipulatorController()
        : Node("hello_moveit", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)),
          tfBuffer(this->get_clock()),
          tfListener(tfBuffer)
    {
        subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target_pose", 10, std::bind(&MoveitManipulatorController::pose_callback, this, std::placeholders::_1));

        string_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/select_signal", 10, std::bind(&MoveitManipulatorController::string_callback, this, std::placeholders::_1));
    }

    void initialize_move_group()
    {
        move_group_interface = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "ur_manipulator");
        auto current_pose = move_group_interface->getCurrentPose();
    }

private:
    std::vector<geometry_msgs::msg::PoseStamped> accumulated_poses_;
    void string_callback(const std_msgs::msg::String::SharedPtr msg) {
        if (msg->data == "finish") {
            // Create waypoints and execute trajectory
            create_and_execute_waypoints();
        } 
        else if (msg->data == "reset") {
            // Clear accumulated_poses_
            accumulated_poses_.clear();
            RCLCPP_INFO(this->get_logger(), "Accumulated poses reset.");
        } 
    }

    void create_and_execute_waypoints() {
        // Ensure there are poses to work with
        // if (accumulated_poses_.empty()) {
        //     RCLCPP_WARN(this->get_logger(), "No poses accumulated to create waypoints.");
        //     return;
        // }

        // // Assuming you have an instance of MoveGroupInterface named move_group_interface
        auto current_pose = move_group_interface->getCurrentPose();

        // To print the pose
        RCLCPP_INFO(this->get_logger(), "Current Pose: Position (x, y, z) = (%.2f, %.2f, %.2f); Orientation (w, x, y, z) = (%.2f, %.2f, %.2f, %.2f)", 
                    current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z,
                    current_pose.pose.orientation.w, current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z);
        std::string reference_frame = move_group_interface->getPoseReferenceFrame();

        std::vector<geometry_msgs::msg::Pose> waypoints;

        geometry_msgs::msg::Pose home;
        home.orientation.x = current_pose.pose.orientation.x;
        home.orientation.y = current_pose.pose.orientation.y;
        home.orientation.z = current_pose.pose.orientation.z;
        home.orientation.w = current_pose.pose.orientation.w;
        home.position.x = current_pose.pose.position.x;
        home.position.y = current_pose.pose.position.y;
        home.position.z = current_pose.pose.position.z;

        geometry_msgs::msg::Pose place_pose1_base;
        place_pose1_base.orientation.x = home.orientation.x;
        place_pose1_base.orientation.y = home.orientation.y;
        place_pose1_base.orientation.z = home.orientation.z;
        place_pose1_base.orientation.w = home.orientation.w;
        place_pose1_base.position.x = home.position.x;
        place_pose1_base.position.y = home.position.y;
        place_pose1_base.position.z = home.position.z-0.3;
        
        
        waypoints.push_back(home);
        waypoints.push_back(place_pose1_base);
        waypoints.push_back(home);

        // for (size_t i = 0; i < accumulated_poses_.size(); ++i) {
        //     // Convert the quaternion from the message to Eigen
        //     Eigen::Quaterniond quaternion_target(
        //         accumulated_poses_[i].pose.orientation.w,
        //         accumulated_poses_[i].pose.orientation.x,
        //         accumulated_poses_[i].pose.orientation.y,
        //         accumulated_poses_[i].pose.orientation.z
        //     );
        //     Eigen::Matrix3d rotation_matrix_target = quaternion_target.toRotationMatrix();

        //     // Extract the translation from the message
        //     Eigen::Vector3d translation_target(
        //         accumulated_poses_[i].pose.position.x,
        //         accumulated_poses_[i].pose.position.y,
        //         accumulated_poses_[i].pose.position.z
        //     );

        //     // Create the transformation matrix (4x4) from target pose to tool0
        //     Eigen::Matrix4d T_targetpose_to_tool0 = Eigen::Matrix4d::Identity();
        //     T_targetpose_to_tool0.block<3, 3>(0, 0) = rotation_matrix_target;
        //     T_targetpose_to_tool0.block<3, 1>(0, 3) = translation_target;

        //     // Get the current pose of the tool in relation to base frame
        //     auto current_pose = move_group_interface->getCurrentPose();

        //     Eigen::Quaterniond quaternion_tool0(
        //         current_pose.pose.orientation.w,
        //         current_pose.pose.orientation.x,
        //         current_pose.pose.orientation.y,
        //         current_pose.pose.orientation.z
        //     );
        //     Eigen::Matrix3d rotation_matrix_tool0 = quaternion_tool0.toRotationMatrix();
        //     Eigen::Vector3d translation_tool0(
        //         current_pose.pose.position.x,
        //         current_pose.pose.position.y,
        //         current_pose.pose.position.z
        //     );

        //     // Create the transformation matrix (4x4) from tool0 to base
        //     Eigen::Matrix4d T_tool0_to_base = Eigen::Matrix4d::Identity();
        //     T_tool0_to_base.block<3, 3>(0, 0) = rotation_matrix_tool0;
        //     T_tool0_to_base.block<3, 1>(0, 3) = translation_tool0;

        //     // Calculate the transformation matrix from target to base
        //     Eigen::Matrix4d T_target_to_base = T_tool0_to_base * T_targetpose_to_tool0;

        //     // Decompose the final transformation matrix into translation and quaternion
        //     Eigen::Matrix3d rotation_matrix_final = T_target_to_base.block<3, 3>(0, 0);
        //     Eigen::Quaterniond quaternion_final(rotation_matrix_final);
        //     Eigen::Vector3d translation_final = T_target_to_base.block<3, 1>(0, 3);

        //     // Set the target pose for move_group
        //     geometry_msgs::msg::Pose target_pose;
        //     target_pose.orientation.x = quaternion_final.x();
        //     target_pose.orientation.y = quaternion_final.y();
        //     target_pose.orientation.z = quaternion_final.z();
        //     target_pose.orientation.w = quaternion_final.w();
        //     target_pose.position.x = translation_final.x();
        //     target_pose.position.y = translation_final.y();
        //     target_pose.position.z = translation_final.z();

        //     geometry_msgs::msg::Pose place_pose1 = place_pose1_base;
        //     place_pose1.position.z += 0.12 * i;

        //     geometry_msgs::msg::Pose place_pose2 = place_pose1;
        //     place_pose2.position.x += 0.3;

        //     geometry_msgs::msg::Pose place_pose3 = place_pose2;
        //     place_pose3.position.z -= 0.1;

        //     waypoints.push_back(home);
        //     waypoints.push_back(target_pose);
        //     waypoints.push_back(place_pose1);
        //     waypoints.push_back(place_pose2);
        //     waypoints.push_back(place_pose3);
        //     waypoints.push_back(place_pose2);
        //     waypoints.push_back(place_pose1);
        //     waypoints.push_back(home);
        // }

        moveit_msgs::msg::RobotTrajectory trajectory;
        const double jump_threshold = 0.0;
        const double eef_step = 0.01;
        double fraction = move_group_interface->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);


        move_group_interface->execute(trajectory); 

        accumulated_poses_.clear();
    }
    void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        // Accumulate the incoming pose data
        accumulated_poses_.push_back(*msg);

    }

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr string_subscription_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_interface;
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto controller = std::make_shared<MoveitManipulatorController>();
    controller->initialize_move_group();  // Initialize move group here
    rclcpp::spin(controller);
    rclcpp::shutdown();
    return 0;
}