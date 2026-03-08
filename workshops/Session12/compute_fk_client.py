import rclpy
from rclpy.node import Node

from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState


class ComputeFKClient(Node):
    def __init__(self):
        super().__init__("compute_fk_client")

        self.client = self.create_client(GetPositionFK, "/compute_fk")

        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for service...")

    def call_fk(self, joint_names: list[str], joint_positions: list[float], link_name: str, frame_id: str = "world"):
        request = GetPositionFK.Request()

        request.header.frame_id = frame_id

        request.fk_link_names = [link_name]

        robot_state = RobotState()
        joint_state = JointState()
        joint_state.position = joint_positions
        joint_state.name = joint_names
        robot_state.joint_state = joint_state

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()

            self.get_logger().info(f"{response}")
        else:
            self.get_logger().error("Service call failed.")


def main():
    rclpy.init()

    node = ComputeFKClient()

    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    joint_positions = [0.0, -1.0, 1.0, 0.0, 0.5, 0.0]

    link_name = "tool0"

    node.call_fk(
        joint_names=joint_names,
        joint_positions=joint_positions,
        link_name=link_name,
        frame_id="world",
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
