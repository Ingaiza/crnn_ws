from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    number_publisher_node = Node(

        package="crnn_cpp",
        executable="audio",
        name="audio_sub"
    )
    number_subscriber_node = Node(

        package="crnn_py",
        executable="upload",
        name="upload_node"
    )

    ld.add_action(number_publisher_node)
    ld.add_action(number_subscriber_node)

    return ld