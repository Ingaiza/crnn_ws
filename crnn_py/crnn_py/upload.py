#!/usr/bin/env python3
import os
import shutil
import rclpy
from rclpy.node import Node

# Google Drive Imports
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

class UploadNode(Node): 
    def __init__(self):
        super().__init__("upload_node")
        
        # 1. Declare Parameters
        self.declare_parameter("source_folder", "/home/ingaiza/GoogleDrive/Ambience")
        self.declare_parameter("archive_folder", "/home/ingaiza/GoogleDrive/Ambience/Processed")
        self.declare_parameter("drive_folder_id", "1vMijtAK7qTrn0Famp50RY0uBc6x9r_Nj") 

        # 2. Get Parameters
        self.source_path = self.get_parameter("source_folder").value
        self.archive_path = self.get_parameter("archive_folder").value
        self.drive_parent_id = self.get_parameter("drive_folder_id").value

        # 3. Ensure directories exist
        os.makedirs(self.source_path, exist_ok=True)
        os.makedirs(self.archive_path, exist_ok=True)

        # 4. Authentication Block
        try:
            self.get_logger().info("Attempting Google Drive Authentication...")
            self.gauth = GoogleAuth()
            self.gauth.settings['get_refresh_token'] = True
            script_dir = os.path.dirname(os.path.realpath(__file__))
            secrets_file = os.path.join(script_dir, "client_secrets.json")
            self.gauth.LoadClientConfigFile(secrets_file)

            creds_file = os.path.join(script_dir, "mycreds.txt")
            self.gauth.LoadCredentialsFile(creds_file)
            
            if self.gauth.credentials is None:
                self.gauth.LocalWebserverAuth()
            elif self.gauth.access_token_expired:
                self.gauth.Refresh()
            else:
                self.gauth.Authorize()

            self.gauth.SaveCredentialsFile(creds_file)
            self.drive = GoogleDrive(self.gauth)
            self.get_logger().info("Google Drive Authentication Successful.")
            
        except Exception as e:
            self.get_logger().error(f"Authentication Failed: {str(e)}")
            return

        # --- THIS WAS MISSING ---
        # 5. Start the Timer (Run monitor_callback every 2 seconds)
        self.timer = self.create_timer(2.0, self.monitor_callback)
        self.get_logger().info(f"Monitoring started on: {self.source_path}")

    def monitor_callback(self):
        """Checks source directory for files."""
        try:
            # List all files in the directory
            files = [f for f in os.listdir(self.source_path) if os.path.isfile(os.path.join(self.source_path, f))]

            for file_name in files:
                full_file_path = os.path.join(self.source_path, file_name)
                
                # Upload Logic
                self.upload_to_drive(full_file_path, file_name)
                
                # Move Logic
                self.move_to_archive(full_file_path, file_name)

        except Exception as e:
            self.get_logger().error(f"Error in monitor loop: {str(e)}")

    def upload_to_drive(self, file_path, file_name):
        """Uploads the file to the specified Google Drive folder."""
        self.get_logger().info(f"Uploading {file_name}...")
        
        try:
            # Define file metadata
            file_metadata = {'title': file_name}
            
            # If a specific folder ID is provided, set it as the parent
            if self.drive_parent_id:
                file_metadata['parents'] = [{'id': self.drive_parent_id}]

            # Create and upload
            gfile = self.drive.CreateFile(file_metadata)
            gfile.SetContentFile(file_path)
            gfile.Upload()
            
            self.get_logger().info(f"Successfully uploaded: {file_name}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to upload {file_name}: {str(e)}")
            return False

    def move_to_archive(self, src_path, file_name):
        """Moves the file from source to archive."""
        dest_path = os.path.join(self.archive_path, file_name)
        try:
            shutil.move(src_path, dest_path)
            self.get_logger().info(f"Moved {file_name} to archive.")
        except Exception as e:
            self.get_logger().error(f"Failed to move {file_name}: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = UploadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()