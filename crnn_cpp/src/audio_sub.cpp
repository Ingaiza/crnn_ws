#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"
#include "example_interfaces/msg/bool.hpp"

class AudioSubscriber : public rclcpp::Node 
{
public:
    AudioSubscriber() : Node("audio_sub") 
    {   
        save_path_ = "/home/ingaiza/GoogleDrive/Ambience/Ambience";
        target_samples_= 15 * 16000;
        
        mkdir(save_path_.c_str(), 0777);

        subscription_ = this->create_subscription<std_msgs::msg::Int16MultiArray>("ambience",10,std::bind(&AudioSubscriber::sub_callback,this,std::placeholders::_1));
        fire_sub_ = this->create_subscription<example_interfaces::msg::Bool>("fire",1,std::bind(&AudioSubscriber::fire_callback,this,std::placeholders::_1));

        RCLCPP_WARN(this->get_logger(), "Audio Subscription Initialized");
    }

private:
    void sub_callback(const std_msgs::msg::Int16MultiArray::SharedPtr msg)
    {
        audio_buffer_.insert(audio_buffer_.end(), msg->data.begin(), msg->data.end());

        if(audio_buffer_.size() >= target_samples_)
        {
            save_wav_file();
            audio_buffer_.clear();

        }

    }

    void fire_callback(const example_interfaces::msg::Bool::SharedPtr msg)
    {
        if(!fire_)
        {
            fire_ = msg->data;
        }
        else if(fire_ && is_status_sent_)
        {
            fire_ = msg->data;
            is_status_sent_ = false;
        }
    }

    void save_wav_file()
    {
        //Timestamp file name
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss<< std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
        std::string filename;

        if(fire_)
        {
            filename = save_path_ + "_YES_" + ss.str() + ".wav";
            is_status_sent_ = true;
        }
        else
        {
            filename = save_path_ + "_NO_" + ss.str() + ".wav";
            is_status_sent_ = true;
        }

        // WAV header and data
        std::ofstream f(filename, std::ios::binary);

        // WAV header parameters
        int32_t sample_rate = 16000;
        int16_t num_channels = 1;
        int16_t bits_per_sample = 16;
        int32_t data_chunk_size = audio_buffer_.size() * sizeof(int16_t);
        int32_t file_size = 36 + data_chunk_size;

        // Header writing (Standard PCM WAV)
        f.write("RIFF", 4);
        f.write(reinterpret_cast<const char*>(&file_size), 4);
        f.write("WAVE", 4);
        f.write("fmt ", 4);
        int32_t fmt_chunk_size = 16;
        f.write(reinterpret_cast<const char*>(&fmt_chunk_size), 4);
        int16_t audio_format = 1; // PCM
        f.write(reinterpret_cast<const char*>(&audio_format), 2);
        f.write(reinterpret_cast<const char*>(&num_channels), 2);
        f.write(reinterpret_cast<const char*>(&sample_rate), 4);
        int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
        f.write(reinterpret_cast<const char*>(&byte_rate), 4);
        int16_t block_align = num_channels * bits_per_sample / 8;
        f.write(reinterpret_cast<const char*>(&block_align), 2);
        f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
        f.write("data", 4);
        f.write(reinterpret_cast<const char*>(&data_chunk_size), 4);

        // Write Audio Data
        f.write(reinterpret_cast<const char*>(audio_buffer_.data()), data_chunk_size);
        f.close();

        RCLCPP_INFO(this->get_logger(), "Saved: %s", filename.c_str());

    }

    rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr subscription_;
    rclcpp::Subscription<example_interfaces::msg::Bool>::SharedPtr fire_sub_;
    std::vector<int16_t> audio_buffer_;
    size_t target_samples_;
    std::string save_path_;
    bool volatile fire_ = false;
    bool volatile is_status_sent_ = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AudioSubscriber>(); 
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
