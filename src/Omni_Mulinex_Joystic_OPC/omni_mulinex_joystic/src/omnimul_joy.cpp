#include "omni_mulinex_joystic/omnimul_joy.hpp"
#include "rclcpp/qos.hpp"
#include <cassert>
#include <cmath>
#include "rclcpp/serialization.hpp"

#define BASE_WS "/home/punk-opc/Documents/"
#define BAG_NAME "/Experiment_OM"
#define BAG_TOPIC "Joystic_Command"
#define CONTROLLER "state_broadcaster/"
#define OM_CONTROLLER "omni_control/"
#define X_VEL_AX 1
#define Y_VEL_AX 0
#define OM_VEL_AX 3
#define HEIGHT_VEL_AX 4
#define HOMING_BUTTON 0
#define EMERG_BUTTON 1
#define MAX_COUNTER_STT_STAMP 500

namespace omni_mulinex_joy
{
    using namespace std::chrono_literals;
    using std::placeholders::_1;
    void OmniMulinex_Joystic::get_param()
    {
        // get the parameter and saturate their value with the define values
        sup_vx_ =  this->get_parameter("sup_vel_x").as_double();
        sup_vx_ = sup_vx_>MAX_LIN_VEL?MAX_LIN_VEL:sup_vx_;
        sup_vy_ = this->get_parameter("sup_vel_y").as_double();
        sup_vy_ = sup_vy_>MAX_LIN_VEL?MAX_LIN_VEL:sup_vy_;
        sup_omega_ = this->get_parameter("sup_omega").as_double();
        sup_omega_ = sup_omega_>MAX_ROT_VEL?MAX_ROT_VEL:sup_omega_;
        sup_height_rate_ = this->get_parameter("sup_height_rate").as_double();
        sup_height_rate_ = sup_height_rate_>MAX_HEIGHT_RATE?MAX_HEIGHT_RATE:sup_height_rate_;
        register_state_ = this->get_parameter("save_state").as_bool();
        timer_dur_ = this->get_parameter("timer_duration").as_int();
        bag_folder_ = this->get_parameter("bag_folder").as_string();
        deadzone_ = this->get_parameter("deadzone_joy").as_double();
        if(register_state_)
            stt_period_ = this->get_parameter("state_duration").as_int();

    };

    void OmniMulinex_Joystic::set_tools()
    {
        rclcpp::QoS cmd_qos(10),stt_qos(10),input_qos(10);
        rclcpp::ServicesQoS srvs_qos = rclcpp::ServicesQoS();
        std::string bag_exp_name;
        std::chrono::milliseconds dur = std::chrono::milliseconds(timer_dur_), stt_dur = std::chrono::milliseconds(stt_period_);
        std::chrono::milliseconds dur_qos =std::chrono::milliseconds(timer_dur_ + 5);
        std::shared_ptr<rclcpp::CallbackGroup> node_cb_grp = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions opt_stt, input_opt;
        rclcpp::PublisherOptions opt_cmd;

        // get local time to make unique the bag name of each experiment
        time_t tm_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        auto lt_now = std::localtime(&tm_now);
        // save the bag path on string         
        bag_exp_name =  bag_folder_ + BAG_NAME + std::string("_") + std::to_string(lt_now->tm_year+1900) + "_" + std::to_string(lt_now->tm_mon+1) + "_" + std::to_string(lt_now->tm_mday) + "_" +
            std::to_string(lt_now->tm_hour) + ":" + std::to_string(lt_now->tm_min) + ":" +std::to_string(lt_now->tm_sec);

        // create the writer 
        writer_ = std::make_unique<rosbag2_cpp::Writer>();
        // try to open the new bag
        try
        {
            writer_->open(bag_exp_name);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            assert(true);
        }



        
        // set the QOS for the Wireless communication for both the command data, containing the joystic input, and the 
        // state data, provided by the interface 
        cmd_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        cmd_qos.deadline(dur_qos);
    
        // stt_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        // stt_qos.deadline(stt_dur);
        input_qos.deadline(dur);


        opt_cmd.event_callbacks.deadline_callback = 
        [this](rclcpp::QOSDeadlineOfferedInfo & event)
        {
            RCLCPP_WARN(this->get_logger(),"not respect pub time");
        };

        
      

        // associate to the subscriber both the callback group to access to the data variable with Mutex Exclusion and the event
        // callback tp detect the missed deadline if the parameter is set
        if(register_state_)
        {
            RCLCPP_WARN(this->get_logger(),"the state register is active");
            opt_stt.callback_group = node_cb_grp;
            opt_stt.event_callbacks.deadline_callback = [logger = this -> get_logger()](rclcpp::QOSDeadlineRequestedInfo& event)->void
            {
                RCLCPP_WARN(logger,"The state subscriber deadline has been missed, the counter is %d and its variation is %d"
                ,event.total_count,event.total_count_change);
            };
            stt_sub_ = this->create_subscription<OM_State>(CONTROLLER+std::string("joints_state"),stt_qos,std::bind(&OmniMulinex_Joystic::stt_callback,this,_1));  
        }
        // associate the subscription to joy topic to the CallbackGroup and the deadline event callback

        input_opt.callback_group = node_cb_grp;
        input_opt.event_callbacks.deadline_callback = [logger = this -> get_logger()](rclcpp::QOSDeadlineRequestedInfo& event)->void
        {
            RCLCPP_WARN(logger,"The command deadline has been missed, the counter is %d and its variation is %d"
            ,event.total_count,event.total_count_change);
        };

        // create the necessary subscriber, publisher and timer
        cmd_sub_ = this->create_subscription<JoyCommand>("joy",10,std::bind(&OmniMulinex_Joystic::joy_command,this,_1),input_opt);
        

        cmd_pub_ = this->create_publisher<OM_JoyCmd>(OM_CONTROLLER+std::string("command"),cmd_qos,opt_cmd);

        timer_ = this->create_wall_timer(dur,std::bind(&OmniMulinex_Joystic::main_callback,this),node_cb_grp);

        hom_srv_ = this->create_client<TransictionService>(OM_CONTROLLER+std::string("homing_srv"),srvs_qos.get_rmw_qos_profile(),node_cb_grp);
        emrgy_srv_ = this->create_client<TransictionService>(OM_CONTROLLER+std::string("emergency_srv"),srvs_qos.get_rmw_qos_profile(),node_cb_grp);



    }
    void OmniMulinex_Joystic::joy_command(const std::shared_ptr<JoyCommand> msg)
    {
        double n2_v;

        // axis 1 is vx, axis 0 is vy, axis 3 is omega and axis 4 is height rate
        v_x_ =  msg->axes[X_VEL_AX];
        v_x_ = (v_x_ < deadzone_ && v_x_  > - deadzone_ ) ? 0.0:sup_vx_*v_x_; 
        v_y_ = msg->axes[Y_VEL_AX];
        v_y_ = (v_y_ < deadzone_ && v_y_  > - deadzone_ ) ? 0.0:sup_vy_*v_y_;

       
        // normlize if the commanded velocity exed 1
        n2_v = std::sqrt(std::pow(v_x_,2) + std::pow(v_y_,2));
        // RCLCPP_INFO(this->get_logger(),"vx: %f, vy:%f and n2_v %f",v_x_,v_y_,n2_v);
        if(n2_v > MAX_LIN_VEL)
        {
            v_x_ *= (MAX_LIN_VEL/n2_v);
            v_y_ *= (MAX_LIN_VEL/n2_v);
        }
        // RCLCPP_INFO(this->get_logger(),"vx: %f, vy:%f",v_x_,v_y_);

        
        omega_ = msg->axes[OM_VEL_AX];
        omega_ = (omega_ < deadzone_ && omega_  > - deadzone_ ) ? 0.0:sup_omega_*omega_;
        h_rate_ = msg->axes[HEIGHT_VEL_AX];
        h_rate_ = (h_rate_ < deadzone_ && h_rate_  > - deadzone_ ) ? 0.0:sup_height_rate_*h_rate_;

        // RCLCPP_INFO(this->get_logger(),"h_rate %f",h_rate_);
        // RCLCPP_INFO(this->get_logger(),"%d",(!old_hom_but_ && msg->buttons[HOMING_BUTTON]==1.0));
        // RCLCPP_INFO(this->get_logger(),"PASS");
        if(!old_hom_but_ && msg->buttons[HOMING_BUTTON]==1.0)
        {

            RCLCPP_INFO(this->get_logger(),"%d",hom_srv_->service_is_ready());
            if(hom_srv_->service_is_ready())
            {
                srv_req_->data = true;
                old_hom_but_ = true;
                using ServiceResponseFuture = rclcpp::Client<TransictionService>::SharedFuture;
                auto response_received_callback = [this](ServiceResponseFuture future) {
                auto result = future.get();
                RCLCPP_INFO(this->get_logger(), "Result is: %s with message %s", result->success?std::string("True").c_str():std::string("False").c_str(),
                result->message.c_str());
                
                };
                auto res_f = hom_srv_->async_send_request(srv_req_,response_received_callback);
                
            }
        }
        else
            old_hom_but_ = false;
        if(!old_emg_but_ && msg->buttons[EMERG_BUTTON] == 1.0)
        {
            RCLCPP_INFO(this->get_logger(),"%d",emrgy_srv_->service_is_ready());
            if(emrgy_srv_->service_is_ready())
            {
                srv_req_->data = true;
                old_emg_but_ = true;
                using ServiceResponseFuture = rclcpp::Client<TransictionService>::SharedFuture;
                auto response_received_callback = [this](ServiceResponseFuture future) {
                auto result = future.get();
                RCLCPP_INFO(this->get_logger(), "Result is: %s with message %s", result->success?std::string("True").c_str():std::string("False").c_str(),
                result->message.c_str());
                
                };
                auto res = emrgy_srv_->async_send_request(srv_req_,response_received_callback);
            }
        }
        else    
            old_emg_but_ = false;
        
    }

    void OmniMulinex_Joystic::stt_callback(std::shared_ptr<OM_State> msg)
    {
        count_ ++;
        if(count_%MAX_COUNTER_STT_STAMP == 0)
        {
            
            auto temp = msg->temperature;
            auto cur = msg->current;

            // for(size_t i = 0; i <4; i++)
            //     RCLCPP_INFO(this->get_logger(),"the joint %s has temperature %f and current %f",
            //     focused_name_[i].c_str(),temp[i],cur[i]);
            RCLCPP_INFO(this->get_logger()," ");  
            count_ = 0;
        }
        

    }

    // void OmniMulinex_Joystic::stt_callback(std::shared_ptr<rclcpp::SerializedMessage> msg)
    // {
    //     rclcpp::Time time_stamp = this->now();
    //     const std::string name = CONTROLLER+std::string("joints_state");
    //     const std::string type = "pi3hat_moteus_int_msgs/msg/JointsStates";
    //     writer_-> write(msg,name,type,time_stamp);
        
    //     RCLCPP_INFO(this->get_logger(),"pass");
        

    // }

    void OmniMulinex_Joystic::main_callback()
    {
        // set the message 
        auto time_stamp = this->now();
        cmd_msg_.set__v_x(v_x_);
        cmd_msg_.set__v_y(v_y_);
        cmd_msg_.set__omega(omega_);
        cmd_msg_.set__height_rate(h_rate_);
       
        cmd_msg_.header.set__stamp(time_stamp);
       
        cmd_pub_->publish(cmd_msg_);
        writer_ -> write(cmd_msg_,BAG_TOPIC,time_stamp);
    }
}
