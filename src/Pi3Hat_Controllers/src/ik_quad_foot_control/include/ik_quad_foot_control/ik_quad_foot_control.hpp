#ifndef IK_QUAD_FOOT_HPP
#define IK_QUAD_FOOT_HPP
#include <memory>
#include <string>
#include <vector>
#include <array>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/subscription.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

#include "pi3hat_moteus_int_msgs/msg/quad_foot_state.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/set_bool.hpp"

#include <chrono>
#include <mutex>


#define BASE_REF 3

#define SIM true

using namespace std::chrono;



namespace ik_quad_foot_control
{

    enum Controller_State{
        INACTIVE = 0,
        HOMING,
        PUSHUP,
        STANDUP,
        ACTIVE,
    };
    enum LEG_IND {LF=0,LH,RF,RH};
    const std::vector<std::string> foot_name = {"LF","LH","RF","RH"};

    using CmdMsg = pi3hat_moteus_int_msgs::msg::QuadFootState;
    using SttMsg = sensor_msgs::msg::JointState;
    using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
    using TransactionService = std_srvs::srv::SetBool;

    class IK_Quad_Foot_Controller : public controller_interface::ControllerInterface
    {   
        public:
            IK_Quad_Foot_Controller(){};
            ~IK_Quad_Foot_Controller(){};

            
            CallbackReturn on_init() override;
            
            controller_interface::InterfaceConfiguration command_interface_configuration() const override;

            controller_interface::InterfaceConfiguration state_interface_configuration() const override;

            CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;


            CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override; 

            CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous_state) override; 

            CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

            controller_interface::return_type update(
                const rclcpp::Time & time, const rclcpp::Duration & period
            ) override;

            void IK_comp(double &q1, double &q2,double y, double x, LEG_IND leg, bool correct);
            void FK_comp(double q1, double q2,double &y, double &x, LEG_IND leg);

        private:

            void cmd_callback(const std::shared_ptr<CmdMsg> msg)
            {
                std::lock_guard<std::mutex> lg(var_mutex_);
                dl_miss_count_ = 0;

                if(c_stt_ == Controller_State::ACTIVE)
                {
                    feet_cmd_.set__foot_id(msg->foot_id);
                    feet_cmd_.set__position(msg->position);
                    feet_cmd_.set__velocity(msg->velocity);
                    feet_cmd_.set__force(msg->force);
                }
                // for(int i = 0; i<4;i++)
                // {
                //     pos = feet_cmd_.position[i];
                //     vel = feet_cmd_.velocity[i];
                //     forc = feet_cmd_.force[i];
                //     if(feet_cmd_.foot_id[i][1] != 'H')
                //     {

                //         feet_cmd_.position[i] = -pos;
                //         feet_cmd_.velocity[i] = -vel;
                //         feet_cmd_.force[i] = - forc;
                //     }
                // }
            }
            void command_sat(int i)
            {
                
                    if(std::abs(feet_cmd_.position[i].x) > bound_x_)
                    {
                        if(feet_cmd_.position[i].x>0)
                            feet_cmd_.position[i].x = bound_x_;
                        else
                            feet_cmd_.position[i].x = -bound_x_;
                        feet_cmd_.velocity[i].x=0;
                    }

                    if(std::abs(feet_cmd_.position[i].z) > bound_z_)
                    {
                        if(feet_cmd_.position[i].z>0)
                            feet_cmd_.position[i].z = bound_z_;
                        else
                            feet_cmd_.position[i].z = -bound_z_;
                        feet_cmd_.velocity[i].x=0;
                    }
                
            }
            
            void check_cont(double &q1, int jnt,bool correct)
            {   
                double jnt_diff = joint_cmd_app_.position[jnt] - q1;
                if(correct)
                {
                    if(jnt_diff > M_PI_2 && jnt_diff > 0)
                        joints_counters_[jnt]++;
                    if(jnt_diff < -M_PI_2 && jnt_diff < 0) 
                        joints_counters_[jnt]--;
                }
                joint_cmd_app_.position[jnt]  = q1;
                if(correct)
                {
                    if(joints_counters_[jnt] > 0)
                        q1 = (joints_counters_[jnt]+1)*M_PI +q1;
                    else if(joints_counters_[jnt] < 0)
                        q1 = (joints_counters_[jnt]-1)*M_PI +q1;
                }
            }
            void set_cmd2jnt(bool active)
            {
                // joint_cmd_.set__position(joint_cmd_app_.position);
               double scale;
               if(!active)
               	    scale=0.0;
                else
                    scale=1.0;
               for(size_t i = 0; i < jnt_names_.size(); i++)
                {
                    if(!SIM)
                    {
                        command_interfaces_[5*i].set_value(joint_cmd_.position[i]);
                        command_interfaces_[5*i + 1].set_value(0.0);
                        command_interfaces_[5*i + 2].set_value(0.0);
                        command_interfaces_[5*i + 3].set_value(scale);
                        command_interfaces_[5*i + 4].set_value(scale);
                       
                    }
                    else
                    {
                        command_interfaces_[i].set_value(joint_cmd_.position[i]);
                    }
                }
                
            }
            void get_stt()
            {
                
                for(size_t i = 0; i < jnt_names_.size(); i++)
                {
                    joint_stt_.position[i] = state_interfaces_[3*i].get_value();
                    joint_stt_.velocity[i] = state_interfaces_[3*i + 1].get_value();
                    joint_stt_.effort[i] = state_interfaces_[3*i + 2].get_value();
                }
            }

            void emergency_srv(const std::shared_ptr<TransactionService::Request> req, 
                                  const std::shared_ptr<TransactionService::Response> res)
            {
                std::lock_guard<std::mutex> lg(var_mutex_);
                // RCLCPP_INFO_STREAM(get_node()->get_logger(), req->data);
                // RCLCPP_INFO_STREAM(get_node()->get_logger(), c_stt_)
                if(c_stt_ != Controller_State::INACTIVE && req->data)
                {
                    c_stt_ = Controller_State::INACTIVE;
                    res ->success = true;
                    res ->message = std::string("Emergency mode has been activated");

                }
                else
                {
                    res ->success = false;
                    if(req->data)
                        res ->message = std::string("The state is just in Emergency mode");
                    else
                        res -> message = std::string("Request is not correct");                
                }
            }

            void  homing_start_srv(const std::shared_ptr<TransactionService::Request> req, 
                                  const std::shared_ptr<TransactionService::Response> res)
            {
                std::lock_guard<std::mutex> lg(var_mutex_);
                if(c_stt_ == Controller_State::INACTIVE && req->data)
                {
                    c_stt_ = Controller_State::HOMING;
                    res ->success = true;
                    not_active_timer_ = 0.0;
                    res ->message = std::string("Homing mode has been activated");
                    // RCLCPP_INFO(get_node()->get_logger(),"the inital pos is")
                }
                else
                {
                    res ->success = true;
                    if(req->data)
                        res ->message = std::string("The state is just in Homing mode");
                    else
                        res -> message = std::string("Request is not correct");  
                }

            }
            void  is_active_srv(const std::shared_ptr<TransactionService::Request> req, 
                                  const std::shared_ptr<TransactionService::Response> res)
            {
                std::lock_guard<std::mutex> lg(var_mutex_);
                if(c_stt_ == Controller_State::ACTIVE && req->data)
                {

                    res ->success = true;

                    res ->message = std::string("Active mode has been activated");
                    RCLCPP_INFO(get_node()->get_logger(),"THE ACTUAL STATE IS %d",c_stt_);
                }
                else
                {
                    res ->success = false;
                    if(req->data)
                        res ->message = std::string("The state is just in Active mode");
                    else
                        res -> message = std::string("Request is not correct");  
                }

            }
            rclcpp::Publisher<SttMsg>::SharedPtr joints_cmd_pub_;
            rclcpp::Subscription<CmdMsg>::SharedPtr cmd_sub_;
            
            SttMsg joint_cmd_,joint_cmd_app_,joint_stt_, init_jnt_pos_;
            CmdMsg feet_cmd_, feet_stt_, homing_pos_;
            std::string logger_name_;
            std::mutex var_mutex_;
            duration<double,std::milli> deadmis_to_;
            Controller_State c_stt_ = Controller_State::INACTIVE;
            int dl_miss_count_ = 0;
            std::vector<std::string> jnt_names_ ={"LF_HFE","LF_KFE","LH_HFE","LH_KFE","RF_HFE","RF_KFE","RH_HFE","RH_KFE" };
            // std::vector<double> homing_pos_;
            double push_up_, hom_dur_, push_dur_, not_active_timer_=0.0, bound_x_,bound_z_;
            std::vector<int> joints_counters_ = {
               0,0,0,0,0,0,0,0
            };
            
           
            rclcpp::Service<TransactionService>::SharedPtr homing_serv_,emergency_serv_,is_active_serv_;
    };
};


#endif
