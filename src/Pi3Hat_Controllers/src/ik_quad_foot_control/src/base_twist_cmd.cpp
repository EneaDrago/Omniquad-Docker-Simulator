#include <memory>
#include <string>
#include <vector>
#include <array>

#include "pi3hat_moteus_int_msgs/msg/quad_foot_state.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/set_bool.hpp"

#include <chrono>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#define LEG_LENGTH 0.19
#define BASE_LENGTH 0.235
 

namespace ik_base_cmd
{
    using namespace std::chrono;
    using namespace std::chrono_literals;
    using CmdMsg = pi3hat_moteus_int_msgs::msg::QuadFootState;
    using TransictionService = std_srvs::srv::SetBool;
    enum BaseMotion
    {
        LIN_X=0,
        LIN_Z,
        PITCH
    };
    class IK_Base_Command : public rclcpp::Node
    {
        public:
            IK_Base_Command():rclcpp::Node("ik_base_cmd"){
            
                // declare parameters 
                this->declare_parameter<bool>("BestEffort_QOS",true);
                this->declare_parameter<bool>("DeadMiss_event",false);
                this->declare_parameter<int>("input_frequency",100);
                this->declare_parameter<int>("motion",0);
                this->declare_parameter<double>("amplitude",0.5);
                this->declare_parameter<double>("frequency",0.1);
                this->declare_parameter<double>("init_feet_pos_x",-LEG_LENGTH/8);
                this->declare_parameter<double>("init_feet_pos_z",-0.25);
                


                // get parameters
                ampl_ = this->get_parameter("amplitude").as_double();
                freq_ = this->get_parameter("frequency").as_double();
                int motion =  this->get_parameter("motion").as_int();
                init_p_x_ = this->get_parameter("init_feet_pos_x").as_double();
                init_p_z_ = this->get_parameter("init_feet_pos_z").as_double();
                motion =2;
                this->set_parameter(rclcpp::Parameter("use_sim_time", true));
 		if(motion == 0)
 			mot_ = BaseMotion::LIN_X;
 		else if(motion == 0)
 			mot_ = BaseMotion::LIN_Z;
 		else 
 			mot_ = BaseMotion::PITCH;
                milliseconds dur{this->get_parameter("input_frequency").as_int()};
		rclcpp::ServicesQoS srvs_qos = rclcpp::ServicesQoS();
		std::shared_ptr<rclcpp::CallbackGroup> node_cb_grp = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
		
                feet_cmd_.position.resize(4);
                feet_cmd_.velocity.resize(4);
                feet_cmd_.force.resize(4);
                feet_cmd_.foot_id.resize(4);
                feet_cmd_.set__foot_id({"LF","LH","RF","RH"});
                rclcpp::QoS qos(10);
                
                srv_req_ = std::make_shared<TransictionService::Request>();
                srv_res_ = std::make_shared<TransictionService::Response>();
			
                //set qos protocol
                if(this->get_parameter("BestEffort_QOS").as_bool())
                    qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
                else   
                    qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
                qos.deadline(dur);

                foot_cmd_pub_ =this->create_publisher<CmdMsg>("ik_foot_contr/command",qos);

                //set up message 
                for(int i = 0; i < 4; i++)
                {
                    if(i%2==0)
                        feet_cmd_.position[i].x = init_p_x_;
                    else
                        feet_cmd_.position[i].x = -init_p_x_;
                    feet_cmd_.position[i].z = init_p_z_;
                }
                
                //setup services client
                hom_srv_ = this->create_client<TransictionService>("ik_foot_contr/"+std::string("homing_srv"),srvs_qos.get_rmw_qos_profile());
            	emrgy_srv_ = this->create_client<TransictionService>("ik_foot_contr/"+std::string("emergency_srv"),srvs_qos.get_rmw_qos_profile());
                is_active_srv_ = this->create_client<TransictionService>("ik_foot_contr/"+std::string("is_active_srv"),srvs_qos.get_rmw_qos_profile());
                
                 //set up timer 
                timer_ = this->create_wall_timer(dur,std::bind(&IK_Base_Command::timer_callback,this));
                RCLCPP_INFO(get_logger(),"Base Motion Node has been created");
                
            };
            ~IK_Base_Command(){};
            void timer_callback()
            {
            
            	
            	if(!homing_req_)
            	{
            		
    			if(hom_srv_->service_is_ready())
			{
				 srv_req_->data = true;
				 using ServiceResponseFuture = rclcpp::Client<TransictionService>::SharedFuture;
				 auto response_received_callback = [this](ServiceResponseFuture future) {
				 auto result = future.get();
				 RCLCPP_INFO(this->get_logger(), "Result is: %s with message %s", result->success?std::string("True").c_str():std::string("False").c_str(),
				 result->message.c_str());
				 if(result->success)
				 	this->homing_req_ = true;
				
				};
				auto res_f = hom_srv_->async_send_request(srv_req_,response_received_callback);
				
				
			}
			else
			{
				RCLCPP_WARN(this->get_logger(),"The homing service is not active");
				this->homing_req_ = false;
			}
            	}
            	else if(!homing_done_ && homing_req_)
            	{
            		
            		if(is_active_srv_->service_is_ready())
			{
				 srv_req_->data = true;
				 using ServiceResponseFuture = rclcpp::Client<TransictionService>::SharedFuture;
				 auto response_received_callback = [this](ServiceResponseFuture future) {
				 auto result = future.get();
				 RCLCPP_INFO(this->get_logger(), "Result is: %s with message %s", result->success?std::string("True").c_str():std::string("False").c_str(),
				 result->message.c_str());
			  	if(result->success)
					this->homing_done_ = true;
				
				};
				auto res_f = is_active_srv_->async_send_request(srv_req_,response_received_callback);
				
			}
			else
				RCLCPP_WARN(this->get_logger(),"The active service is not active");
            	}
            	else if(homing_done_ && homing_req_ && get_clock()->get_clock_type()==1)
            	{
            		rclcpp::Time now;
            		double dur;
            		RCLCPP_INFO(get_logger(),"the time is %f",t_);
            		if(first_cmd_)
            		{
            			old_t_ = this->get_clock()->now();
            		 	first_cmd_ = false;
            		} 
            		now =  this->get_clock()->now();
            			
            		dur = now.seconds() - old_t_.seconds();
            		old_t_ = now;
            		t_ += dur;
            		RCLCPP_INFO(get_logger(),"the sec is %f and the new is %f",now.seconds(),old_t_.seconds());
            		
            		
            		base_to_feet(t_);
            		RCLCPP_INFO(get_logger(),"the foot pose is %f,%f at %f", feet_cmd_.position[2].x, feet_cmd_.position[2].z,t_);
            		RCLCPP_INFO(get_logger(),"the foot pose is %f,%f at %f", feet_cmd_.position[3].x, feet_cmd_.position[3].z,t_);
            		foot_cmd_pub_->publish(feet_cmd_);	
            	}
		
            }

            void base_to_feet(double t)
            {
            double pitch =0.0;
            double pitch_pi= 0.0;
                    switch (mot_)
                    {
                    case LIN_Z:
                        for(int i = 0; i < 4; i++)
                        {
                            feet_cmd_.position[i].z = init_p_z_ + ampl_*sin(2*M_PI*freq_*t);
                        }
                        break;
                    case LIN_X:
                        for(int i = 0; i < 4; i++)
                        {
                            if(i%2==0)
                                feet_cmd_.position[i].x = init_p_x_ + ampl_*sin(2*M_PI*freq_*t);
                            else
                                feet_cmd_.position[i].x = init_p_x_ - ampl_*sin(2*M_PI*freq_*t);
                        }
                        break;
                    case PITCH:
                    	pitch =  ampl_*sin(2*M_PI*freq_*t);
                    	pitch_pi =  ampl_*sin(2*M_PI*freq_*t + M_PI);
                    	
                    	for(int i = 0; i<4; i++)
                    	{
                    		if(i%2==0)
                    		{
                    			//feet_cmd_.position[i].z = init_p_z_ + ampl_*sin(2*M_PI*freq_*t);
                    			feet_cmd_.position[i].x = (init_p_x_ + BASE_LENGTH/2)*std::cos(pitch) - init_p_z_ * std::sin(pitch) - BASE_LENGTH/2;
                    			feet_cmd_.position[i].z = (init_p_x_ + BASE_LENGTH/2)* std::sin(pitch) + init_p_z_ * std::cos(pitch);
                    		}
                    		else
                    		{
                    			//feet_cmd_.position[i].z = init_p_z_ - ampl_*sin(2*M_PI*freq_*t);
                    			feet_cmd_.position[i].x = (init_p_x_ + BASE_LENGTH/2)*std::cos(pitch_pi) - init_p_z_ * std::sin(pitch_pi) - BASE_LENGTH/2;
                    			feet_cmd_.position[i].z = (init_p_x_ + BASE_LENGTH/2)* std::sin(pitch_pi) + init_p_z_ * std::cos(pitch_pi);
                    		}
                    	}
                    	break;
                    default:
                        break;
                    }
            }
        private:

            double freq_,ampl_,init_p_x_,init_p_z_;
            
            BaseMotion mot_;
            CmdMsg feet_cmd_;
            rclcpp::Publisher<CmdMsg>::SharedPtr foot_cmd_pub_;
            bool homing_done_ = false, homing_req_=false;
            std::shared_ptr<rclcpp::Client<TransictionService>> hom_srv_,emrgy_srv_, is_active_srv_;
            std::shared_ptr<TransictionService::Request> srv_req_;
            std::shared_ptr<TransictionService::Response> srv_res_;
            double t_;
            rclcpp::Time old_t_;
		            std::shared_ptr<rclcpp::TimerBase> timer_;
		            bool first_cmd_=true;

    };
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ik_base_cmd::IK_Base_Command>());
  rclcpp::shutdown();
  return 0;
}
  
