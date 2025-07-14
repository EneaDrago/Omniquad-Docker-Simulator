
#include "pluginlib/class_list_macros.hpp"

#include "controller_interface/helpers.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include <cmath>
#include "ik_quad_foot_control/ik_quad_foot_control.hpp"


#define LEG_LENGTH 0.19
#define X_PUSHUP 0.0
using namespace std::chrono;
using namespace std::chrono_literals;
const std::vector<double> homing_jnt_pos = {-1.0471975511965976,2.0943951023931957,1.0471975511965976,-2.0943951023931957,1.0471975511965976,-2.0943951023931957,-1.0471975511965976,2.0943951023931957};
std::vector<double> pushup_jnt_pos = {-1.929780308020812,-2.6333785685984306,1.929780308020812,2.6333785685984306,1.929780308020812,2.6333785685984306,-1.929780308020812,-2.6333785685984306};
// const std::vector<double> standup_jnt_pos = {-4.1866667,-1.0472,4.1866667,1.0472,4.1866667,1.0472,-4.1866667,-1.0472};
const std::vector<double> standup_feet_pos = {-LEG_LENGTH/8, 0.2};

namespace ik_quad_foot_control
{

    void IK_Quad_Foot_Controller::IK_comp(double &q1, double &q2, double y, double x,LEG_IND leg,bool correct=false)
    {
        double q1_a,q2_a,c2,s2,c1,s1;

        if(leg == LEG_IND::LH || leg == LEG_IND::RH)
        {
            x *=-1;
            // y*= -1;
        }
            
        y *=-1;


        c2 =-( 1 - (std::pow(x,2) + std::pow(y,2))/(2* std::pow(LEG_LENGTH,2)));
        s2 =-std::sqrt(1 - std::pow(c2,2));
        // RCLCPP_ERROR_STREAM(get_node()->get_logger(),c_a<<" "<<s_a<<std::endl);
        q2_a = std::atan2(s2,c2);
        // q2 = (q2/std::abs(q2))*std::abs(M_PI - q2);
        // c2 = std::cos(q2);
        // s2 = std::sin(q2);
        // RCLCPP_ERROR_STREAM(get_node()->get_logger(),c2<<" "<<s2<<std::endl);
        // RCLCPP_ERROR_STREAM(get_node()->get_logger(),x <<"*(1+"<<c2<<")+("<<y<<"*"<<s2<<")"<<std::endl);
        // RCLCPP_ERROR_STREAM(get_node()->get_logger(),x <<"*(1+"<<c2<<")"<<std::endl);
        c1 = (x*(1+c2)+(y*s2))/(2*LEG_LENGTH*(1+c2));
        s1 = (y*(1+c2)- x*s2)/(2*LEG_LENGTH*(1+c2));
        // RCLCPP_ERROR_STREAM(get_node()->get_logger(),leg<<" elab start");
        
        q1_a = std::atan2(s1,c1);
        if(leg == LEG_IND::RF || leg == LEG_IND::LH)
        {
            q1_a *=-1;
            q2_a *= -1;
        }
    
        check_cont(q1_a,2*leg,correct);
        
        check_cont(q2_a,2*leg+1,correct);
        
        

        q1 = q1_a;
        q2 = q2_a;

       
        


    };

    void IK_Quad_Foot_Controller::FK_comp(double q1, double q2, double &y, double &x,LEG_IND leg)
    {
        // if(leg == LEG_IND::RH || leg == LEG_IND::LF)
        // {
        //     q1 *=-1;
        //     q2 *= -1;
        // }

        y = LEG_LENGTH*(std::sin(q1) + std::sin(q1+q2));
        x = LEG_LENGTH*(std::cos(q1) + std::cos(q1+q2));

        // if(leg == LEG_IND::LH || leg == LEG_IND::RH)
        // {
        //     x *=-1;
        //     // y *= -1;
        // }
    }

    CallbackReturn IK_Quad_Foot_Controller::on_init()
    {
        try
        {
            auto_declare<int>("input_frequency",150);   
            auto_declare<std::vector<double>>("homing_pos",std::vector<double>());
            auto_declare<double>("homing_dur",10);
            auto_declare<double>("pushup_height",0.1);
            auto_declare<double>("pushup_dur",10);
            auto_declare<double>("x_pos_bound",0.3);
            auto_declare<double>("z_pos_bound",0.1);
            auto_declare<bool>("BestEffort_QOS",true);
            auto_declare<bool>("DeadMiss_event",true);
            auto_declare<bool>("call_dm",false);
        }
         catch(const std::exception & e)
        {
            RCLCPP_ERROR(rclcpp::get_logger(logger_name_),"Exception thrown during declaration of joints name with message: %s", e.what());
            return CallbackReturn::ERROR;
        }
        // add physics omni_wheel physics parameter autodeclare
        RCLCPP_INFO(get_node()->get_logger(),"initialize succesfully");
        return CallbackReturn::SUCCESS;
    };
       CallbackReturn IK_Quad_Foot_Controller::on_configure(const rclcpp_lifecycle::State & )
    {
        
        rclcpp::QoS out_qos(10),in_qos(10);
        // get parameters
        
        auto homing_par = get_node()->get_parameter("homing_pos").as_double_array();
        hom_dur_ = get_node()->get_parameter("homing_dur").as_double();
        push_up_ = get_node()->get_parameter("pushup_height").as_double();
        push_dur_ =get_node()->get_parameter("pushup_dur").as_double();
        bound_x_ = get_node()->get_parameter("x_pos_bound").as_double();
        bound_z_ = get_node()->get_parameter("z_pos_bound").as_double();
        milliseconds dur{get_node()->get_parameter("input_frequency").as_int() + 5};
        RCLCPP_INFO(get_node()->get_logger(),"the foot bound are %f,%f",bound_x_,bound_z_);
        deadmis_to_ = dur;

        // fill base to wheel kin matrix

        size_t dof = jnt_names_.size();
        
        joint_cmd_.name.resize(dof);
        joint_cmd_.set__name(jnt_names_);
        joint_cmd_.position.resize(dof);
        joint_cmd_.velocity.resize(dof);
        joint_cmd_.effort.resize(dof);
        joint_stt_.name.resize(dof);
        joint_stt_.set__name(jnt_names_);
        joint_stt_.position.resize(dof);
        joint_stt_.velocity.resize(dof);
        joint_stt_.effort.resize(dof);
        init_jnt_pos_.set__name(jnt_names_);
        init_jnt_pos_.position.resize(dof);
        init_jnt_pos_.velocity.resize(dof);
        init_jnt_pos_.effort.resize(dof);
        joint_cmd_app_.set__name(jnt_names_);
        joint_cmd_app_.position.resize(dof);
        joint_cmd_app_.velocity.resize(dof);
        joint_cmd_app_.effort.resize(dof);
        feet_cmd_.foot_id.resize(4);
        feet_cmd_.position.resize(4);
        feet_cmd_.velocity.resize(4);
        feet_cmd_.force.resize(4);
        feet_stt_.foot_id.resize(4);
        feet_stt_.position.resize(4);
        feet_stt_.velocity.resize(4);
        feet_stt_.force.resize(4);
        homing_pos_.foot_id.resize(4);
        homing_pos_.position.resize(4);
        homing_pos_.velocity.resize(4);
        homing_pos_.force.resize(4);
        homing_pos_.set__foot_id(foot_name);
        for(int i = 0; i < 4; i++)
        {
            feet_stt_.position[i].set__x(0.0);
            feet_stt_.position[i].set__y(0.0);
            feet_stt_.position[i].set__z(0.0);
            feet_stt_.velocity[i].set__x(0.0);
            feet_stt_.velocity[i].set__y(0.0);
            feet_stt_.velocity[i].set__z(0.0);
            feet_cmd_.position[i].set__x(0.0);
            feet_cmd_.position[i].set__y(0.0);
            feet_cmd_.position[i].set__z(0.0);
            feet_cmd_.velocity[i].set__x(0.0);
            feet_cmd_.velocity[i].set__y(0.0);
            feet_cmd_.velocity[i].set__z(0.0);
            homing_pos_.position[i].x = homing_par[2*i];
            homing_pos_.position[i].z = homing_par[2*i + 1];
        }
        
        //set qos protocol
        if(get_node()->get_parameter("BestEffort_QOS").as_bool())
        {
            out_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
            in_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        }
        else
        {
            out_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
            in_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        }
        //set the in qos for deadline miss

        in_qos.deadline(deadmis_to_);
        rclcpp::SubscriptionOptions sub_opt;
        sub_opt.event_callbacks.deadline_callback = 
        [this](rclcpp::QOSDeadlineRequestedInfo & event) 
        {
            this->dl_miss_count_ ++;
            if(dl_miss_count_ > 10)
            {
                std::lock_guard<std::mutex> l_g(var_mutex_);
                c_stt_ = Controller_State::INACTIVE;
            }
            RCLCPP_WARN(this->get_node()->get_logger(),"Passed deadline");
        };
        //create subscriber and publisher
        cmd_sub_ = get_node()->create_subscription<CmdMsg>(
            "~/command",in_qos,std::bind(&IK_Quad_Foot_Controller::cmd_callback,this,std::placeholders::_1),sub_opt);

        joints_cmd_pub_ = get_node()->create_publisher<SttMsg>("~/joints_reference",out_qos);


        // create servicies 

        rclcpp::ServicesQoS srvs_qos;


        homing_serv_ = get_node()->create_service<TransactionService>("~/homing_srv",
        std::bind(&IK_Quad_Foot_Controller::homing_start_srv,this,std::placeholders::_1,std::placeholders::_2),
        srvs_qos.get_rmw_qos_profile()
        );

        emergency_serv_ = get_node()->create_service<TransactionService>("~/emergency_srv",
        std::bind(&IK_Quad_Foot_Controller::emergency_srv,this,std::placeholders::_1,std::placeholders::_2),
        srvs_qos.get_rmw_qos_profile()
        );
        
        is_active_serv_ = get_node()->create_service<TransactionService>("~/is_active_srv",
        std::bind(&IK_Quad_Foot_Controller::is_active_srv,this,std::placeholders::_1,std::placeholders::_2),
        srvs_qos.get_rmw_qos_profile()
        );
        RCLCPP_INFO(get_node()->get_logger(),"configure succesfully");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn IK_Quad_Foot_Controller::on_activate(const rclcpp_lifecycle::State & )
    {
       
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn IK_Quad_Foot_Controller::on_deactivate(const rclcpp_lifecycle::State & )
    {
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn IK_Quad_Foot_Controller::on_cleanup(const rclcpp_lifecycle::State &)
    {
        return CallbackReturn::SUCCESS;
    }

     controller_interface::InterfaceConfiguration IK_Quad_Foot_Controller::state_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration stt_int_cnf;
        stt_int_cnf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
        for(auto &it : jnt_names_)
        {
            stt_int_cnf.names.push_back(it + "/" + hardware_interface::HW_IF_POSITION);
            stt_int_cnf.names.push_back(it + "/" + hardware_interface::HW_IF_VELOCITY);
            stt_int_cnf.names.push_back(it + "/" + hardware_interface::HW_IF_EFFORT);
        }
        return stt_int_cnf;
    }

    controller_interface::InterfaceConfiguration IK_Quad_Foot_Controller::command_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration cmd_int_cnf;
        cmd_int_cnf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
        for(auto &it : jnt_names_)
        {
            
            
            if(!SIM)
            {
            	cmd_int_cnf.names.push_back(it + "/" + hardware_interface::HW_IF_POSITION);
                cmd_int_cnf.names.push_back(it + "/" + hardware_interface::HW_IF_VELOCITY);
                cmd_int_cnf.names.push_back(it + "/" + hardware_interface::HW_IF_EFFORT);
                cmd_int_cnf.names.push_back(it + "/" + "kp_scale_value");
                cmd_int_cnf.names.push_back(it + "/" + "kd_scale_value");
            }
            else
                cmd_int_cnf.names.push_back(it + "/" + hardware_interface::HW_IF_POSITION);
        }
        return cmd_int_cnf;
    }
    controller_interface::return_type IK_Quad_Foot_Controller::update(const rclcpp::Time & time, const rclcpp::Duration & period)
    {

        // get joint state
        get_stt();
        double q1=0,q2=0,t,k;
        // entry in mutex 
        std::lock_guard<std::mutex> l_g(var_mutex_);
        switch (c_stt_)
        {
        case INACTIVE:
            // make the FK to get the actual position and store it, send command with kp/kd zero  or the readed joint state in sim 
            for(int i =0; i<4; i++)
            {
                feet_stt_.foot_id[i] = foot_name[i];
                // FK_comp(
                //     joint_stt_.position[2*i],
                //     joint_stt_.position[2*i+1],
                //     feet_stt_.position[i].z,
                //     feet_stt_.position[i].x,
                //     (LEG_IND)i);
                // feet_cmd_.set__foot_id(feet_stt_.foot_id);
                // feet_cmd_.set__position(feet_stt_.position);
         
                
            }
		joint_cmd_.set__name(joint_stt_.name);
                joint_cmd_.set__position(joint_stt_.position);
                joint_cmd_.velocity.resize(joint_cmd_.position.size(),0.0);

            // RCLCPP_INFO(get_node()->get_logger(),"the homig feet pos has value %f,%f",
            // joint_stt_.position[4],joint_stt_.position[5]);
            init_jnt_pos_.set__position(joint_stt_.position);
            // for(int i =0; i< 4 ; i++)
            //     RCLCPP_INFO(get_node()->get_logger(), "feet start  %d is %f,%f",i,feet_stt_.position[i].x,feet_stt_.position[i].z);
            set_cmd2jnt(false);
            // FK_comp(q1,q2,k,t,(LEG_IND)0);
            // RCLCPP_INFO(get_node()->get_logger()," the position ee is %f,%f",t,k);
            // IK_comp(q1,q2,k,t,(LEG_IND)0);
            
           
            
            break;
        case HOMING:
            not_active_timer_ += period.seconds();
            // compute the linear interpolation from starting to homing
            if(not_active_timer_ >= hom_dur_)
                not_active_timer_ = hom_dur_;
            
            for(int i = 0; i < 8; i++)
            {
                joint_cmd_.position[i] = homing_jnt_pos[i]*(not_active_timer_/hom_dur_)  +
                    init_jnt_pos_.position[i]*(1 - (not_active_timer_/hom_dur_));
                // feet_cmd_.position[i].set__z(
                //     homing_pos_.position[i].z*(not_active_timer_/hom_dur_)  +
                //     feet_stt_.position[i].z*(1 - (not_active_timer_/hom_dur_))
                //     );
                
                
                // IK_comp(
                //     joint_cmd_.position[2*i],
                //     joint_cmd_.position[2*i + 1],
                //     feet_cmd_.position[i].z,feet_cmd_.position[i].x,(LEG_IND)i);
                // FK_comp( joint_cmd_.position[2*i],
                //     joint_cmd_.position[2*i + 1],
                //     a,b,(LEG_IND)i);
                // RCLCPP_INFO(get_node()->get_logger(),"the computed error is %f %f",a,b);

            }
            
            set_cmd2jnt(true);
            if(not_active_timer_ == hom_dur_)
            {
                c_stt_ = Controller_State::PUSHUP;
                not_active_timer_ = 0.0;
                RCLCPP_INFO(get_node()->get_logger(),"[%f,%f]",pushup_jnt_pos[0],pushup_jnt_pos[1]);

                for(int i = 0; i< 4; i++)
                {
                    if(i == LEG_IND::LH || i == LEG_IND::RH)
                        IK_comp(pushup_jnt_pos[2*i],pushup_jnt_pos[2*i+1],standup_feet_pos[1],-standup_feet_pos[0],(LEG_IND) i,false);
                    else
                        IK_comp(pushup_jnt_pos[2*i],pushup_jnt_pos[2*i+1],standup_feet_pos[1],standup_feet_pos[0],(LEG_IND) i,false);
                    RCLCPP_INFO(get_node()->get_logger(),"the computed %d IK for pushup are [%f,%f]",pushup_jnt_pos[2*i],pushup_jnt_pos[2*i + 1]);
                    // if(i == 2)
                    // RCLCPP_INFO_STREAM(get_node()->get_logger(),feet_stt_.position[i].z<<" "<<homing_pos_.position[i].z);
                }
                
            }
            //use period to update internal timer to execute homing
                // compute the reference as interpolation linear between the first stt inetactive and desireh homing
            //reset timer 
                // compute the reference as interpolation linear between the desired homing and pushup pos
                // go to active

            break;
        case PUSHUP:
            not_active_timer_ += period.seconds();
            // compute the linear interpolation from starting to homing
            if(not_active_timer_ >= hom_dur_)
                not_active_timer_ = hom_dur_;
            for(int i = 0; i < 8; i++)
            {

                joint_cmd_.position[i] = pushup_jnt_pos[i]*(not_active_timer_/hom_dur_)  +
                    homing_jnt_pos[i]*(1 - (not_active_timer_/hom_dur_));
                // feet_cmd_.position[i].set__x(
                //     feet_stt_.position[i].x*(not_active_timer_/hom_dur_)  +
                //     homing_pos_.position[i].x*(1 - (not_active_timer_/hom_dur_))
                //     );
                // feet_cmd_.position[i].set__z(
                //     feet_stt_.position[i].z*(not_active_timer_/hom_dur_)  +
                //     homing_pos_.position[i].z*(1 - (not_active_timer_/hom_dur_))
                //     );
                // feet_cmd_.position[i].set__y(0.0);

                // IK_comp(
                //     joint_cmd_.position[2*i],
                //     joint_cmd_.position[2*i + 1],
                //     feet_cmd_.position[i].z,feet_cmd_.position[i].x,(LEG_IND)i);
            }
            if(not_active_timer_ >= hom_dur_)
            {
                c_stt_ = Controller_State::STANDUP;
               
                not_active_timer_ = 0.0;
                joint_cmd_app_.set__position(joint_cmd_.position);
                
                // feet_cmd_.set__velocity(feet_stt_.position);

            }
            set_cmd2jnt(true);
            break;
        case STANDUP:
            not_active_timer_ += period.seconds();
            // compute the linear interpolation from starting to homing
            if(not_active_timer_ >= push_dur_)
                not_active_timer_ = push_dur_;
            for(int i = 0; i < 4; i++)
            {

            //    joint_cmd_.position[i] = standup_jnt_pos[i]*(not_active_timer_/hom_dur_)  +
            //         pushup_jnt_pos[i]*(1 - (not_active_timer_/hom_dur_));
                feet_cmd_.position[i].set__x(
                    X_PUSHUP*(not_active_timer_/push_dur_)  +
                    standup_feet_pos[0]*(1 - (not_active_timer_/push_dur_))
                    );
                feet_cmd_.position[i].set__z(
                    push_up_*(not_active_timer_/push_dur_)  +
                   standup_feet_pos[1]*(1 - (not_active_timer_/push_dur_))
                    );
                
                feet_cmd_.position[i].set__y(0.0);
                command_sat(i);
                if(i == LEG_IND::LH || i == LEG_IND::RH)
                        IK_comp(
                    joint_cmd_.position[2*i],
                    joint_cmd_.position[2*i + 1],
                    feet_cmd_.position[i].z,-feet_cmd_.position[i].x,(LEG_IND)i,true);
                    else
                        IK_comp(
                    joint_cmd_.position[2*i],
                    joint_cmd_.position[2*i + 1],
                    feet_cmd_.position[i].z,feet_cmd_.position[i].x,(LEG_IND)i,true);
                
            }

            if(not_active_timer_ >= push_dur_)
            {
                c_stt_ = Controller_State::ACTIVE;
               	for(int i = 0; i<4;i++)
               	{
               		feet_cmd_.position[i].set__x(
                    X_PUSHUP
                    );
                    feet_cmd_.position[i].set__z(
                    push_up_);
               	}	
                // not_active_timer_ = 0.0;
                // feet_cmd_.set__velocity(feet_stt_.position);

            }
            set_cmd2jnt(true);
            break;
        case ACTIVE:
            for(int i = 0; i < 4; i++)
            {
            	command_sat(i);
                if(i == LEG_IND::LH || i == LEG_IND::RH)
                        IK_comp(
                    joint_cmd_.position[2*i],
                    joint_cmd_.position[2*i + 1],
                    feet_cmd_.position[i].z,-feet_cmd_.position[i].x,(LEG_IND)i,true);
                    else
                        IK_comp(
                    joint_cmd_.position[2*i],
                    joint_cmd_.position[2*i + 1],
                    feet_cmd_.position[i].z,feet_cmd_.position[i].x,(LEG_IND)i,true);
            }
            set_cmd2jnt(true); 

            break;
        default:
            break;
        }
       	//RCLCPP_INFO(get_node()->get_logger(),"HAHAAHAHAH the RF foot position are %f,%f",feet_cmd_.position[(int)LEG_IND::RF].x,feet_cmd_.position[(int)LEG_IND::RF].z);
        joints_cmd_pub_->publish(joint_cmd_);
        return controller_interface::return_type::OK;
    }
};


PLUGINLIB_EXPORT_CLASS(
    ik_quad_foot_control::IK_Quad_Foot_Controller, controller_interface::ControllerInterface
);



