import math
import numpy as np
import sys
import math
import cv2
import time
import torch
import onnxruntime
#for ros image publish
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header, Float64MultiArray

num_privileged =5
class ObsInfo:
    def __init__(self) -> None:
        self.get_info = False
        self.obs_data = np.zeros((62+num_privileged,1), dtype=np.float64) # OBS + TIMESTAMP + PRIVI
        self.obs = np.zeros((61+num_privileged,1), dtype=np.float64)
        self.time_data = np.zeros((1,1), dtype=np.float64)
    
    def check(self):
        if not self.get_info:
            rospy.logwarn_throttle(1, "No lidar info received")

    def obs_callback(self,msg):
        # subscribe observation
        self.get_info = True
        self.obs_data = np.array(msg.data, dtype=np.float64).reshape((62+num_privileged,1)) 
        self.obs = self.obs_data[:61+num_privileged]
        self.time_data = self.obs_data[61+num_privileged]

def main():
    # estimator_model_torch_path = "../model/deploy/model_13600_privi.pt"
    estimator_model_torch_path = "../model/deploy/pen/pen_134000_estimator.pt"
    # estimator_model_torch_path = "../model/deploy/pen/agg_model_15000_privi.pt"
    # estimator_model_torch_path = "../model/deploy/model_10000_privi.pt"
    # estimator_model_torch_path = "../model/deploy/ex_mf_fusmodel_10000_privi.onnx"
    # estimator_model_torch_path = "../model/deploy/ex_mf_fusaggmodel_15000_privi.onnx"
    # estimator_model_path ="/home/orinnx/workspace/sim2real-locomotion-deployment/unitree_legged_sdk/model/abs_n_mass_co_train_dagger_stand_privi.onnx"
    # estimator_model_torch_path ="/home/orinnx/workspace/sim2real-locomotion-deployment/unitree_legged_sdk/model/cotrain/abs_n_mass_co_train_dagger_stand_privi.pt"
    # estimator_onnx_session = onnxruntime.InferenceSession(estimator_model_path)
    # esti_output_name = estimator_onnx_session.get_outputs()[0].name
    # esti_input_name = estimator_onnx_session.get_inputs()[0].name
    # estimator_model_torch_path ="/home/orinnx/workspace/sim2real-locomotion-deployment/unitree_legged_sdk/model/deploy/ex_mfmodel_10000_privi.pt"
    
    estimator = torch.load(estimator_model_torch_path).to("cpu")
    #ros init
    obs_info = ObsInfo()
    rospy.init_node('estimator', anonymous=True)
    pub_mass = rospy.Publisher('mass_value', Float64MultiArray, queue_size=10)
    rospy.Subscriber('observation', Float64MultiArray, obs_info.obs_callback)



    rate = rospy.Rate(50)
    obs_history = np.zeros((50,50),dtype=np.float32)
    obs = np.zeros((50),dtype=np.float32)
    esti_input = np.zeros(2451,dtype=np.float32)
    cur_obs_t = -1
    while not rospy.is_shutdown():
        # subscribe observation
        obs = obs_info.obs.squeeze(-1)
        time = obs_info.time_data

        if time != cur_obs_t: # update history
            obs_history = np.roll(obs_history, 1, axis=0)
            obs_history[0] = obs[:50]
            cur_obs_t = time
            pro_obs_history = np.concatenate((obs_history[:,:13], obs_history[:,14:50]),axis = -1)
            esti_input = np.concatenate((pro_obs_history.flatten(), obs[13:14]))
        esti_tensor = torch.from_numpy(esti_input).float()
        # mass = estimator_onnx_session.run([esti_output_name], {esti_input_name: esti_input})[0]
        mass = estimator(esti_tensor).detach().numpy()
        mass = np.clip(mass, -1, 11)
        mass_msg = Float64MultiArray()
        mass_msg.data = list(mass)
        pub_mass.publish(mass_msg)
        print(time, mass)
        

        rate.sleep()


if __name__ == "__main__":
    main()
