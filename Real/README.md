# Real

This directory has all of the files used to fly the real crazyflie 2.x using the crazyradio and VICON motioncapture system.

AgentControl.py, AgentControl2.py, and AgentControl3.py are all evolutions of the same control structure with AgentControl3.py being the most updated. AgentControl3.py includes methods to control using both MLP and MLP+LSTM modules. There are a few dependencies that need to be installed in order to use these such as vicondssdk and the crazyflie lib for python. ***In order to use the direct motor control you must edit the following line of code of send_setpoint in commander.py in the crazyflie lib***

`pk.data = struct.pack('<fffH',roll, -pitch, yawrate, thrust)`

to

`pk.data = struct.pack('<HHHH',roll, pitch, yawrate, thrust)`

## Motion Capture Modeling
The directory motion capture modeling contains all of the files used to characterize the motion capture system, its latency and errors. ***LatencTest.py*** collects the latency measurements and ***Physics_Test.py*** collects position data. Inside the frequency modeling and position modeling subdirectories you will find programs to turn the generated data into images of their probability desnisty functions and fillted gaussian RVs.

## SeniorDesign_Group2
This directory is a copy of the senior design repository which is used to fly the crazyflie 2.x with a cascaded PID controller. This is saved here for posterity.

## Trajectories
This directory contains csv files containing all kinds of metrics about real flights that took place using the MLP and MLP+LSTM controllers
