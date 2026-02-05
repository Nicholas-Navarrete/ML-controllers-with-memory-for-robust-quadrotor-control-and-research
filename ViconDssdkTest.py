from vicon_dssdk import ViconDataStream
import keyboard
import time 
import numpy as np
host_name = '128.101.167.111'

client = ViconDataStream.Client()
client.Connect(host_name)
client.EnableSegmentData()
Times=[]

while keyboard.is_pressed("esc") == False:
    Times.append(time.perf_counter())
    client.GetFrame()
    output = client.GetSegmentGlobalTranslation("Drone4", "Drone4")
    Rotations = client.GetSegmentGlobalRotationEulerXYZ("Drone4", "Drone4")
    print(f"Position: {output[0][0]:.1f}, {output[0][1]:.1f}, {output[0][2]:.1f} | Rotation: {Rotations[0][0]:.1f}, {Rotations[0][1]:.1f}, {Rotations[0][2]:.1f}", end='\r')

Frequencies = np.diff(Times)          # time differences
Frequencies = 1 / Frequencies       # instantaneous frequencies
print("Average Control Frequency: ", np.mean(Frequencies), "Hz")