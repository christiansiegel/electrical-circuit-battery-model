#!/usr/bin/python3

import sys, math
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from simulator import *


if __name__ == "__main__":

  # Li-Ion battery
  # 1 Watt intermittent constant power load (T_ON = T_OFF = 1h)

  t, E_Batt, U_Batt, I_Batt = \
    Simulator(LiPo(), ConstantPowerLoad(1.0, 3600.0, 3600.0)).run()
  
  plt.xlabel('t [h]')
  plt.ylabel('U_Batt [V]')
  plt.plot(t / 3600.0, U_Batt)
  plt.show()
  
  plt.xlabel('E_Batt [Wh]')
  plt.ylabel('U_Batt [V]')
  plt.plot(E_Batt / 3600.0, U_Batt)
  plt.show()
  
  plt.xlabel('t [h]')
  plt.ylabel('I_Batt [mA]')
  plt.plot(t / 3600.0, I_Batt * 1000.0)
  plt.show()
