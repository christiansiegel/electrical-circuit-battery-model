#!/usr/bin/python3

import sys, math
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from simulator import *


if __name__ == "__main__":
  # Alkaline battery
  # 6 Ohm load

  plt.xlabel('t [h]')
  plt.ylabel('U_Batt [V]')

  # not adaptive
  #  -> always: T_ON = T_OFF = 50s
  load = ResistorLoad(6.0, 50.0, 50.0)
  t, E_Batt, U_Batt, I_Batt = Simulator(Alkaline(), load).run()
  plt.plot(t / 3600.0, U_Batt, 'r-')

  # adaptive: U_thresh = 1.0 V
  #  -> start: T_ON = T_OFF = 50s
  #  -> save:  T_ON = T_OFF = 1s
  load = ResistorLoad(6.0, 50.0, 50.0, \
    U_thresh = 1.0, T_ON_save = 1.0, T_OFF_save = 1.0)
  t, E_Batt, U_Batt, I_Batt = Simulator(Alkaline(), load).run()
  plt.plot(t / 3600.0, U_Batt, 'b-')
  
  plt.show()
  

