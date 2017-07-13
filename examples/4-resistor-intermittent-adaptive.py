#!/usr/bin/python3

import sys, math
import matplotlib.pyplot as plt

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
  plt.plot([x / 3600.0 for x in t], U_Batt, 'r-')

  # adaptive: U_thresh = 1.0 V
  #  -> start: T_ON = T_OFF = 50s
  #  -> save:  T_ON = T_OFF = 1s
  load = AdaptiveResistorLoad(1.0,\
                              6.0, 50.0, 50.0,\
                              6.0, 1.0,  1.0)
  t, E_Batt, U_Batt, I_Batt = Simulator(Alkaline(), load).run()
  plt.plot([x / 3600.0 for x in t], U_Batt, 'b-')
  
  plt.show()
  
