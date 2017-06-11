#!/usr/bin/python3

import sys, math
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from simulator import *


if __name__ == "__main__":

  # Alkaline battery
  # 10 Ohm continuous resistor load

  t, E_Batt, U_Batt, I_Batt = \
    Simulator(Alkaline(), ResistorLoad(10.0)).run()
  
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
