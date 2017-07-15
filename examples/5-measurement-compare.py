#!/usr/bin/python3

import sys, math
import matplotlib.pyplot as plt

sys.path.append("../")
from csvhelper import *
from measurement import *
from simulator import *


if __name__ == "__main__":
  # NOTE: unzip "/measurements/*.zip" and run "/measurements/filter.py" first!

  # experiment parameters
  step = 0.1
  R = 24.8
  T_ON = 50.0
  T_OFF = 50.0

  # simulate
  t, E_Batt, U_Batt, _ = \
    Simulator(LiPo(), ResistorLoad(R, T_ON, T_OFF)).run(step, mstep = 10.0)
  
  # load measurement
  t_real, U_Batt_real = \
    CSV.load2("../measurements/24.8ohm_50s_50s_lipo-savgol7.csv", step)
  E_Batt_real = get_energy(t_real, U_Batt_real, R, T_ON, T_OFF)
  
  # plot curves
  plt.xlabel('t [h]')
  plt.ylabel('$U_{Batt}$ [V]')
  plt.plot([x / 3600.0 for x in t], U_Batt, 'b-')
  plt.plot([x / 3600.0 for x in t_real], U_Batt_real, 'r-')
  plt.show()
  
  plt.xlabel('$E_{Batt}$ [Wh]')
  plt.ylabel('$U_{Batt}$ [V]')
  plt.plot([x / 3600.0 for x in E_Batt], U_Batt, 'b-')
  plt.plot([x / 3600.0 for x in E_Batt_real], U_Batt_real, 'r-')
  plt.show()

  # save plot data as csv, e.g. for gnuplot
  CSV.saveX("../plots/compare_24.8ohm_50s_50s_lipo.csv", \
    [t, E_Batt, U_Batt, t_real, E_Batt_real, U_Batt_real])
  
  # print error stats
  stats("24.8ohm_50s_50s_lipo", t_real, t, U_Batt_real, U_Batt, E_Batt_real, E_Batt)
  