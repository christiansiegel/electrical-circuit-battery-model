#!/usr/bin/python3

import sys, math
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from simulator import *


if __name__ == "__main__":

  # Alkaline battery
  # 10 Ohm continuous resistor load

  t_cutoff, E_cutoff = \
    Simulator(Alkaline(), ResistorLoad(10.0)).run(simple=True)
  
  print("t_cutoff = {} h".format(t_cutoff / 3600.0))
  print("E_cutoff = {} Wh".format(E_cutoff / 3600.0))
