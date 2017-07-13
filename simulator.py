import sys

from batteries import *
from loads import *


inf = float('inf')


class Simulator:
  def __init__(self, model, load):
    self._model = model
    self._load = load


  def _printSOC(self, SOC):
    if abs(self._lastPrintedSOC - SOC) >= 0.01:
      self._lastPrintedSOC = SOC
      sys.stdout.write("SOC = {}%   \r".format(int(SOC * 100)))
      sys.stdout.flush()
  

  def run(self, step = 1.0, **kwargs): 
    simple = kwargs['simple'] if 'simple' in kwargs else False
    arraystep = kwargs['astep'] if 'astep' in kwargs else step
    modelstep = kwargs['mstep'] if 'mstep' in kwargs else step

    # reset model
    self._load.reset()
    C_nominal = self._model.getC_nominal() * 3600.0 # [mAs]
    
    # init result arrays
    if not simple:
      t_list = list()
      E_Batt_list = list()
      U_Batt_list = list()
      I_Batt_list = list()
    
    # init variables
    C_Batt = C_nominal
    t = 0
    E_Batt= 0
    U_Batt, _, _, _, _, _ = self._model.getCircuitParams(1.0)
    Q_CTS = Q_CTL = 0
    U_TS = U_TL = 0
    SOC = 1
    
    cutoff = self._model.getU_cutoff()
    
    self._lastPrintedSOC = 0
    while True:
      # update model parameters
      # (can be done every 'modelstep' seconds only to speed up simulation)
      if t % modelstep <= step:
        U_Eq, R_S, R_TS, C_TS, R_TL, C_TL = self._model.getCircuitParams(SOC)

      if self._load.isDischarge(t, U_Batt):
        # get load current at time t
        I_Batt = self._load.calcCurrent(U_Batt)
      
        # minimize effect of calculating the current at time t with U_Batt at 
        # time t-1 (and hence a potentially still too high ohmic overpotential)
        I_Batt = self._load.calcCurrent(U_Eq - R_S * I_Batt - U_TS - U_TL) 
      else:
        I_Batt = 0.0
      
      # update transient response RC circuit
      I_RTS = U_TS / R_TS
      I_CTS = I_Batt - I_RTS
      Q_CTS += I_CTS * step
      U_TS = Q_CTS / C_TS
      
      I_RTL = U_TL / R_TL
      I_CTL = I_Batt - I_RTL
      Q_CTL += I_CTL * step
      U_TL = Q_CTL / C_TL
  
      # new U_Batt
      U_Batt = U_Eq - R_S * I_Batt - U_TS - U_TL
      assert U_Batt < 5, "U_Batt drifts off"

      # update used energy and remaining capacity
      P_Batt = U_Batt * I_Batt
      E_Batt += P_Batt * step
      C_Batt -= (I_Batt * 1000.0) * step # [mAs] 
      
      # update SOC
      SOC = C_Batt / C_nominal 
      self._printSOC(SOC) 

      # log simulation data in array
      if (not simple) and (t % arraystep <= step):       
        t_list.append(t)
        E_Batt_list.append(E_Batt)
        U_Batt_list.append(U_Batt)
        I_Batt_list.append(I_Batt)

      # stop criteria
      if U_Batt < cutoff: 
        print("\nU_Batt < U_cutoff\n")
        break
    
      # update time
      t += step 
    
    if not simple:   
      return t_list, E_Batt_list, U_Batt_list, I_Batt_list
    else:
      return t, E_Batt 
