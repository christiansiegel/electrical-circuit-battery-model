import math
import numpy as np
from abc import ABCMeta, abstractmethod


class BatteryLoad:
  __metaclass__ = ABCMeta
  
  def __init__(self, param, T_ON = float('inf'), T_OFF = 0, **kwargs):
    assert T_ON > 0
    
    self._param = float(param)
    self._T_ON = T_ON
    self._T_OFF = T_OFF
    
    self._adaptive = "U_thresh" in kwargs
    if self._adaptive:
      self._U_thresh = kwargs['U_thresh']
      self._param_save = float(kwargs['param_save']) if 'param_save' in kwargs else param
      self._T_ON_save = kwargs['T_ON_save'] if 'T_ON_save' in kwargs else T_ON
      self._T_OFF_save = kwargs['T_OFF_save'] if 'T_OFF_save' in kwargs else T_OFF
    
    self.reset()
  
  def reset(self):
    np.random.seed(0)
    self._next = self._T_ON
    self._discharge = True
    self._saving = False
    self._t_thresh = float('inf')
    
  def t_thresh(self):
    return self._t_thresh
  
  @abstractmethod
  def calcCurrent(self, U_Batt, param): pass 
  
  def getCurrent(self, t, U_Batt):
    U_Batt = float(U_Batt)
    if self._adaptive and (not self._saving) and U_Batt < self._U_thresh:
      self._saving = True
      self._t_thresh = t
      
    if t >= self._next:
      if self._discharge: 
        self._next += self._T_OFF if not self._saving else self._T_OFF_save
      else: 
        self._next += self._T_ON if not self._saving else self._T_ON_save
      self._discharge = not self._discharge
      
    if self._discharge:  
      if self._saving:
        return self.calcCurrent(U_Batt, self._param_save)
      else:
        return self.calcCurrent(U_Batt, self._param)
    else:
      return 0.0 

class ResistorLoad(BatteryLoad):
  def calcCurrent(self, U_Batt, param):
    return U_Batt / param


class ConstantCurrentLoad(BatteryLoad):
  def calcCurrent(self, U_Batt, param):
    return param
 
   
class ConstantPowerLoad(BatteryLoad):
  def calcCurrent(self, U_Batt, param):
    return P / param



