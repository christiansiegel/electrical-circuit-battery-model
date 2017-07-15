from abc import ABCMeta, abstractmethod


class BatteryLoad:
  __metaclass__ = ABCMeta
  
  def __init__(self, param, T_ON = float('inf'), T_OFF = 0):
    assert T_ON > 0
    self._param = float(param)
    self._T_ON = T_ON if T_OFF > 0 else float('inf')
    self._T_OFF = T_OFF
    self.reset()

  def reset(self):
    self._next = self._T_ON
    self._discharge = True

  def isDischarge(self, t, U_Batt, U_Eq, SOC):   
    if t >= self._next:
      if self._discharge: 
        self._next += self._T_OFF
      else: 
        self._next += self._T_ON
      self._discharge = not self._discharge
    return self._discharge

  @abstractmethod
  def calcCurrent(self, U_Batt): pass 
  

class ResistorLoad(BatteryLoad):
  def calcCurrent(self, U_Batt):
    return U_Batt / self._param


class ConstantCurrentLoad(BatteryLoad):
  def calcCurrent(self, U_Batt):
    return self._param
 
   
class ConstantPowerLoad(BatteryLoad):
  def calcCurrent(self, U_Batt):
    return self._param / U_Batt




class AdaptiveBatteryLoad(BatteryLoad):
  __metaclass__ = ABCMeta
  
  def __init__(self, U_thresh, param, T_ON, T_OFF, param_save, T_ON_save, T_OFF_save):
    self._param_orig = param
    self._T_ON_orig = T_ON if T_OFF > 0 else float('inf')
    self._T_OFF_orig = T_OFF
    self._U_thresh = U_thresh
    self._param_save = param_save
    self._T_ON_save = T_ON_save if T_OFF_save > 0 else float('inf')
    self._T_OFF_save = T_OFF_save
    super(AdaptiveBatteryLoad,self).__init__(param, T_ON, T_OFF)

  def reset(self):
    super(AdaptiveBatteryLoad,self).reset()
    self._param = self._param_orig
    self._T_ON = self._T_ON_orig
    self._T_OFF = self._T_OFF_orig
    self._saving = False
    self._t_thresh = float('inf')
    
  def t_thresh(self):
    return self._t_thresh    

  def isDischarge(self, t, U_Batt, U_Eq, SOC):  
    if (not self._saving) and (U_Batt < self._U_thresh):
      self._saving = True
      self._param = self._param_save
      self._T_ON = self._T_ON_save
      self._T_OFF = self._T_OFF_save
      self._t_thresh = t
   
    if t >= self._next:
      if self._discharge: 
        self._next += self._T_OFF
      else: 
        self._next += self._T_ON
      self._discharge = not self._discharge
    return self._discharge


class AdaptiveResistorLoad(AdaptiveBatteryLoad):
  def calcCurrent(self, U_Batt):
    return U_Batt / self._param


class AdaptiveConstantCurrentLoad(AdaptiveBatteryLoad):
  def calcCurrent(self, U_Batt):
    return self._param
 
   
class AdaptiveConstantPowerLoad(AdaptiveBatteryLoad):
  def calcCurrent(self, U_Batt):
    return self._param / U_Batt
