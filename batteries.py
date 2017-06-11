import math
import numpy as np
from abc import ABCMeta, abstractmethod


class Battery:
  __metaclass__ = ABCMeta
    
  @abstractmethod
  def getCircuitParams(self, SOC): pass
    
  @abstractmethod
  def getC_nominal(self): pass

  @abstractmethod
  def getU_cutoff(self): pass
    
  def expfunc(self, SOC, p):
    p = np.array(p)
    p = np.pad(p, (0,10 - p.shape[0]), 'constant', constant_values=(0,0))
    return p[0]*np.exp(-p[1]*SOC) + p[2] + p[3]*SOC - p[4]*np.power(SOC,2) + p[5]*np.power(SOC,3) + p[6]*np.power(SOC,4) + p[7]*np.power(SOC,5) + p[8]*np.power(SOC,6) + p[9]*np.power(SOC,7)


class Alkaline(Battery):
  def getC_nominal(self): return 1200.0
  def getU_cutoff(self): return 0.9
  def getCircuitParams(self, SOC):
    U_Eq = np.poly1d([1033.96676322, -4746.43834753, 9247.09175351, -9971.36083234, 6513.8071321, -2656.08285339, 674.632903135, -104.757754367, 10.1022340299, 0.643372799281])(SOC)
    R_S = self.expfunc(SOC, [6.91569502522, 20.5082342043, 0.294603148545, 1.96285741923, 4.51144319875, 2.59828787972])
    R_TS = self.expfunc(SOC, [54.815818916, 27.1305255071, 0.524851096019, 0.0290779831529, 0.544157939065, 0.234658197106])
    C_TS = self.expfunc(SOC, [5.44277350399e-14, -37.986640267, 25.5861579495, -282.089172291, -1602.36745758, -1296.84287667])
    R_TL = self.expfunc(SOC, [33.3588395401, 28.4297082587, 1.10355945353, -1.15157164305, -2.67575723621, -2.22205866931])
    C_TL = self.expfunc(SOC, [9.12443473416e-19, -50.8507647222, -530.206263913, 13715.6388842, 24463.7942431, 15004.7007411])
    return U_Eq, R_S, R_TS, C_TS, R_TL, C_TL


class LiPo(Battery):
  def getC_nominal(self): return 2173.90522318
  def getU_cutoff(self): return 3.0
  def getCircuitParams(self, SOC):
    U_Eq = np.poly1d([-3933.96387239, 21731.2354339, -51866.0902379, 69939.6015871, -58467.4834242, 31276.0961581, -10667.5175921, 2244.44160736, -272.747658209, 17.6294222044, 3.00169064379])(SOC)
    R_S = self.expfunc(SOC, [0.0540589195231, 11.6684054652, 0.162793162179])
    R_TS = self.expfunc(SOC, [1040.61321502, 0.397860524947, -1040.49903925, 413.248303586, 79.9520561627, 8.1879700004])
    C_TS = self.expfunc(SOC, [2.73615654563e-19, -51.0674811386, 73.7133219771, 11243.4082474, 24182.2187033, 14231.9099271])
    R_TL = self.expfunc(SOC, [1702.55657549, 0.444019053071, -1702.42142482, 754.692451645, 163.185091915, 18.8131173849])
    C_TL = self.expfunc(SOC, [-121529067.387, -0.487845984711, 121554250.909, 59544944.7595, -13560388.5669, 3326885.09937])
    return U_Eq, R_S, R_TS, C_TS, R_TL, C_TL







