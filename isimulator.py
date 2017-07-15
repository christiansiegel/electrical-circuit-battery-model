from ipywidgets import FloatProgress, VBox, HTML
from IPython.display import display

from simulator import *


class ISimulator(Simulator):  
  def __init__(self, model, load):
    self._progressBar = FloatProgress(min=0, max=100)
    self._progressLabel = HTML()
    display(VBox([self._progressLabel, self._progressBar]))
    super(ISimulator,self).__init__(model, load)

  def _printSOC(self, SOC):
    if abs(self._lastPrintedSOC - SOC) >= 0.01:
      self._lastPrintedSOC = SOC
      self._progressBar.value = SOC * 100
      self._progressLabel.value = "SOC = {}%".format(int(SOC*100))

  def _printStopReason(self, s):
    self._progressLabel.value += " ... " + s