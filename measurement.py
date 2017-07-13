import sys, math


# Expects time in s; return Ws
def get_energy(t_list, U_Batt_list, R, T_ON, T_OFF = 0):
  E_Batt_list = list()

  t_last = 0.0
  E_Batt = 0.0
  for i in range(0, len(t_list)):
    t = t_list[i]
    U_Batt = U_Batt_list[i]
  
    if T_OFF == 0:
      discharge = True
    else:
      t_period = t % (T_ON + T_OFF)
      discharge = t_period <= T_ON
  
    if discharge:
      P_Batt = (U_Batt * U_Batt) / R
      E_Batt += P_Batt * (t - t_last)
    
    E_Batt_list.append(E_Batt)
    t_last = t
  
  return E_Batt_list


# Expects time in s; return W
def get_power(t_list, U_Batt_list, R, T_ON, T_OFF = 0):
  P_Batt_list = list()

  t_last = 0.0
  E_Batt = 0.0
  for i in range(0, len(t_list)):
    t = t_list[i]
    U_Batt = U_Batt_list[i]
  
    if T_OFF == 0:
      discharge = True
    else:
      t_period = t % (T_ON + T_OFF)
      discharge = t_period <= T_ON
  
    if discharge:
      P_Batt = (U_Batt * U_Batt) / R
    
    P_Batt_list.append(P_Batt)
    t_last = t
  
  return P_Batt_list
  

def mean_absolute_error(v_true, v_pred): 
  len_pred = len(v_pred)
  len_true = len(v_true)
  if len_pred < len_true:
    v_pred += [v_pred[-1]] * (len_true - len_pred) # pad with edge value
  elif len_pred > len_true:
    v_pred = v_pred[:len_true]
  diff = [abs(x-y) for x, y in zip(v_true, v_pred)]
  return sum(diff) / float(len(diff))
  
  
def signed_percentage_error(v_true, v_pred): 
  return (float(v_pred - v_true) / float(v_true)) * 100.0
    
    
def stats(label, t_true, t_pred, U_true, U_pred, E_true = None, E_pred = None):
    print("\nStats %s:" % label)
    
    # t_cutoff error
    t_cutoff_true = t_true[-1]
    t_cutoff_pred = t_pred[-1]
    Error_t_cutoff = signed_percentage_error(t_cutoff_true, t_cutoff_pred)
    
    print("  t_cutoff measurement: {} h".format(t_cutoff_true / 3600.0))
    print("  t_cutoff simulation:  {} h".format(t_cutoff_pred / 3600.0))
    print("   -> Error = {} %".format(Error_t_cutoff))
    
    
    # E_cutoff error
    if E_true is not None and E_pred is not None:
      E_cutoff_true = E_true[-1]
      E_cutoff_pred = E_pred[-1]
      Error_E_cutoff = signed_percentage_error(E_cutoff_true, E_cutoff_pred)
    
      print("  E_cutoff measurement: {} Wh".format(E_cutoff_true / 3600.0))
      print("  E_cutoff simulation:  {} Wh".format(E_cutoff_pred / 3600.0))
      print("   -> Error = {} %".format(Error_E_cutoff))
    
    # MAE_U
    MAE_U = mean_absolute_error(U_true, U_pred)
    print("  MAE_U: {} V".format(MAE_U))  
