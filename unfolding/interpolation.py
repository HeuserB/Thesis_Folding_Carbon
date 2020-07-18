import numpy as np
from functions_folding import load_h5

def interpolated_values(r, halogen_type=0, file_F = "../data/halogen_rotation/b3lyp_angle_45.0_F_6-311++g**.h5", file_Cl = "../data/halogen_rotation//b3lyp_angle_45.0_Cl_6-311++g**.h5"):
    ### returns the halogen_angle, halogen_distance, H_angle, and H_distance for a given C-C distance 
    d_CC, E_init, E_final, CCF, CCH, d_HCs, d_FCs, basis_set, functional, halogen, angle = load_h5(file_F)
    d_CC_Cl, E_init_Cl, E_final_Cl, CCF_Cl, CCH_Cl, d_HCs_Cl, d_FCs_Cl, basis_set_Cl, functional_Cl, halogen_Cl, angle_Cl = load_h5(file_Cl)
    r_mask = np.where(d_CC >= 1.60)
    d_CC, CCF, CCH, d_HCs, d_FCs = d_CC[r_mask], CCF[r_mask], CCH[r_mask], d_HCs[r_mask], d_FCs[r_mask]
    r_mask_Cl = np.where(d_CC_Cl >= 2.0)
    d_CC_Cl, CCF_Cl, CCH_Cl, d_HCs_Cl, d_FCs_Cl = d_CC_Cl[r_mask_Cl], CCF_Cl[r_mask_Cl], CCH_Cl[r_mask_Cl], d_HCs_Cl[r_mask_Cl], d_FCs_Cl[r_mask_Cl]
    if halogen_type == 0:
        angle_halogen = np.interp(r,np.hstack([d_CC,d_CC[-1]+0.01]),np.hstack([CCF,0.]))
        distance_halogen = np.interp(r,d_CC,d_FCs) 
        angle_H = np.interp(r,np.hstack([d_CC,d_CC[-1]+0.01]),np.hstack([CCH,0.]))
        distance_H = np.interp(r,d_CC,d_HCs)
    elif halogen_type == 1:
        angle_halogen = np.interp(r,np.hstack([d_CC_Cl,d_CC_Cl[-1]+0.01]),np.hstack([CCF_Cl,0.]))
        distance_halogen = np.interp(r,d_CC_Cl,d_FCs_Cl) 
        angle_H = np.interp(r,np.hstack([d_CC_Cl,d_CC_Cl[-1]+0.01]),np.hstack([CCH_Cl,0.]))
        distance_H = np.interp(r,d_CC_Cl,d_HCs_Cl)
    else:
        print("Please enter a valid halogen (0/1)")
    return np.deg2rad(angle_halogen), distance_halogen, np.deg2rad(angle_H), distance_H
