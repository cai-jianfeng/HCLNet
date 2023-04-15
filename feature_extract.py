"""
created on:2022/11/13 9:21
@author:caijianfeng
@Purpose: Here is a code implementation of part target decomposition.
          We suggest that professional software is still used to generate the target decomposition automatically,
          and the code generation may have some deviation and incomplete(only 28 target decomposition features, 7 groups)
"""
import numpy as np


class feature:
    def __init__(self):
        pass
    
    def eigen_decomposition(self, T):
        """
        eigenvalue decomposition
        :param T: Coherent matrix
        :return: eigen(three features)
        """
        eig, eig_v = np.linalg.eig(T)
        return np.abs(eig)
    
    def H_aerfa_Ani_decomposition(self, T):
        """
        H/α/Ani decomposition
        :param T: Coherent matrix
        :return: H/α/Ani(three features)
        """
        eig, eig_v = np.linalg.eig(T)
        # print('eig:', eig)
        p = eig / np.sum(eig)
        H = -1 * np.sum(p * np.log(p) / np.log(3))
        aerfa = np.sum(p * np.cos(eig_v[0, :]))
        Ani = (eig[1] - eig[2]) / (eig[1] + eig[2])
        return np.abs(H), np.abs(aerfa), np.abs(Ani)
    
    def RF_decomposition(self, T):
        """
        RF decomposition
        :param T: Coherent matrix
        :return: RF(six features)
        """
        SPAN = T[0, 0] + T[1, 1] + T[2, 2]
        RF = np.array([
            10 * np.log10(SPAN),
            T[1, 1] / SPAN,
            T[2, 2] / SPAN,
            np.abs(T[0, 1]) / np.sqrt(T[0, 0] * T[1, 1]),
            np.abs(T[0, 2]) / np.sqrt(T[0, 0] * T[2, 2]),
            np.abs(T[1, 2]) / np.sqrt(T[1, 1] * T[2, 2])
        ])
        return np.abs(RF)
    
    def Freeman_decomposition(self, T):
        """
        Freeman decomposition
        :param T: Coherent matrix
        :return: Pv/Pd/Ps(three features)
        """
        Pv = 4 * T[2, 2]
        x11 = T[0, 0] - 2 * T[2, 2]
        x22 = T[1, 1] - T[2, 2]
        if x11 > x22:
            aerfa = 0
            beta = T[0, 1] / x11
            Ps = x11 + np.square(np.abs(T[0, 1])) / x11
            Pd = x22 - np.square(np.abs(T[0, 1])) / x11
        else:
            beta = 0
            aerfa = T[0, 1] / x22
            Pd = x22 + np.square(np.abs(T[0, 1])) / x22
            Ps = x11 - np.square(np.abs(T[0, 1])) / x22
        return np.abs(Pv), np.abs(Pd), np.abs(Ps)
    
    def Holm_decomposition(self, T):
        """
        Holm decomposition
        :param T: Coherent matrix
        :return: Ps/Pd/Pr(three features)
        """
        eig, eig_v = np.linalg.eig(T)
        p = eig / np.sum(eig)
        Ps = p[0] - p[1]
        Pd = 2 * (p[1] - p[2])
        Pr = 3 * p[2]
        return np.abs(Ps), np.abs(Pd), np.abs(Pr)
    
    def krogager_decomposition(self, T):
        """
        krogager decomposition
        :param T: Coherent matrix
        :return: Srl/Srr/Sll(three features)
        """
        Srl = np.sqrt(T[0, 0] / 2)
        Srr = np.sqrt(0.5 * T[2, 2] + 0.5j * T[2, 1] - 0.5j * T[1, 2] + 0.5 * T[1, 1])
        Sll = np.sqrt(0.5 * T[2, 2] - 0.5j * T[2, 1] + 0.5j * T[1, 2] + 0.5 * T[1, 1])
        
        return np.abs(Srl), np.abs(Srr), np.abs(Sll)
    
    def seven_component_scattering_power_decomposition(self, T):
        """
        seven component scattering power decomposition
        :param T: Coherent matrix
        :return: Ph/Pmd/Pcd/Pod/Ps/Pd/Pv(seven faetures)
        """
        Ph = 2 * np.abs(T[1, 2].imag)
        Pmd = 2 * np.abs(T[1, 2].real)
        Pod = 2 * np.abs(T[0, 2].real)
        Pcd = 2 * np.abs(T[0, 2].imag)
        Pv = 2 * T[2, 2] - Ph - Pmd - Pod - Pcd
        C1 = T[0, 0] - T[1, 1] - 7 * T[2, 2] / 8 + (Pmd + Ph) / 16 - 15 * (Pcd + Pod) / 16
        if C1 > 0:
            if Pv < 0:
                Ph = Pod = Pcd = Pmd = 0
                Pv = 2 * T[2, 2]
            dB = 10 * np.log((T[0, 0] + T[1, 1] - 2 * T[0, 1].real) / (T[0, 0] + T[1, 1] + 2 * T[0, 1].real))
            if dB > 2:
                Pv = 15 * Pv / 8
                S = T[0, 0] - Pv / 2 - Pod / 2 - Pcd / 2
                D = T[1, 1] - 7 * Pv / 30 - Ph / 2 - Pmd / 2
                C = T[0, 1] + Pv / 6
            elif dB < -2:
                Pv = 2 * Pv
                S = T[0, 0] - Pv / 2 - Pod / 2 - Pcd / 2
                D = T[1, 1] - Pv / 4 - Ph / 2 - Pmd / 2
                C = T[0, 1]
            else:
                Pv = 15 * Pv / 8
                S = T[0, 0] - Pv / 2 - Pod / 2 - Pcd / 2
                D = T[1, 1] - 7 * Pv / 30 - Ph / 2 - Pmd / 2
                C = T[0, 1] - Pv / 6
            C0 = 2 * T[0, 0] + Ph + Pmd
            if C0 > 0:
                Ps = S + np.square(np.abs(C)) / S
                Pd = D - np.square(np.abs(C)) / S
            else:
                Ps = S - np.square(np.abs(C)) / D
                Pd = D + np.square(np.abs(C)) / D
        else:
            if Pv < 0:
                Pv = 0
                Pmw = Pmd + Ph
                Pcw = Pcd + Pod
                if Pmw > Pcw:
                    Pcw = 2 * T[2, 2] - Pmw
                    Pcw = Pcw if Pcw > 0 else 0
                    if Pod > Pcd:
                        Pcd = Pcw - Pod
                        Pcd = Pcd if Pcd > 0 else 0
                    else:
                        Pod = Pcw - Pcd
                        Pod = Pod if Pod > 0 else 0
                else:
                    Pmw = 2 * T[2, 2] - Pcw
                    Pmw = Pmw if Pmw > 0 else 0
                    if Pmd > Ph:
                        Ph = Pmw - Pmd
                        Ph = Ph if Ph > 0 else 0
                    else:
                        Pmd = Pmw - Ph
                        Pmd = Pmd if Pmd > 0 else 0
            Pv = 15 * Pv / 16
            S = T[0, 0] - Pod / 2 - Pcd / 2
            D = T[1, 1] - 7 * Pv / 15 - Ph / 2 - Pmd / 2
            C = T[0, 1]
            Ps = S - np.square(np.abs(C)) / D
            Pd = D + np.square(np.abs(C)) / D
        return np.abs(Ph), np.abs(Pmd), np.abs(Pcd), np.abs(Pod), np.abs(Ps), np.abs(Pd), np.abs(Pv)
