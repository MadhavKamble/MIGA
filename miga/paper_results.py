"""
Hardcoded paper results from Figueroa-García et al. (2023).

Sources:
  Table 1  — Biggest differences between available and complete datasets (p. 960)
  Table 4  — RMSE of benchmark datasets (p. 962)
  Table 5  — MAD  of benchmark datasets (p. 962)
  Table 6  — CoD  of benchmark datasets (p. 963)
  Section 4 — Application example fitness decomposition (p. 954–955)
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Section 4: Application example (Dataset 1, 311 × 6)
# ------------------------------------------------------------------

# Fitness function decomposition (r=∞) for each method
SECTION4_FITNESS = {
    "MIGA": {
        "F_inf":   0.2110,
        "d_means": 0.0163,
        "d_cov":   0.0796,
        "d_skew":  0.1151,
    },
    "EM": {
        "F_inf":   1.3862,
        "d_means": 0.0883,
        "d_cov":   0.5228,
        "d_skew":  0.7751,
    },
    "Aux. Regression": {
        "F_inf":   0.9117,
        "d_means": 0.1545,
        "d_cov":   0.2711,
        "d_skew":  0.4861,
    },
}

# Table 1: Biggest absolute differences per method
SECTION4_TABLE1 = {
    # columns: max|x̄_jA - x̄_jC|, max|s²_jA - s²_jC|, max|r_jA - r_jC|, max|b_jA - b_jC|
    "MIGA":            (0.301,  0.751, 0.025, 0.116),
    "EM algorithm":    (0.560, 31.952, 0.165, 0.775),
    "Aux. regression": (0.723, 17.803, 0.189, 0.486),
}

# ------------------------------------------------------------------
# Table 3: MIGA parameters per benchmark dataset
# ------------------------------------------------------------------

TABLE3_PARAMS = {
    #             c   c1  c2  c3    l      G     r
    "Iris":      dict(c=3, c1=3, c2=2, c3=5,  l=100, G=1000, r=2),
    "Wine":      dict(c=3, c1=3, c2=2, c3=5,  l=100, G=1000, r=1),
    "Glass":     dict(c=3, c1=3, c2=3, c3=5,  l=100, G=1000, r=2),
    "Haberman":  dict(c=3, c1=3, c2=3, c3=10, l=100, G=1000, r=float("inf")),
    "Wholesale": dict(c=3, c1=3, c2=3, c3=10, l=100, G=1000, r=float("inf")),
    "Cardio":    dict(c=5, c1=5, c2=5, c3=10, l=200, G=2000, r=float("inf")),
    "Adult":     dict(c=5, c1=5, c2=5, c3=10, l=200, G=2000, r=float("inf")),
}
# All benchmark runs use Q=12
BENCHMARK_Q = 12

# ------------------------------------------------------------------
# Table 4: RMSE (lower is better; bold = best method per row)
# ------------------------------------------------------------------

TABLE4_RMSE = {
    #   dataset   pct   MIGA    CMIM    ANNI    GPNNI   IARI    FCMI    DMI     k-NNI   Mean
    "Iris": {
        30: dict(MIGA=0.0987, CMIM=0.1273, ANNI=0.1319, GPNNI=0.1328, IARI=0.1424, FCMI=0.1912, DMI=0.144,  kNNI=0.1621, Mean=0.2994),
        40: dict(MIGA=0.0998, CMIM=0.148,  ANNI=0.1564, GPNNI=0.1638, IARI=0.2198, FCMI=0.1654, DMI=0.1854, kNNI=0.3457, Mean=None),
        50: dict(MIGA=0.1302, CMIM=0.1655, ANNI=0.1774, GPNNI=0.1821, IARI=0.2554, FCMI=0.1872, DMI=0.2098, kNNI=0.3801, Mean=None),
        60: dict(MIGA=0.1668, CMIM=0.186,  ANNI=0.198,  GPNNI=0.2018, IARI=0.2922, FCMI=0.2034, DMI=0.2355, kNNI=0.4119, Mean=None),
    },
    "Wine": {
        30: dict(MIGA=0.0971, CMIM=0.1004, ANNI=0.1131, GPNNI=0.1227, IARI=0.1242, FCMI=0.1404, DMI=0.1278, kNNI=0.124,  Mean=0.1645),
        40: dict(MIGA=0.0997, CMIM=0.1169, ANNI=0.1251, GPNNI=0.1401, IARI=0.1446, FCMI=0.1632, DMI=0.1517, kNNI=0.1441, Mean=0.1866),
        50: dict(MIGA=0.1332, CMIM=0.1384, ANNI=0.147,  GPNNI=0.1635, IARI=0.1723, FCMI=0.184,  DMI=0.1734, kNNI=0.1644, Mean=0.218),
        60: dict(MIGA=0.1492, CMIM=0.1529, ANNI=0.1611, GPNNI=0.1751, IARI=0.1888, FCMI=0.2064, DMI=0.1893, kNNI=0.1963, Mean=0.2406),
    },
    "Glass": {
        30: dict(MIGA=0.0878, CMIM=0.1486, ANNI=0.1344, GPNNI=0.1572, IARI=0.1415, FCMI=0.2284, DMI=0.1561, kNNI=0.1818, Mean=0.2343),
        40: dict(MIGA=0.1003, CMIM=0.1348, ANNI=0.1344, GPNNI=0.1572, IARI=0.1415, FCMI=0.2284, DMI=0.1561, kNNI=0.1818, Mean=0.2343),
        50: dict(MIGA=0.1391, CMIM=0.1432, ANNI=0.1556, GPNNI=0.1749, IARI=0.1594, FCMI=0.2468, DMI=0.1728, kNNI=0.2055, Mean=0.2851),
        60: dict(MIGA=0.1461, CMIM=0.1449, ANNI=0.1618, GPNNI=0.1977, IARI=0.1756, FCMI=0.269,  DMI=0.1964, kNNI=0.2195, Mean=0.2851),
    },
    "Haberman": {
        30: dict(MIGA=0.2121, CMIM=0.3233, ANNI=0.355,  GPNNI=0.3572, IARI=0.3634, FCMI=0.3413, DMI=0.3737, kNNI=0.4116, Mean=0.3359),
        40: dict(MIGA=0.2567, CMIM=0.374,  ANNI=0.4078, GPNNI=0.4095, IARI=0.4237, FCMI=0.4044, DMI=0.4267, kNNI=0.4622, Mean=0.3959),
        50: dict(MIGA=0.3301, CMIM=0.4306, ANNI=0.4483, GPNNI=0.4549, IARI=0.4766, FCMI=0.4416, DMI=0.4852, kNNI=0.5164, Mean=0.4473),
        60: dict(MIGA=0.3761, CMIM=0.4774, ANNI=0.5034, GPNNI=0.5067, IARI=0.5194, FCMI=0.495,  DMI=0.5213, kNNI=0.5577, Mean=0.4992),
    },
    "Wholesale": {
        30: dict(MIGA=0.1176, CMIM=0.1368, ANNI=0.1764, GPNNI=0.186,  IARI=0.161,  FCMI=0.2098, DMI=0.171,  kNNI=0.1662, Mean=0.2003),
        40: dict(MIGA=0.1231, CMIM=0.1603, ANNI=0.2084, GPNNI=0.2239, IARI=0.1918, FCMI=0.2592, DMI=0.2011, kNNI=0.1979, Mean=0.2641),
        50: dict(MIGA=0.1671, CMIM=0.1918, ANNI=0.2438, GPNNI=0.2564, IARI=0.2258, FCMI=0.2807, DMI=0.248,  kNNI=0.2282, Mean=0.2641),
        60: dict(MIGA=0.2001, CMIM=0.2157, ANNI=0.2641, GPNNI=0.2732, IARI=0.2528, FCMI=0.3211, DMI=0.2645, kNNI=0.2545, Mean=0.2993),
    },
    "Cardio": {
        30: dict(MIGA=0.0521, CMIM=0.0493, ANNI=0.0608, GPNNI=0.0605, IARI=0.0507, FCMI=0.1193, DMI=0.0644, kNNI=0.0657, Mean=0.1207),
        40: dict(MIGA=0.0554, CMIM=0.0554, ANNI=0.0714, GPNNI=0.0708, IARI=0.0638, FCMI=0.1358, DMI=0.0677, kNNI=0.0753, Mean=0.1373),
        50: dict(MIGA=0.0611, CMIM=0.0593, ANNI=0.0714, GPNNI=0.0708, IARI=0.0638, FCMI=0.1532, DMI=0.0761, kNNI=0.0878, Mean=0.1547),
        60: dict(MIGA=0.0643, CMIM=0.0693, ANNI=0.0767, GPNNI=0.0785, IARI=0.0728, FCMI=0.1698, DMI=0.0828, kNNI=0.0981, Mean=0.1712),
    },
    "Adult": {
        30: dict(MIGA=0.0982, CMIM=0.1424, ANNI=0.145,  GPNNI=0.1538, IARI=0.152,  FCMI=0.1641, DMI=0.1615, kNNI=0.155,  Mean=0.1851),
        40: dict(MIGA=0.1087, CMIM=0.171,  ANNI=0.1739, GPNNI=0.181,  IARI=0.1777, FCMI=0.1928, DMI=0.1864, kNNI=0.1817, Mean=0.2128),
        50: dict(MIGA=0.1786, CMIM=0.1942, ANNI=0.1981, GPNNI=0.2034, IARI=0.1992, FCMI=0.2164, DMI=0.2089, kNNI=0.2044, Mean=0.2374),
        60: dict(MIGA=0.1907, CMIM=0.2158, ANNI=0.213,  GPNNI=0.2206, IARI=0.2148, FCMI=0.2328, DMI=0.2271, kNNI=0.2211, Mean=0.2648),
    },
}

# ------------------------------------------------------------------
# Table 5: MAD (lower is better)
# ------------------------------------------------------------------

TABLE5_MAD = {
    "Iris": {
        30: dict(MIGA=0.0187, CMIM=0.0274, ANNI=0.029,  GPNNI=0.03,   IARI=0.0304, FCMI=0.0468, DMI=0.0325, kNNI=0.0349, Mean=0.0732),
        40: dict(MIGA=0.0211, CMIM=0.0361, ANNI=0.0385, GPNNI=0.039,  IARI=0.0397, FCMI=0.0615, DMI=0.0437, kNNI=0.0452, Mean=0.0972),
        50: dict(MIGA=0.0327, CMIM=0.0456, ANNI=0.0479, GPNNI=0.0483, IARI=0.0499, FCMI=0.0805, DMI=0.0513, kNNI=0.0562, Mean=0.1182),
        60: dict(MIGA=0.0464, CMIM=0.0566, ANNI=0.0588, GPNNI=0.0592, IARI=0.0604, FCMI=0.1008, DMI=0.0672, kNNI=0.0704, Mean=0.1399),
    },
    "Wine": {
        30: dict(MIGA=0.0091, CMIM=0.013,  ANNI=0.0146, GPNNI=0.0147, IARI=0.0149, FCMI=0.0189, DMI=0.0149, kNNI=0.0156, Mean=0.0225),
        40: dict(MIGA=0.0101, CMIM=0.0175, ANNI=0.0189, GPNNI=0.0233, IARI=0.0193, FCMI=0.0251, DMI=0.0215, kNNI=0.0207, Mean=0.0293),
        50: dict(MIGA=0.0132, CMIM=0.0216, ANNI=0.0234, GPNNI=0.0261, IARI=0.024,  FCMI=0.0328, DMI=0.0254, kNNI=0.0241, Mean=0.0361),
        60: dict(MIGA=0.0167, CMIM=0.0276, ANNI=0.0289, GPNNI=0.0309, IARI=0.0304, FCMI=0.0384, DMI=0.0321, kNNI=0.0316, Mean=0.0461),
    },
    "Glass": {
        30: dict(MIGA=0.0102, CMIM=0.0122, ANNI=0.0138, GPNNI=0.0179, IARI=0.0142, FCMI=0.0242, DMI=0.0176, kNNI=0.0152, Mean=0.0276),
        40: dict(MIGA=0.0137, CMIM=0.0164, ANNI=0.0177, GPNNI=0.0205, IARI=0.019,  FCMI=0.0329, DMI=0.0203, kNNI=0.0221, Mean=0.0368),
        50: dict(MIGA=0.0187, CMIM=0.021,  ANNI=0.0231, GPNNI=0.0279, IARI=0.0243, FCMI=0.0405, DMI=0.0267, kNNI=0.029,  Mean=0.0471),
        60: dict(MIGA=0.0201, CMIM=0.0235, ANNI=0.0269, GPNNI=0.0322, IARI=0.0286, FCMI=0.0491, DMI=0.0304, kNNI=0.034,  Mean=0.0555),
    },
    "Haberman": {
        30: dict(MIGA=0.0687, CMIM=0.0853, ANNI=0.0893, GPNNI=0.0926, IARI=0.0938, FCMI=0.0879, DMI=0.0988, kNNI=0.1067, Mean=0.0876),
        40: dict(MIGA=0.0962, CMIM=0.1129, ANNI=0.1129, GPNNI=0.1238, IARI=0.1234, FCMI=0.1358, DMI=0.1235, kNNI=0.1715, Mean=0.1566),
        50: dict(MIGA=0.1311, CMIM=0.1464, ANNI=0.1575, GPNNI=0.158,  IARI=0.1584, FCMI=0.1529, DMI=0.1645, kNNI=0.1715, Mean=0.1566),
        60: dict(MIGA=0.1832, CMIM=0.1766, ANNI=0.1848, GPNNI=0.1859, IARI=0.1879, FCMI=0.1827, DMI=0.1923, kNNI=0.2087, Mean=0.1844),
    },
    "Wholesale": {
        30: dict(MIGA=0.0109, CMIM=0.0211, ANNI=0.0243, GPNNI=0.0253, IARI=0.0239, FCMI=0.0271, DMI=0.0249, kNNI=0.0238, Mean=0.0341),
        40: dict(MIGA=0.0133, CMIM=0.0289, ANNI=0.0334, GPNNI=0.0357, IARI=0.0321, FCMI=0.0382, DMI=0.0342, kNNI=0.0329, Mean=0.0458),
        50: dict(MIGA=0.0267, CMIM=0.0387, ANNI=0.0432, GPNNI=0.0463, IARI=0.0423, FCMI=0.047,  DMI=0.0452, kNNI=0.0433, Mean=0.0595),
        60: dict(MIGA=0.0401, CMIM=0.0471, ANNI=0.0528, GPNNI=0.0549, IARI=0.0511, FCMI=0.0572, DMI=0.0548, kNNI=0.0527, Mean=0.0723),
    },
    "Cardio": {
        30: dict(MIGA=0.0031, CMIM=0.002,  ANNI=0.0037, GPNNI=0.0036, IARI=0.0031, FCMI=0.0107, DMI=0.0053, kNNI=0.0044, Mean=0.01),
        40: dict(MIGA=0.0037, CMIM=0.0028, ANNI=0.0051, GPNNI=0.0058, IARI=0.0043, FCMI=0.0141, DMI=0.0061, kNNI=0.0063, Mean=0.0143),
        50: dict(MIGA=0.0045, CMIM=0.0037, ANNI=0.0064, GPNNI=0.0073, IARI=0.006,  FCMI=0.0179, DMI=0.0077, kNNI=0.0084, Mean=0.0181),
        60: dict(MIGA=0.0061, CMIM=0.0066, ANNI=0.0082, GPNNI=0.009,  IARI=0.0078, FCMI=0.0219, DMI=0.0101, kNNI=0.0133, Mean=0.0222),
    },
    "Adult": {
        30: dict(MIGA=0.0097, CMIM=0.0112, ANNI=0.012,  GPNNI=0.0143, IARI=0.0129, FCMI=0.016,  DMI=0.0151, kNNI=0.0143, Mean=0.0191),
        40: dict(MIGA=0.0122, CMIM=0.0155, ANNI=0.0168, GPNNI=0.0187, IARI=0.0172, FCMI=0.0213, DMI=0.0205, kNNI=0.019,  Mean=0.0232),
        50: dict(MIGA=0.0167, CMIM=0.0191, ANNI=0.0205, GPNNI=0.0228, IARI=0.0215, FCMI=0.0267, DMI=0.026,  kNNI=0.0239, Mean=0.0285),
        60: dict(MIGA=0.0211, CMIM=0.0263, ANNI=0.025,  GPNNI=0.0279, IARI=0.0255, FCMI=0.0316, DMI=0.0309, kNNI=0.0286, Mean=0.0334),
    },
}

# ------------------------------------------------------------------
# Table 6: CoD (higher is better)
# ------------------------------------------------------------------

TABLE6_COD = {
    "Iris": {
        30: dict(MIGA=0.9911, CMIM=0.9821, ANNI=0.9783, GPNNI=0.9779, IARI=0.9777, FCMI=0.963,  DMI=0.9762, kNNI=0.9713, Mean=0.9094),
        40: dict(MIGA=0.9881, CMIM=0.9759, ANNI=0.9721, GPNNI=0.9715, IARI=0.9705, FCMI=0.9514, DMI=0.9654, kNNI=0.9629, Mean=0.8793),
        50: dict(MIGA=0.9876, CMIM=0.9703, ANNI=0.9675, GPNNI=0.9647, IARI=0.964,  FCMI=0.9345, DMI=0.9617, kNNI=0.9531, Mean=0.8542),
        60: dict(MIGA=0.9677, CMIM=0.9618, ANNI=0.9581, GPNNI=0.9574, IARI=0.9543, FCMI=0.9144, DMI=0.95,   kNNI=0.9401, Mean=0.8286),
    },
    "Wine": {
        30: dict(MIGA=0.9901, CMIM=0.9897, ANNI=0.987,  GPNNI=0.9863, IARI=0.9852, FCMI=0.9801, DMI=0.985,  kNNI=0.9831, Mean=0.9726),
        40: dict(MIGA=0.9853, CMIM=0.986,  ANNI=0.9834, GPNNI=0.9815, IARI=0.9815, FCMI=0.973,  DMI=0.9786, kNNI=0.9779, Mean=0.9648),
        50: dict(MIGA=0.9828, CMIM=0.9804, ANNI=0.9774, GPNNI=0.9744, IARI=0.9685, FCMI=0.9658, DMI=0.9722, kNNI=0.9669, Mean=0.9521),
        60: dict(MIGA=0.9781, CMIM=0.976,  ANNI=0.9731, GPNNI=0.9687, IARI=0.9685, FCMI=0.9572, DMI=0.9659, kNNI=0.9669, Mean=0.9417),
    },
    "Glass": {
        30: dict(MIGA=0.9891, CMIM=0.9891, ANNI=0.9838, GPNNI=0.9734, IARI=0.983,  FCMI=0.9578, DMI=0.9746, kNNI=0.9796, Mean=0.9537),
        40: dict(MIGA=0.9816, CMIM=0.9781, ANNI=0.9794, GPNNI=0.9566, IARI=0.9784, FCMI=0.9419, DMI=0.9574, kNNI=0.9613, Mean=0.9418),
        50: dict(MIGA=0.9798, CMIM=0.9781, ANNI=0.9767, GPNNI=0.9489, IARI=0.9726, FCMI=0.9321, DMI=0.9501, kNNI=0.9533, Mean=0.9276),
        60: dict(MIGA=0.9777, CMIM=0.9763, ANNI=0.976,  GPNNI=0.9424, IARI=0.9676, FCMI=0.92,   DMI=0.9436, kNNI=0.9478, Mean=0.9155),
    },
    "Haberman": {
        30: dict(MIGA=0.9391, CMIM=0.8959, ANNI=0.8726, GPNNI=0.8713, IARI=0.8682, FCMI=0.8822, DMI=0.8525, kNNI=0.8314, Mean=0.8842),
        40: dict(MIGA=0.9217, CMIM=0.8611, ANNI=0.8325, GPNNI=0.8208, IARI=0.8357, FCMI=0.8034, DMI=0.8034, kNNI=0.7879, Mean=0.8497),
        50: dict(MIGA=0.9002, CMIM=0.8156, ANNI=0.7977, GPNNI=0.7842, IARI=0.773,  FCMI=0.8043, DMI=0.7637, kNNI=0.7361, Mean=0.7986),
        60: dict(MIGA=0.8931, CMIM=0.7739, ANNI=0.759,  GPNNI=0.749,  IARI=0.7488, FCMI=0.7581, DMI=0.7157, kNNI=0.6693, Mean=0.7518),
    },
    "Wholesale": {
        30: dict(MIGA=0.9894, CMIM=0.9795, ANNI=0.9653, GPNNI=0.9651, IARI=0.9718, FCMI=0.9471, DMI=0.9647, kNNI=0.9694, Mean=0.9561),
        40: dict(MIGA=0.9837, CMIM=0.9723, ANNI=0.9574, GPNNI=0.9555, IARI=0.9604, FCMI=0.9257, DMI=0.9547, kNNI=0.9575, Mean=0.9408),
        50: dict(MIGA=0.9787, CMIM=0.961,  ANNI=0.9383, GPNNI=0.9383, IARI=0.9451, FCMI=0.9157, DMI=0.9375, kNNI=0.9446, Mean=0.9263),
        60: dict(MIGA=0.9697, CMIM=0.9514, ANNI=0.9253, GPNNI=0.926,  IARI=0.9318, FCMI=0.8923, DMI=0.9249, kNNI=0.9315, Mean=0.9057),
    },
    "Cardio": {
        30: dict(MIGA=0.9987, CMIM=0.9987, ANNI=0.995,  GPNNI=0.9948, IARI=0.9964, FCMI=0.9808, DMI=0.9946, kNNI=0.9941, Mean=0.9807),
        40: dict(MIGA=0.9935, CMIM=0.9972, ANNI=0.995,  GPNNI=0.9948, IARI=0.9964, FCMI=0.9808, DMI=0.9946, kNNI=0.9941, Mean=0.9841),
        50: dict(MIGA=0.9901, CMIM=0.996,  ANNI=0.9938, GPNNI=0.9929, IARI=0.9953, FCMI=0.9755, DMI=0.993,  kNNI=0.992,  Mean=0.982),
        60: dict(MIGA=0.9878, CMIM=0.9952, ANNI=0.9926, GPNNI=0.991,  IARI=0.9945, FCMI=0.9699, DMI=0.99,   kNNI=0.989,  Mean=0.9801),
    },
    "Adult": {
        30: dict(MIGA=0.9898, CMIM=0.9781, ANNI=0.976,  GPNNI=0.9735, IARI=0.9769, FCMI=0.9731, DMI=0.9737, kNNI=0.976,  Mean=0.9702),
        40: dict(MIGA=0.9793, CMIM=0.9698, ANNI=0.967,  GPNNI=0.9667, IARI=0.9684, FCMI=0.9628, DMI=0.9632, kNNI=0.967,  Mean=0.9603),
        50: dict(MIGA=0.9701, CMIM=0.9606, ANNI=0.9549, GPNNI=0.9583, IARI=0.9532, FCMI=0.9532, DMI=0.9547, kNNI=0.9582, Mean=0.9512),
        60: dict(MIGA=0.9667, CMIM=0.9516, ANNI=0.9549, GPNNI=0.9505, IARI=0.9539, FCMI=0.9458, DMI=0.9482, kNNI=0.9511, Mean=0.9424),
    },
}

# ------------------------------------------------------------------
# Convenience: all methods in order
# ------------------------------------------------------------------

METHODS = ["MIGA", "CMIM", "ANNI", "GPNNI", "IARI", "FCMI", "DMI", "kNNI"]
DATASETS = ["Iris", "Wine", "Glass", "Haberman", "Wholesale", "Cardio", "Adult"]
PERCENTAGES = [30, 40, 50, 60]
