'''
Author: Dan Gawne
Date  : 2021-01-13
'''

import requests
import json

raw_data = [1391.85405,1330.26226,1620.97758,1643.72307,1639.08265, 1628.47979,
            1618.05658,1415.34531,1318.10964,1614.15862,1647.36346,1651.90418,
            1636.54375,1576.93197,1382.87708,1297.21916,1578.69079,1586.4823,
            1559.68569,1520.37206,1423.23782,1272.17085,1141.7573,1111.28338,
            1130.11683,1263.94091,1299.86398,1295.08753,1215.44897,1107.11488]

figures = [[[i] for i in raw_data]]

data = json.dumps({'data' : figures, 'num_pred' : 3})

r = requests.post('http://127.0.0.1:6000/invocations',data = data)
print('Status code:', r.status_code)
print('Response:', r.content)