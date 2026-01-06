import os 
import numpy as np


arrs = os.listdir('fineweb_data')
arrs = sorted(arrs)

for arr in arrs:
    
    print(arr)
    arr = np.load(os.path.join('fineweb_data', arr))
    print(arr.min(), arr.max())
    print()