import numpy as np
import pandas as pd

training_data=[]
for i in range(1,1000001):
	if i%2 == 0:
		training_data.append([i,0])
	else:
		training_data.append([i,1])

df = pd.DataFrame(training_data,columns=['num','label'])
df.to_csv('odd_even_data.csv',index=False)
