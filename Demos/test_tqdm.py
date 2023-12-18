from tqdm import tqdm,trange
import time
import math

'''
https://blog.csdn.net/weixin_44878336/article/details/124894210?spm=1001.2014.3001.5506
'''

data=[chr(i) for i in range(97,123)]
pbar=tqdm(data,
	      #desc="进度",
	      #postfix={},
	      unit="epoch",
	      ascii=True,
	      ncols=100,
	      unit_scale=False,
		  
	  )
# epoch_iter=trange(4,unit="epoch",ncols=100,leave=True)

for i,c in enumerate(pbar):
	time.sleep(1)
	print("")
	pbar.set_description(f"eee {i+1}")
	pbar.set_postfix({"loss":math.exp(len(data)-i),'learn':0.1})
	for epoch in range(4):
		print("00000")

	