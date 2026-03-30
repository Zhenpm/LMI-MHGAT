import numpy as np
import matplotlib.pyplot as plt

# a stacked bar plot with errorbars
#N = 2

ind = 5#np.arange(N)    # the x locations for the groups
width = 0.25       # the width of the bars: can also be len(x) sequence

model1=(0.8938,0.7826)
model1_std=(float(1.3523935077182842e-03),float(4.87248074457254e-03))
model2=(0.9002,0.7945)
model2_std=(float(8.133014133584965e-03),float(2.8909096521415685e-03))
model3=(0.9114,0.8151)
model3_std=(float(2.1527113676882603e-03),float(5.6515334195772994e-03))
model4=(0.9041,0.7995)
model4_std=(float(6.14032672281501e-03),1.479384494990171e-03)
model5=(0.8981,0.7926)
model5_std=(float(8.986192396251911e-03),float(2.8849539504412527e-03))

p1 = plt.bar((ind-10*width,ind-9*width), model1, width, facecolor='#8ac6d1', edgecolor='#FFFFFF', yerr=model1_std, capsize=3,label="NS=5")
p2 = plt.bar((ind-7*width,ind-6*width), model2, width, facecolor='#bbded6', edgecolor='#FFFFFF',yerr=model2_std, capsize=3,label="NS=10")
p3 = plt.bar((ind-4*width,ind-3*width), model3, width, facecolor='#fae3d9', edgecolor='#FFFFFF', yerr=model3_std, capsize=3,label="NS=20")
p4 = plt.bar((ind-1*width,ind), model4, width, facecolor='#ffe3b0', edgecolor='#FFFFFF', yerr=model4_std, capsize=3,label="NS=30")
p5 = plt.bar((ind+2*width,ind+3*width), model5, width, facecolor='#51eaea', edgecolor='#FFFFFF', yerr=model5_std, capsize=3,label="NS=40")

fig = plt.gcf()
fig.set_size_inches(6,5)
plt.ylim([0.775, 0.925])
plt.ylabel('value')

plt.title('Negative sample(NS)')
plt.xticks((ind-10*width,ind-9*width,ind-7*width,ind-6*width,ind-4*width,ind-3*width,ind-1*width,ind,ind+2*width,ind+3*width),
('PR-AUC', 'F1-score','PR-AUC', 'F1-score','PR-AUC', 'F1-score','PR-AUC', 'F1-score','PR-AUC', 'F1-score'))
plt.xticks((ind-9.5*width,ind-6.5*width,ind-3.5*width,ind-0.5*width,ind+2.5*width),
('NS=5','NS=10','NS=20','NS=30','NS=40'))
plt.grid(True, linestyle='-', color = 'black', linewidth = '0.05')

plt.show()

plt.savefig("filename.png")
