import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

X = np.array([1,2,3,4,5,6,7]).reshape(-1, 1)
y = np.array([5,7,9,11,13,15,17]).reshape(-1, 1)

#print(X,y)

clf = LinearRegression().fit(X,y)
print("Coefficient : ",clf.coef_)
print("Intercept : ",clf.intercept_)
c = np.array(range(7,20)).reshape(-1,1)

d = clf.predict(c)
print("Result is :",d)
plt.plot(X,y,label= "For data X")
plt.plot(c,d,label="For linear Regression prediction d").scatter(c,d)

plt.xlabel('X')
plt.ylabel('regression')
plt.legend()
plt.show()
