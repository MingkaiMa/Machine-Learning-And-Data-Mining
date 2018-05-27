import matplotlib.pyplot as plt


x = [1,2,3,4]
x_1 = [2,4,6,8]
y = [1,2,3,4]
y_1 = [1,3,5,7]

plt.plot(x, x_1, label='plot1')
plt.plot(y, y_1, label='plot2')
plt.xlabel('k')
plt.ylabel('accuracy')

plt.legend()
plt.show()
