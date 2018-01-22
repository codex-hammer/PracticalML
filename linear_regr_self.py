from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
xs = np.array([1,2,3,4,5,6,7,8,9], dtype=np.float64)
ys = np.array([4,5,7,8,9,10,11,12,13], dtype=np.float64)

def best_fit_slop_intercept(xs,ys):
    m = ((mean(xs)*mean(ys)-mean(xs*ys)) / (mean(xs)*mean(xs) - mean(xs*xs)))
    b = mean(ys)-m*mean(xs)
    return m,b
def squared_error(ys_original,ys_line):
    return sum((ys_line-ys_original)**2)

def coef_of_determination(ys_original,ys_line):
    y_mean_line=[mean(ys_original) for y in ys_original]
    squared_error_reg= squared_error(ys_original,ys_line)
    squared_error_y_mean = squared_error(ys_original,y_mean_line)
    return 1-(squared_error_reg/squared_error_y_mean)

m,b=best_fit_slop_intercept(xs,ys)
reg=np.array([(m*i)+b for i in xs])
print(coef_of_determination(ys,reg))
predict_x=11
predict_y=m*predict_x + b

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='r')
plt.plot(xs,reg)
plt.show()


