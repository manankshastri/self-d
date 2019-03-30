def gaussian_update(mean1, mean2, var1, var2):
    new_mean = (var2*mean1 + var1*mean2) / (var2 + var1)
    new_var = 1./ (1./var2 + 1./var1)
    
    return new_mean, new_var

def gaussian_predict(mean1, mean2, var1, var2):
    new_mean = mean1 + mean2
    new_var = var2 + var1
    
    return new_mean, new_var

mean, var = gaussian_update(5, 25, 8, 8)
print("Update: ",mean, var)

mean, var = gaussian_predict(5, 25, 8, 8)
print("\nPredict: ",mean, var)