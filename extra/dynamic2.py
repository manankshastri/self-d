import time

fib_cache = {}
def fib(x):
    
    if fib_cache.get(x) != None:
        return fib_cache[x]
    
    result = None
    if x == 0:
        result = 0
    elif x == 1:
        result = 1
    else:
        result = fib(x-1) + fib(x-2)
        
    if fib_cache.get(x) == None:
        fib_cache[x] = result
        
    return result
    
startTime = time.time()

print("%-14s:%d" % ("Result:" , fib(32)))
print("%-14s:%.4f seconds" % ("Elapsed time: ", time.time() - startTime))

