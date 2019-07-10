import time

def fib(x):
	if x ==0:
		return 0
	
	elif x == 1:
		return 1
	else:
		return fib(x-1) + fib(x-2)

startTime = time.time()

print("%-14s:%d" % ("Result:" , fib(32)))
print("%-14s:%.4f seconds" % ("Elapsed time: ", time.time() - startTime))

