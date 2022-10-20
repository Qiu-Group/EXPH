import time

start_time = time.time()
a = list(range(60000000))
# a = list(range(30000))
res = 0
for i in a:
    res += i


end_time = time.time()

print("the running time is: %.3f s"%(end_time - start_time))
print(res)