import time
#
# if __name__ =="__main__":
#
#     start_time = time.time()
#     a = list(range(60000000))
#     # a = list(range(30000))
#     res = 0
#     for i in a:
#         res += i
#
#
#     end_time = time.time()
#
#     print("the running time is: %.3f s"%(end_time - start_time))
#     print(res)
start_time = time.time()
res= 0
res1 = 0
res2 = 0
for i in range(144):
    for j in range(144):
        for k in range(144):
            res += 1
            res1 += 1
            res2 += 1

end_time = time.time()

print("the running time is: %.3f s"%(end_time - start_time))
print(res)