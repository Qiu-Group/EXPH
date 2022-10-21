# import time
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

def add(x,y):
    return x + y

def multi(x,y):
    return  x * y

def call_add_mul(nihao=lambda x,y:add(x,y)):
    return nihao

