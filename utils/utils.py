def print_debug(*args):
    print('='*20)
    for arg in args:
        print(arg.shape, arg.dtype, arg.device)