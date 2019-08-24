import numpy as np
import andcircuit
import orcircuit
import nandcircuit

def XOR(x1, x2):
    s1 = nandcircuit.NAND(x1, x2)    
    s2 = orcircuit.OR(x1, x2)
    y = andcircuit.AND(s1, s2)
    return y

if __name__ == "__main__":
    print("XOR(0,0)", XOR(0,0))
    print("XOR(1,0)", XOR(1,0))
    print("XOR(0,1)", XOR(0,1))
    print("XOR(1,1)", XOR(1,1))