
#Looking to generate # of repeated digits for 1/n as a decimal
#LCM of length of decimal period for 1/n1, 1/n2... 1/nm
#

import decimal
import math

def generate_prime(start, limit):
    """
    Returns a list of values that return number of digits
    in the repeated decimal for the number n, the index of the list

    Each index is the number n, and we want digits for 1/n

    Parameter limit: the upper bound for this generation process
    Precondition: limit is an integer greater than 0
    """
    numlist = []

    #Unfinished, REWORK
    for pos in range(start, limit + 1):
        if(check_prime(pos)):
            numlist.append(pos)
        #numlist.append(get_period(pos))
        #check_prime(numlist[-1])
    return numlist
    #return numlist

def check_prime(num):
    """
    Returns true if number follows prime pattern

    Parameter num: the number to check
    Precondition: num is a prime number
    """
    declist = get_declist(num)
    #print(declist)
    period = get_period(declist)
    print(str(num) + ': ' + str(period))
    if(period != -1):
        power2 = (num - 1) / period
        #print(power2)
        if(power_of_2(power2)):
            return True
    return False

def power_of_2(n):
    """
    Checks if a number n is a power of 2

    Parameter n: the number to check if its a power of 2
    Precondition: n is an integer
    """
    #print(n)
    if(n == 1.0):
        return True

    if(n%2 == 0):
        n = n / 2.0
        return power_of_2(n)
    return False


def get_period(declist):
    """
    Checks if dec_period is small enough

    Parameter declist: a list containing the decimals of 1/n
    Precondition: declist is a list of integers

    Returns: length of period, -1 if no period
    """
    if(declist[0] in declist[1:]):
        rep1 = declist[1:].index(declist[0]) + 1
        if(declist[0] in declist[rep1 + 1:]):
            rep2 = declist[rep1+1:].index(declist[0])+rep1+1
        else:
            return -1  #NOT PERFECT AT FINDING ALL PRIMES
        #print(rep1)
        #print(rep2)
        if(declist[:rep1] == declist[rep1:rep2]):
            return len(declist[:rep1])
        #FIX EXCEPTION WHERE PERIOD CONTAINS INTIIAL #
        while(True):
            #rep1 = rep2
            if(declist[0] in declist[rep1+1:]):
                rep1 = declist[rep1+1:].index(declist[0])+rep1+1
            else:
                return -1
            if(rep1*2 < len(declist)):
                rep2 = 2*rep1
                #print('rep1: ' + str(rep1))
                #print('rep2: ' + str(rep2))
                if(declist[:rep1] == declist[rep1:rep2]):
                    return len(declist[:rep1])
            else:
                return -1
    return -1

    #try:
        #rep_index = declist[:10] in declist[10:]
        #rep2_index = declist[:10] in declist[rep_index + 10:]
        #return rep2_index - rep1_index
    #except Exception:
        #return -1


def get_declist(num):
    """
    Finds number of digits in repeated decimal for 1/n

    Parameter num: the number n to check
    Precondition: n is a integer greater than 0 and <= limit

    Returns: the decimal list
    """
    numsused = [False] * 10
    decimal.getcontext().prec = 100
    decimals = decimal.Decimal(1) / decimal.Decimal(num)

    declist = []
    for x in range(200):
        decimals *= 10
        declist.append(int(decimals))
        decimals = decimals - declist[-1]
        if(decimals == 0.0):
            break
    return declist

def division_gen(num, limit):
    zeroesgone = False
    result = []
    while(not zeroesgone):
        next_dig = int((1/num)*10)
        if next_dig == 0:
            num /= 10
        else:
            zeroesgone = True

    for x in range(limit):
        print(1/num)
        next_dig = int((1/num)*10)
        num *= 10
        num -= next_dig
        result.append(next_dig)

    return result


print(division_gen(101, 300))
    #for pos in range(len(declist)):
        #if(numused[declist[y]] == True):
            #first_num_index = declist[:pos].index(declist[y])
            #first_rep_segment = declist[first_num_index:pos]
            #for pos in len(first_rep_segment):
