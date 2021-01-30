num = int(input('Enter a Number: \n'))
steps = 0
while num > 0:
    if num % 2 == 0:
        num /= 2 
    else:
        num -=1 
    steps += 1
print("no of steps :" +str(steps))