def convertToBinary(number):
    binary = ''
    while number > 0:
        binary = bin(number)[2:]
        number = number // 2
    return binary

print(convertToBinary(155))