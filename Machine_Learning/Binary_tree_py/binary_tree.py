# def print_factorial(x):
#     y = -1
#     if x < 0:
#         x = x * y
#     if x > 1:
#         return print_factorial(x + y) * x
#     elif x == 0:
#         return x
#     return x

# x = int(input('Informe um numero: '))
# print('Fatorial de {}: {}'.format(x, print_factorial(x)))

class node:
    def __init__(self, x):
        self.left = None
        self.right = None
        self.x = x
    
    def insertChild(self, y):
        if self.x != None:
            if y < self.x:
                if self.left == None:
                    self.left = node(y)
                else:
                    self.left.insertChild(y)
            elif y > self.x:
                if self.right == None:
                    self.right = node(y)
                else:
                    self.right.insertChild(y) 
        else:
            self.x = y
    
    def printTree(self):
        if self.left != None:
            self.left.printTree()  
        
        print('L: {}'.format(self.x))

        if self.right != None:
            self.right.printTree()

x = int(input('Informe o valor de x: '))
obj = node(x)
while True:
    y = int(input('Informe um valor y: '))
    obj.insertChild(y)
    r = input('Deseja continuar: S ou N')
    if r is 'S':
        True
    else:
        break
obj.printTree()
