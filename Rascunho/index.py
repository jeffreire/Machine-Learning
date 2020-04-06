class CalcularMedia:
    def total(self, a, b, c):
        p = a + b + c
        return p
        
def calcular_media(a, b, c):
    p = a + b + c
    return p

CalcularMedia = CalcularMedia()

print('qtd numeros')
n = int(input())

nums = []
for i in range(n):
    print('Digite o {} número'.format(i + 1))
    v = int(input())
    nums.append(v)

while True:
    if v == 0:
        break

print("informe o valor 1: ")
a = int(input())
print("informe o valor 2: ")
b = int(input())
print('informe o valor 3')
c = int(input())

x = CalcularMedia.total(a, b, c)
x = calcular_media(a, b, c)

if x <= 0:
    print('Informe os dados para calcular a media')
if x > 0:
    media = x / 3
    print(media)