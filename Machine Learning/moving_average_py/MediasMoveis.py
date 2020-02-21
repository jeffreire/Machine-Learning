# print('hello world')


# class C1:

#     def __init__(self, initial_name):
#         self.name = initial_name

#     def m1(self, new_name):
#         self.name = new_name

#     def soma(self, a, b):
#         r = a + b
#         return r


# c1 = C1('Jefferson')
# r = c1.soma(1.6, 2)
# print(r)


# a = [[1,2],[3,6]]
# # a.append(1)
# # a.append('s')


# print(a[0][0])

# print('Digite o 1º número')
# n1 = float(input())
# print('Digite o 2º número')
# n2 = float(input())
# r = n1 + n2
# print(r)


# teste = 0 if 1 == 1 else 2

# teste_lista = [i for i in range(15, 10, -2)]
# print(teste, teste_lista)


quantidade = 5
dados = []
condicao = True
while condicao:
    numero = float(input('Informe os numeros para calcular a media: '))
    dados.append(numero)
    
    if(len(dados) > 5):
        print(dados[-5:])
    else:
        print(dados)
    
    if quantidade > len(dados):
        print('\33[1;35m Informe mais {} numeros para calcular\33[m a média: ' .format(quantidade - len(dados)))
    else:
        print('Media: {}'.format(sum(dados[-5:]) / quantidade))

        resposta = input('Voce deseja continuar? Digite NAO para sair ')
        if resposta == 'nao':
            condicao = False
        else:
            condicao = True
else:
    print("Calculo encerrado") 