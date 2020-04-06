import random 
dias = 0
media_acumulativa = 0
condicao = True
while condicao:
    dias += 1
    print('dias:{}'.format(dias))

    vendas = random.randint(1,5)
    print("Vendas: {}".format(vendas))

    valor_vendas = 0
    for i in range(vendas):
        print('Informe Valor da venda {}º'.format(i + 1))
        valor = float(input())
        
        valor_vendas += valor

    media_vendas = valor_vendas / vendas
    
    media_acumulativa = (1 - (1 / dias)) * media_acumulativa + (1 / dias) * media_vendas
    
    resposta = input('Deseja continuar calculando? ')
    if(resposta == 'nao'):
        condicao = False
else:
    print('Calculo finalizado')
    print('Media acumulativa: {}'.format(media_acumulativa))
