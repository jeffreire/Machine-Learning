def factor(n):
    if n > 1:
        return factor(n - 1) * n
    else:
        return n

def binomial_x(x,p,n):
    n1 = factor(n)
    x1 = factor(x)
    nx = n1/(x1*(n-x))
    print('nx=', nx) 
    f = nx*(p**x)*((1-p)**(n-x))
    return f

def probability(p):
    return 100/p

x = int(input('Informe o total de sucesso que desejas: '))
while True:
    r = input('Voce sabe a probabilidade de {} sucessos acontecerem? (S) para sim e (N) para nao: '.format(x))
    if r == 'N':
        # obj = int(input('informe a quantidade de objetos de pesquisa: '))
        i = float('inf')
        p = probability(i)
        input('Certo, entao vamos calcular!') 
        input('A probabilidade de sucesso acontecer {} vezez é de {}%'.format(x,p))
        break
    else:
        p = float(input('informe na base 100 a probabilidade total de sucesso: '))
        break
n = int(input('informe o numero total de experimentos: '))
print('f(x) = {}%'.format(binomial_x(x,p,n)*100))