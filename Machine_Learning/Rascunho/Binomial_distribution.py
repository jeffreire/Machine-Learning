def factor(n):
    if n > 1:
        return factor(n - 1) * n
    else:
        return n

def binomial_x(x,p,n):
    n_factor = factor(n)
    x_factor = factor(x)
    n_subtrai_x_factor = factor(n-x)
    nx = n_factor/(x_factor*n_subtrai_x_factor)
    print('nx=', nx)
    q = 1 - p
    f = nx*(p**x)*(q**(n-x))
    return f

def probability(p):
    return 100/p

n = int(input('Informe o valor de n: '))
while True:
    r = input('Voce sabe a probabilidade de {} sucessos acontecerem? (S) para sim e (N) para nao: '.format(n))
    if r == 'N':
        p = probability(int(input('informe a quantidade de objetos de pesquisa: ')))
        input('Certo, entao vamos calcular!') 
        input('A probabilidade de sucesso acontecer {} vezez é de {}%'.format(n,p))
        break
    else:
        p = float(input('informe p: '))
        break
x = int(input('informe n: '))
print('f(x) = {}%'.format(binomial_x(x,p,n)))