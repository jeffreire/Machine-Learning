
print('Weighted average!!!')

cumulative_average = 0
days = 1
sales_number = 2

while True:
    print('Running days {}'.format(days))
    print('Start day? Type 1 to proceed, 0 to stop') 
    option = int(input())

    if option == 0:
        break

    sales = 0
    for i in range(sales_number):
        print('Type the value of the {}º sale'.format(i+1))
        value = float(input())

        sales += value

    average_sales = sales / sales_number    

    cumulative_average = ( 1 - (1 / days)) * cumulative_average + (1 / days) * average_sales
    days += 1

print('Cumulative average: {}'.format(cumulative_average))

