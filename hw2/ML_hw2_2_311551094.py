a = int(input('Please enter a for Beta function: '))
b = int(input('Please enter b for Beta function: '))

def factorial(m):
    ans = 1
    for i in range(1, m + 1):
        ans *= i
    return ans

def C(N, m):
    if N - m < m:
        m = N - m
    
    numerator = 1
    for _ in range(m):
        numerator *= N
        N -= 1
    
    return numerator / factorial(m)

with open('testfile.txt', 'r') as file:
    i = 1
    for line in file:
        line = line.replace('\n', '')
        testcase = list(map(int,line))

        N = len(testcase)
        m = testcase.count(1)
        p = m / N
        likelihood = C(N, m) * (p ** m) * ((1-p) ** (N-m))

        print(f'\ncase {i}: {line}')
        print(f'Likelihood: {likelihood}')

        print(f'Beta prior: a = {a} b = {b}')
        a += m
        b += (N-m) 
        print(f'Beta posterior: a = {a} b = {b}', end = '\n')

        i += 1
