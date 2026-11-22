# fib.py
# Print the first 10 Fibonacci numbers (starting from 0)

a, b = 0, 1
for _ in range(10):
    print(a)
    a, b = b, a + b
