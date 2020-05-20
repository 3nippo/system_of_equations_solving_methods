import context
import approx


X = [0, 0.1, 0.2, 0.3, 0.4]
Y = [1, 1.1052, 1.2214, 1.3499, 1.4918]

to_calc = 0.2

print("First derivative in both first and second level accuracy:")
print("(from left, from right, second level accuracy)")
print(*approx.first_derivative(X, Y, to_calc))
print()
print("Second derivative:")
print(approx.second_derivative(X, Y, to_calc))
