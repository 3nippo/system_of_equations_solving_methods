import context
import approx


X = [0, 0.5, 1, 1.5, 2]
Y = [0, 0.97943, 1.8415, 2.4975, 2.9093]

to_calc = 1.0

print(f"X = {X}")
print()
print(f"Y = {Y}")
print()
print(f"x = {to_calc}")
print()

print("First derivative in both first and second level accuracy:")
print("(from left, from right, second level accuracy)")
print(*approx.first_derivative(X, Y, to_calc))
print()
print("Second derivative:")
print(approx.second_derivative(X, Y, to_calc))
