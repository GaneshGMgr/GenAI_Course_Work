# Regular function: returns a list all at once
def get_numbers():
    return [1, 2, 3]  # All values are returned at once

print("Regular function outputs:", get_numbers())

# Generator function: yields one value at a time
def generate_numbers():
    for i in range(1, 4):
        yield i  # Pauses here and resumes on next() call

gen = generate_numbers()  # gen is a generator object

# Use next() to get values one-by-one
print("Generator outputs:")
print(next(gen))  # ➜ 1
print(next(gen))  # ➜ 2
# print(next(gen))  # ➜ 3

# Decorator function: A decorator is a function that takes another function as input and returns a new function with extra behavior.
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"📞 Calling: {func.__name__} function")
        result = func(*args, **kwargs) # Call the original greet() function
        print(f"✅ Finished: {func.__name__} function")
        return result
    return wrapper

@log_calls
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")

