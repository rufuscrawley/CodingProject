class Vector2D(object):
    """
    Helper class to store a 2D vector. Features simple implementations of
    any relevant vector operations.
    """
    x = 0.0
    y = 0.0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def multiply(self, scalar):
        self.x *= scalar
        self.y *= scalar

    def add(self, scalar):
        self.x += scalar.x
        self.y += scalar.y

    def add_mult(self, scalar):
        self.x += self.x * scalar
        self.y += self.y * scalar

    def magnitude(self):
        result = 0
        result += self.x ** 2
        result += self.y ** 2
        return result
