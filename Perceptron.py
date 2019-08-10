import math
import random

class Perceptron:
    def __init__(self, rate=1, func=None, der_func=None):
        self.rate = rate
        if func is None:
            self.activate = self.LeakyReLu
            self.der_activate = self.der_LeakyReLu
        else:
            self.activate = func
            self.der_activate = der_func

        self.weights = [random.random(), random.random(), random.random()]
        self.inputs = [0, 0, 1]
        self.output = 0

    def __str__(self):
        return str(self.weights)

    def __call__(self, a, b, t, debug=False):
        self.inputs[0] = a
        self.inputs[1] = b
        self.output = self.f()

        if debug:
            print(f"{a: >3}, {b: >3} => {self.output: <24} | {self.loss(t)}")

        delta = [0, 0, 0]
        for i in range(len(self.weights)):
            delta[i] = -self.rate*self.der_loss(t, i)

        for i in range(len(delta)):
            self.weights[i] += delta[i]

        return self.output

    def f(self):
        # f() = act(act(I_1)*w_1+act(I_2)*w_2+act(1)*w_3)
        x = 0
        for i in range(len(self.inputs)):
            x += self.activate(self.inputs[i])*self.weights[i]

        return self.activate(x)

    def pder_f(self, n):
        # df/dw = der_activate(act(I_1)*w_1+act(I_2)*w_2+act(1)*w_3)*(act(I_n)*1 + der_activate(I_n)*w_n*0)
        x = 0
        for i in range(len(self.inputs)):
            x += self.activate(self.inputs[i])*self.weights[i]

        return self.der_activate(x)*self.activate(self.inputs[n])

    def loss(self, t):
        return (t-self.output)*(t-self.output)/2

    def der_loss(self, t, n):
        return (t-self.output)*-self.pder_f(n)

    @staticmethod
    def LeakyReLu(x):
        if x > 0:
            return x
        return 0.01*x

    @staticmethod
    def der_LeakyReLu(x):
        if x > 0:
            return 1
        return 0.01

    @staticmethod
    def Sigmoid(x):
        return 1/(1+math.exp(-x))

    @staticmethod
    def der_Sigmoid(x):
        return math.exp(-x)/math.pow(1+math.exp(-x), 2)

    @staticmethod
    def tanh(x):
        return 2/(1+math.exp(-2*x)) - 1

    @staticmethod
    def der_tanh(x):
        return 1 - math.pow(Perceptron.tanh(x), 2)


if __name__ == '__main__':
    random.seed(0)

    print("OR Function:")
    orPtron = Perceptron(1, func=Perceptron.tanh, der_func=Perceptron.der_tanh)
    for i in range(100000):
        a = random.randint(0, 1)
        b = random.randint(0, 1)
        orPtron(a, b, a|b)
    orPtron(0, 0, 0, debug=True)
    orPtron(0, 1, 1, debug=True)
    orPtron(1, 0, 1, debug=True)
    orPtron(1, 1, 1, debug=True)
    print(orPtron)

    print("\nAND Function:")
    andPtron = Perceptron(10, func=Perceptron.Sigmoid, der_func=Perceptron.der_Sigmoid)
    for i in range(100000):
        a = random.randint(0, 1)
        b = random.randint(0, 1)
        andPtron(a, b, a&b)
    andPtron(0, 0, 0, debug=True)
    andPtron(0, 1, 0, debug=True)
    andPtron(1, 0, 0, debug=True)
    andPtron(1, 1, 1, debug=True)
    print(andPtron)

    print("\nNOT Function:")
    notPtron = Perceptron(10, func=Perceptron.Sigmoid, der_func=Perceptron.der_Sigmoid)
    for i in range(100000):
        a = random.randint(0, 1)
        if a == 0:
            notPtron(a, 0, 1)
        else:
            notPtron(a, 0, 0)
    notPtron(0, 0, 1, debug=True)
    notPtron(1, 0, 0, debug=True)
    print(notPtron)

    print("\nADD Function:")
    addPtron = Perceptron(0.01)
    for i in range(100000):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        addPtron(a, b, a+b)
    addPtron(18, 21, 18+21, debug=True)
    print(addPtron)
