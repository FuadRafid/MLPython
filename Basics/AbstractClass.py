from abc import ABC, abstractmethod


class Vehicle(ABC):

    def __init__(self, wheels, miles, make, model, year, sold_on):
        self.wheels = wheels
        self.miles = miles
        self.make = make
        self.model = model
        self.year = year
        self.sold_on = sold_on

    @abstractmethod
    def make_sound(self): pass


class Car(Vehicle):
    @staticmethod
    def stock_color():
        return "White"

    def make_sound(self):
        return "vroom"


car = Car(4, 1000, "Toyota", "Corolla", "2015", "2010")
print(Car.stock_color())
print(car.make_sound())
