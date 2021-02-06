# parent Class
class Employee:
    count = 0

    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
# function for average Salary
    def get_avg_salary(self):
        return 'The Average Salary is {0}'.format(round(self.salary/12, 2))

# This is Child class that inherits the properties and methods of parent employee class.
class FullTimeEmployee(Employee):
    def __init__(self, name, family, salary, department, age):
        super().__init__(name, family, salary, department)
        self.age = age


emp = Employee("Srinivas", "Family one", 10000, 'CSE')
full_emp = FullTimeEmployee("Ravi", "Family two", 20000, 'ECE', 25)

print(emp.get_avg_salary())
print(full_emp.get_avg_salary())