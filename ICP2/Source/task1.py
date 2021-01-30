height_feet = []
height_cms = []

CONVERSION_FACTOR = 30.48
n = int(input('No.of Students: \n'))
i = 1
while i <= n:
    ht = float(input())
    height_feet.append(ht)
    i = i + 1

for i in height_feet:
    height_cms.append(round((i * CONVERSION_FACTOR), 2))

print(height_cms)